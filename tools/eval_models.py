"""Offline evaluation of small Chinese BERT models for IME candidate re-ranking.

Test design: take multi-character words (3-6 chars) from the Rime luna_pinyin
dictionary. Use the word prefix as the "committed context" and the final
character as the target. Candidates are the target plus same-pinyin
homophones. Score each candidate with masked-LM pseudo log-likelihood and see
whether the target ranks first. Also report a character-frequency baseline.
"""

import io
import json
import os
import random
import sys
import time

import torch
import zhconv
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = r"D:\Workspace\rime\third-party\models"
DATA = r"D:\Workspace\rime\weasel\output\data"
OUT = r"D:\Workspace\rime\third-party\lm-test\results.json"
N_CASES = 200
GROUP_SIZE = 5

MODELS = [
    ("uer_L2_H128", os.path.join(BASE, "uer", "chinese_roberta_L-2_H-128")),
    ("uer_L4_H256", os.path.join(BASE, "uer", "chinese_roberta_L-4_H-256")),
    ("ckip_albert_tiny", os.path.join(BASE, "ckiplab", "albert-tiny-chinese")),
    ("ckip_bert_tiny", os.path.join(BASE, "ckiplab", "bert-tiny-chinese")),
]


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or (0xF900 <= o <= 0xFAFF)


def load_vocab_chars(path):
    chars = set()
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) == 1 and is_han(line):
                chars.add(line)
    return chars


def parse_dict(path):
    words = {}
    with io.open(path, encoding="utf-8") as f:
        started = False
        for line in f:
            line = line.rstrip("\n")
            if not started:
                if line.strip() == "...":
                    started = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].strip()
            if not w:
                continue
            words.setdefault(w, set()).update(parts[1].split())
    return words


def parse_freq(path):
    freq = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    freq[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return freq


def build_cases(dict_words, vocab_chars, n=N_CASES, group_size=GROUP_SIZE):
    rng = random.Random(42)
    code2chars = {}
    for w, codes in dict_words.items():
        if len(w) == 1 and w in vocab_chars:
            for c in codes:
                code2chars.setdefault(c, set()).add(w)
    groups = {c: sorted(chars) for c, chars in code2chars.items()
              if len(chars) >= group_size}

    long_words = [w for w in dict_words
                  if 3 <= len(w) <= 6
                  and all(ch in vocab_chars for ch in w)
                  and w[-1] in dict_words]
    rng.shuffle(long_words)

    cases = []
    seen = set()
    for w in long_words:
        if len(cases) >= n:
            break
        target = w[-1]
        code = next((c for c in sorted(dict_words[target]) if c in groups), None)
        if code is None:
            continue
        pool = [ch for ch in groups[code] if ch != target]
        rng.shuffle(pool)
        distract = pool[:group_size - 1]
        if len(distract) < group_size - 1:
            continue
        context = w[:-1]
        if (context, target) in seen:
            continue
        seen.add((context, target))
        cands = [target] + distract
        rng.shuffle(cands)
        cases.append({"context": context, "target": target,
                      "candidates": cands, "code": code})
    return cases


def to_simplified(cases):
    out = []
    for c in cases:
        ctx = zhconv.convert(c["context"], "zh-cn")
        tgt = zhconv.convert(c["target"], "zh-cn")
        cands = []
        for x in c["candidates"]:
            y = zhconv.convert(x, "zh-cn")
            if y not in cands:
                cands.append(y)
        if tgt not in cands or len(cands) < 2:
            continue
        out.append({"context": ctx, "target": tgt, "candidates": cands,
                    "code": c["code"]})
    return out


def eval_frequency(cases, freq):
    hits1 = hits3 = 0
    mrr = 0.0
    for case in cases:
        ranked = sorted(case["candidates"], key=lambda c: -freq.get(c, 0.0))
        rank = ranked.index(case["target"]) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        mrr += 1.0 / rank
    n = len(cases)
    return {"top1": hits1 / n, "top3": hits3 / n, "mrr": mrr / n}


def eval_model(name, path, cases, transform=None):
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    model.eval()
    load_s = time.perf_counter() - t0

    unk_id = tok.unk_token_id
    mask_id = tok.mask_token_id
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    if transform is None:
        transform = lambda s: s

    def score(context, cand):
        ctx_ids = tok.encode(transform(context), add_special_tokens=False)
        cand_ids = [tok.convert_tokens_to_ids(ch) for ch in transform(cand)]
        if any(i in (unk_id, None) for i in cand_ids):
            return None
        ids = [cls_id] + ctx_ids + [mask_id] * len(cand_ids) + [sep_id]
        start = 1 + len(ctx_ids)
        with torch.inference_mode():
            logits = model(torch.tensor([ids])).logits[0]
            logp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for k, cid in enumerate(cand_ids):
            total += float(logp[start + k, cid])
        return total

    hits1 = hits3 = 0
    mrr = 0.0
    lat = []
    per_case = []
    skipped = 0
    for case in cases:
        scored = []
        seen_tf = set()
        ambiguous = False
        for cand in case["candidates"]:
            tf = transform(cand)
            if tf in seen_tf:
                ambiguous = True
                break
            seen_tf.add(tf)
            t1 = time.perf_counter()
            s = score(case["context"], cand)
            lat.append(time.perf_counter() - t1)
            if s is not None:
                scored.append((s, cand))
        if ambiguous or len(scored) != len(case["candidates"]):
            skipped += 1
            continue
        scored.sort(key=lambda x: -x[0])
        order = [c for _, c in scored]
        rank = order.index(case["target"]) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        mrr += 1.0 / rank
        per_case.append({"context": case["context"], "target": case["target"],
                         "rank": rank, "order": order})
    n = len(per_case)
    return {
        "model": name,
        "cases": n,
        "skipped": skipped,
        "top1": hits1 / n,
        "top3": hits3 / n,
        "mrr": mrr / n,
        "load_s": round(load_s, 2),
        "avg_score_ms": round(1000 * sum(lat) / max(1, len(lat)), 2),
        "samples": per_case[:20],
    }


def main():
    simp = "--simp" in sys.argv
    crosssimp = "--crosssimp" in sys.argv
    vocab_chars = load_vocab_chars(os.path.join(MODELS[0][1], "vocab.txt"))
    dict_words = parse_dict(os.path.join(DATA, "luna_pinyin.dict.yaml"))
    freq = parse_freq(os.path.join(DATA, "essay.txt"))
    cases = build_cases(dict_words, vocab_chars)
    transform = None
    out = OUT
    if simp:
        cases = to_simplified(cases)
        merged = {}
        for k, v in freq.items():
            merged[zhconv.convert(k, "zh-cn")] = merged.get(
                zhconv.convert(k, "zh-cn"), 0.0) + v
        freq = merged
        out = OUT.replace("results.json", "results_simp.json")
    elif crosssimp:
        transform = lambda s: zhconv.convert(s, "zh-cn")
        out = OUT.replace("results.json", "results_crosssimp.json")
    print("cases=%d vocab_chars=%d dict_words=%d simp=%s cross=%s" %
          (len(cases), len(vocab_chars), len(dict_words), simp, crosssimp))

    results = {"baseline_frequency": eval_frequency(cases, freq), "models": []}
    b = results["baseline_frequency"]
    print("baseline frequency: top1=%.3f top3=%.3f mrr=%.3f" %
          (b["top1"], b["top3"], b["mrr"]))

    models = list(MODELS)
    for i, a in enumerate(sys.argv):
        if a == "--extra" and i + 1 < len(sys.argv):
            name, _, path = sys.argv[i + 1].partition("=")
            models.append((name, path))

    for name, path in models:
        r = eval_model(name, path, cases, transform)
        results["models"].append(r)
        print("%-18s top1=%.3f top3=%.3f mrr=%.3f load=%.1fs score=%.2fms skip=%d" %
              (name, r["top1"], r["top3"], r["mrr"], r["load_s"],
               r["avg_score_ms"], r["skipped"]))
        del r["samples"]

    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("written:", out)


if __name__ == "__main__":
    main()
