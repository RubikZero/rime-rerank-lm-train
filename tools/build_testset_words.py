"""Build a small word-level IME re-ranking test set.

Targets are 1-4 character words from the Rime preset vocabulary (essay.txt).
Candidates are same-pinyin words of the same length, built with pypinyin
(e.g. code "shi jian" -> 时间 / 实践 / 事件 / ...). Context is up to 64 chars
before the target in a sentence from the external test corpora. Strict
deduplication against the training corpus is applied as in build_testset.py.
"""

import argparse
import io
import os
import random
import subprocess
import sys
from collections import defaultdict

import zhconv
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_testset import (CORPUS_TEST, GREP, OUT_DIR, TRAIN_CORPUS,  # noqa
                           iter_lccc, iter_lcsts, iter_thucnews, iter_wiki,
                           load_vocab_chars)

HERE = os.path.dirname(os.path.abspath(__file__))
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"

SOURCES = [
    ("wiki", os.path.join(CORPUS_TEST, "wiki.json"), iter_wiki),
    ("lccc", os.path.join(CORPUS_TEST, "lccc_valid.jsonl.gz"), iter_lccc),
    ("lccc", os.path.join(CORPUS_TEST, "lccc_train.jsonl.gz"), iter_lccc),
    ("lcsts", os.path.join(CORPUS_TEST, "lcsts_valid.jsonl"), iter_lcsts),
    ("lcsts", os.path.join(CORPUS_TEST, "lcsts_train.jsonl"), iter_lcsts),
    ("thucnews", os.path.join(CORPUS_TEST, "thucnews_keji.jsonl"),
     iter_thucnews),
    ("thucnews", os.path.join(CORPUS_TEST, "thucnews_shehui.jsonl"),
     iter_thucnews),
]


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
        (0xF900 <= o <= 0xFAFF)


def build_groups(vocab, min_size, max_len):
    words = set()
    with io.open(ESSAY, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                words.add(parts[0])
    groups = defaultdict(lambda: defaultdict(set))
    sim2py = {}
    for w in words:
        if not (1 <= len(w) <= max_len):
            continue
        if not all(is_han(c) for c in w):
            continue
        s = zhconv.convert(w, "zh-cn")
        if len(s) != len(w) or not all(is_han(c) for c in s):
            continue
        if not all(c in vocab for c in s):
            continue
        py = " ".join(lazy_pinyin(w, style=Style.NORMAL, errors="ignore"))
        if not py:
            continue
        groups[py][len(s)].add(s)
        sim2py.setdefault(s, py)
    keep = {}
    for py, by_len in groups.items():
        for length, ws in by_len.items():
            if len(ws) >= min_size:
                keep.setdefault(length, {})[py] = sorted(ws)
    return keep, sim2py


def make_cases(sources, keep, sim2py, word_sets, prefix_sets, rng, quotas,
               oversample, ctx):
    caps = {length: quota * oversample for length, quota in quotas.items()}
    domains = sorted(set(d for d, _, _ in sources))
    per_domain_cap = {L: max(2, caps[L] // len(domains)) for L in quotas}
    buckets = defaultdict(list)
    counts = defaultdict(int)
    seen_targets = defaultdict(set)
    seen_cases = set()
    for domain, path, reader in sources:
        if not os.path.exists(path):
            continue
        if all(len(buckets[L]) >= caps.get(L, 0) for L in quotas):
            break
        if all(counts[(domain, L)] >= per_domain_cap[L] for L in quotas):
            continue
        for sent in reader(path):
            if all(len(buckets[L]) >= caps.get(L, 0) for L in quotas):
                break
            found = None
            for i in range(len(sent)):
                for length in (4, 3, 2, 1):
                    if length not in quotas:
                        continue
                    if len(buckets[length]) >= caps[length]:
                        continue
                    if counts[(domain, length)] >= per_domain_cap[length]:
                        continue
                    if i + length > len(sent):
                        continue
                    if length >= 2 and sent[i:i + 2] not in prefix_sets[length]:
                        continue
                    w = sent[i:i + length]
                    if w in word_sets[length]:
                        found = (i, length, w)
                        break
                if found:
                    break
            if not found:
                continue
            i, length, target = found
            py = sim2py.get(target)
            if py is None:
                continue
            pool = [w for w in keep[length].get(py, []) if w != target]
            if len(pool) < 4:
                continue
            if target in seen_targets[length] and len(seen_targets[length]) < 50:
                continue
            seen_targets[length].add(target)
            context = sent[max(0, i - ctx):i]
            if len(context) < 8:
                continue
            if (context, target) in seen_cases:
                continue
            seen_cases.add((context, target))
            rng.shuffle(pool)
            cands = [target] + pool[:4]
            rng.shuffle(cands)
            buckets[length].append({
                "domain": domain, "context": context, "target": target,
                "candidates": cands, "length": length, "code": py})
            counts[(domain, length)] += 1
    return buckets


def strict_dedup(cases, spans_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    spans_path = os.path.join(OUT_DIR, spans_name)
    with io.open(spans_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c["context"] + c["target"] + "\n")
    print("strict dedup grep ...", flush=True)
    proc = subprocess.run([GREP, "-F", "-h", "-o", "-f", spans_path,
                           TRAIN_CORPUS], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    matched = set(x for x in proc.stdout.decode("utf-8", "ignore").splitlines()
                  if x)
    matched_list = sorted(matched, key=len, reverse=True)

    def contaminated(span):
        if span in matched:
            return True
        for m in matched_list:
            if len(m) < len(span):
                break
            if span in m:
                return True
        return False

    kept = [c for c in cases if not contaminated(c["context"] + c["target"])]
    print("grep exit=%d matched=%d kept=%d/%d" %
          (proc.returncode, len(matched), len(kept), len(cases)), flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota1", type=int, default=200)
    ap.add_argument("--quota2", type=int, default=300)
    ap.add_argument("--quota3", type=int, default=100)
    ap.add_argument("--quota4", type=int, default=50)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    quotas = {1: args.quota1, 2: args.quota2, 3: args.quota3, 4: args.quota4}
    rng = random.Random(args.seed)
    vocab = load_vocab_chars(VOCAB)
    keep, sim2py = build_groups(vocab, min_size=5, max_len=4)
    for L in sorted(keep):
        print("length %d: %d code groups" % (L, len(keep[L])), flush=True)

    word_sets = {L: set(w for ws in keep[L].values() for w in ws)
                 for L in keep}
    prefix_sets = {L: set(w[:2] for w in word_sets[L] if len(w) >= 2)
                   for L in word_sets}

    buckets = make_cases(SOURCES, keep, sim2py, word_sets, prefix_sets, rng,
                         quotas, args.oversample, args.ctx)
    all_cases = []
    for L in sorted(buckets):
        all_cases.extend(buckets[L])
        print("candidates length %d: %d" % (L, len(buckets[L])), flush=True)
    rng.shuffle(all_cases)
    kept = strict_dedup(all_cases, "spans_words.txt")

    final = []
    by_len = defaultdict(list)
    for c in kept:
        by_len[c["length"]].append(c)
    for L in sorted(by_len):
        rng.shuffle(by_len[L])
        take = by_len[L][:quotas[L]]
        final.extend(take)
        print("final length %d: %d" % (L, len(take)), flush=True)
    rng.shuffle(final)
    for i, c in enumerate(final):
        c["id"] = i

    out = os.path.join(OUT_DIR, "testset_words.json")
    with io.open(out, "w", encoding="utf-8") as f:
        import json
        json.dump({"cases": final,
                   "stats": {"total": len(final)}}, f, ensure_ascii=False,
                  indent=1)
    print("saved", out, "total", len(final), flush=True)


if __name__ == "__main__":
    main()
