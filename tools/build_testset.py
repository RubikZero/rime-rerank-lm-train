"""Build a domain-balanced IME re-ranking test set from external corpora.

Sources: Chinese Wikipedia, LCCC, LCSTS, THUCNews. Each case uses a sentence
from one corpus: context = up to 64 chars before a target character, target =
that character, candidates = the target plus same-pinyin homophones.

Strict deduplication: every case's `context + target` span is searched in the
whole CLUECorpusSmall training corpus (git grep -F -f); matches are dropped.
"""

import argparse
import gzip
import io
import json
import os
import random
import re
import subprocess
import sys

import zhconv

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_TEST = r"D:\Workspace\rime\third-party\corpus-test"
TRAIN_CORPUS = r"D:\Workspace\rime\third-party\corpus\CLUECorpusSmall.txt"
DICT = r"D:\Workspace\rime\weasel\output\data\luna_pinyin.dict.yaml"
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
OUT_DIR = os.path.join(HERE, "testset")
GREP = r"D:\Software\Git\usr\bin\grep.exe"

SENT_SPLIT = re.compile(r"[。！？!?；;\n\r]+")
HAN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def is_han(ch):
    return bool(HAN.match(ch))


def load_vocab_chars(path):
    chars = set()
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if len(tok) == 1 and is_han(tok):
                chars.add(tok)
    return chars


def load_homophones(dict_path):
    char2codes = {}
    code2chars = {}
    with io.open(dict_path, encoding="utf-8") as f:
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
            word = parts[0].strip()
            if len(word) != 1 or not is_han(word):
                continue
            simp = zhconv.convert(word, "zh-cn")
            if len(simp) != 1 or not is_han(simp):
                continue
            for code in parts[1].split():
                code2chars.setdefault(code, set()).add(simp)
    groups = {c: sorted(v) for c, v in code2chars.items() if len(v) >= 5}
    for code, chars in groups.items():
        for ch in chars:
            char2codes.setdefault(ch, set()).add(code)
    return char2codes, groups


def chinese_ratio(s):
    if not s:
        return 0.0
    return sum(1 for c in s if is_han(c)) / len(s)


def sentences_from_text(text):
    for piece in SENT_SPLIT.split(text):
        piece = piece.strip()
        if 20 <= len(piece) <= 200 and chinese_ratio(piece) >= 0.7:
            yield piece


def iter_wiki(path):
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item, dict):
            text = item.get("completion") or item.get("text") or ""
        elif isinstance(item, str):
            text = item
        else:
            continue
        for sent in sentences_from_text(text):
            yield sent


def iter_lccc(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, list):
                continue
            for turn in obj:
                if not isinstance(turn, str):
                    continue
                text = turn.replace(" ", "")
                for sent in sentences_from_text(text):
                    yield sent


def iter_lcsts(path):
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or ""
            for sent in sentences_from_text(text):
                yield sent


def iter_thucnews(path):
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sent in sentences_from_text(obj.get("title", "")):
                yield sent
            for sent in sentences_from_text(obj.get("content", "")):
                yield sent


DOMAINS = {
    "wiki": (os.path.join(CORPUS_TEST, "wiki.json"), iter_wiki),
    "lccc": (os.path.join(CORPUS_TEST, "lccc_valid.jsonl.gz"), iter_lccc),
    "lccc_train": (os.path.join(CORPUS_TEST, "lccc_train.jsonl.gz"), iter_lccc),
    "lcsts": (os.path.join(CORPUS_TEST, "lcsts_valid.jsonl"), iter_lcsts),
    "lcsts_test": (os.path.join(CORPUS_TEST, "lcsts_test.jsonl"), iter_lcsts),
    "lcsts_train": (os.path.join(CORPUS_TEST, "lcsts_train.jsonl"), iter_lcsts),
    "thucnews": (os.path.join(CORPUS_TEST, "thucnews_keji.jsonl"), iter_thucnews),
    "thucnews2": (os.path.join(CORPUS_TEST, "thucnews_shehui.jsonl"), iter_thucnews),
}
GROUP_TO_DOMAIN = {
    "wiki": "wiki",
    "lccc": "lccc",
    "lccc_train": "lccc",
    "lcsts": "lcsts",
    "lcsts_test": "lcsts",
    "lcsts_train": "lcsts",
    "thucnews": "thucnews",
    "thucnews2": "thucnews",
}


def make_cases(sentences, char2codes, groups, vocab, rng, want, ctx):
    cases = []
    seen = set()
    for sent in sentences:
        if len(cases) >= want:
            break
        hits = [i for i, ch in enumerate(sent) if ch in char2codes and ch in vocab]
        if not hits:
            continue
        i = hits[rng.randrange(len(hits))]
        target = sent[i]
        codes = [c for c in char2codes[target] if target in groups[c]]
        if not codes:
            continue
        code = codes[rng.randrange(len(codes))]
        pool = [c for c in groups[code] if c != target and c in vocab]
        if len(pool) < 4:
            continue
        rng.shuffle(pool)
        cands = [target] + pool[:4]
        rng.shuffle(cands)
        context = sent[max(0, i - ctx):i]
        if len(context) < 8:
            continue
        if (context, target) in seen:
            continue
        seen.add((context, target))
        cases.append({"domain": None, "context": context, "target": target,
                      "candidates": cands, "code": code})
    return cases


def strict_dedup(cases):
    os.makedirs(OUT_DIR, exist_ok=True)
    spans_path = os.path.join(OUT_DIR, "spans.txt")
    with io.open(spans_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c["context"] + c["target"] + "\n")
    print("running strict dedup grep over training corpus ...", flush=True)
    proc = subprocess.run(
        [GREP, "-F", "-h", "-o", "-f", spans_path, TRAIN_CORPUS],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    matched = set()
    for line in proc.stdout.decode("utf-8", "ignore").splitlines():
        if line:
            matched.add(line)
    print("grep exit=%d matched_spans=%d" % (proc.returncode, len(matched)),
          flush=True)
    matched_list = [m for m in matched if m]
    matched_list.sort(key=len, reverse=True)

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
    return kept, len(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=1250)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocab_chars(VOCAB)
    char2codes, groups = load_homophones(DICT)
    print("vocab=%d chars_with_codes=%d groups=%d" %
          (len(vocab), len(char2codes), len(groups)), flush=True)

    all_cases = []
    for name, (path, reader) in DOMAINS.items():
        if not os.path.exists(path):
            print("skip missing", path)
            continue
        want = args.per_domain * args.oversample
        cases = make_cases(reader(path), char2codes, groups, vocab, rng, want,
                           args.ctx)
        domain = GROUP_TO_DOMAIN[name]
        for c in cases:
            c["domain"] = domain
        all_cases.extend(cases)
        print("domain=%s candidates=%d" % (domain, len(cases)), flush=True)

    rng.shuffle(all_cases)
    kept, n_matched = strict_dedup(all_cases)
    print("after dedup: %d / %d (dropped %d)" %
          (len(kept), len(all_cases), len(all_cases) - len(kept)), flush=True)

    by_domain = {}
    for c in kept:
        by_domain.setdefault(c["domain"], []).append(c)
    final = []
    for domain, cases in sorted(by_domain.items()):
        rng.shuffle(cases)
        take = cases[:args.per_domain]
        final.extend(take)
        print("domain=%-9s final=%d" % (domain, len(take)), flush=True)
    rng.shuffle(final)
    for i, c in enumerate(final):
        c["id"] = i

    out = os.path.join(OUT_DIR, "testset.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": final,
                   "stats": {"total": len(final),
                             "dropped_by_dedup": n_matched}},
                  f, ensure_ascii=False, indent=1)
    print("saved", out, "total", len(final), flush=True)


if __name__ == "__main__":
    main()
