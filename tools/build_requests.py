"""Generate real-IME extraction requests from the external test corpora.

Buckets: single characters (keys = syllable of the char inside its word to get
the right polyphone reading) and 2-4 char words (keys = full word pinyin).
Context is the corpus prefix. Strict dedup against the training corpus.
"""

import argparse
import io
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import zhconv
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_testset import (CORPUS_TEST, GREP, TRAIN_CORPUS, iter_lccc,  # noqa
                           iter_lcsts, iter_thucnews, iter_wiki,
                           load_vocab_chars)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "testset")
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


def load_common_words(vocab, top_n):
    items = []
    with io.open(ESSAY, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                freq = float(parts[1])
            except ValueError:
                continue
            w = zhconv.convert(parts[0], "zh-cn")
            if 2 <= len(w) <= 4 and all(is_han(c) for c in w) and \
                    all(c in vocab for c in w):
                items.append((freq, w))
    items.sort(reverse=True)
    return [w for _, w in items[:top_n]]


def clean_keys(syllables):
    keys = []
    for s in syllables:
        s = s.replace("ü", "v")
        s = "".join(c for c in s if "a" <= c <= "z")
        if not s:
            return None
        keys.append(s)
    return "".join(keys)


def make_cases(sources, common_words, rng, quotas, oversample, ctx):
    word_set = set(common_words)
    prefix2 = set(w[:2] for w in common_words)
    caps = {kind: q * oversample for kind, q in quotas.items()}
    domains = sorted(set(d for d, _, _ in sources))
    per_domain = {kind: max(2, caps[kind] // len(domains))
                  for kind in quotas}
    cases = []
    counts = defaultdict(int)
    seen = set()
    for domain, path, reader in sources:
        if not os.path.exists(path):
            continue
        if all(counts[k] >= caps[k] for k in quotas):
            break
        if all(counts[(domain, k)] >= per_domain[k] for k in quotas):
            continue
        for sent in reader(path):
            if all(counts[k] >= caps[k] for k in quotas):
                break
            kinds_left = [k for k in quotas
                          if counts[k] < caps[k] and
                          counts[(domain, k)] < per_domain[k]]
            if not kinds_left:
                break
            found = None
            for i in range(len(sent)):
                if i + 2 > len(sent) or sent[i:i + 2] not in prefix2:
                    continue
                for length in (4, 3, 2):
                    if i + length > len(sent):
                        continue
                    w = sent[i:i + length]
                    if w in word_set:
                        found = (i, w)
                        break
                if found:
                    break
            if not found:
                continue
            i, word = found
            syllables = lazy_pinyin(word, style=Style.NORMAL,
                                    errors="ignore")
            if len(syllables) != len(word):
                continue
            kind = rng.choice(kinds_left)
            context = sent[max(0, i - ctx):i]
            if len(context) < 8:
                continue
            if kind == "word":
                target = word
                keys = clean_keys(syllables)
                span = context + target
            else:
                idx = rng.randrange(len(word))
                target = word[idx]
                keys = clean_keys([syllables[idx]])
                span = context + target
            if not keys:
                continue
            if (context, target) in seen:
                continue
            seen.add((context, target))
            cases.append({"domain": domain, "kind": kind, "context": context,
                          "target": target, "keys": keys, "span": span})
            counts[kind] += 1
            counts[(domain, kind)] += 1
    return cases


def strict_dedup(cases, spans_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    spans_path = os.path.join(OUT_DIR, spans_name)
    with io.open(spans_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c["span"] + "\n")
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

    kept = [c for c in cases if not contaminated(c["span"])]
    print("grep exit=%d matched=%d kept=%d/%d" %
          (proc.returncode, len(matched), len(kept), len(cases)), flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-quota", type=int, default=300)
    ap.add_argument("--word-quota", type=int, default=300)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--top-words", type=int, default=50000)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocab_chars(VOCAB)
    common = load_common_words(vocab, args.top_words)
    print("common words:", len(common), flush=True)
    quotas = {"char": args.char_quota, "word": args.word_quota}
    cases = make_cases(SOURCES, common, rng, quotas, args.oversample,
                       args.ctx)
    print("candidates: char=%d word=%d" %
          (sum(1 for c in cases if c["kind"] == "char"),
           sum(1 for c in cases if c["kind"] == "word")), flush=True)
    kept = strict_dedup(cases, "spans_real.txt")
    kept.sort(key=lambda c: (c["kind"], c["domain"]))
    rng.shuffle(kept)

    req_path = os.path.join(OUT_DIR, "requests.tsv")
    meta_path = os.path.join(OUT_DIR, "requests_meta.tsv")
    with io.open(req_path, "w", encoding="utf-8", newline="\n") as fr, \
            io.open(meta_path, "w", encoding="utf-8", newline="\n") as fm:
        for i, c in enumerate(kept):
            c["id"] = i
            fr.write("%d\t%s\t%s\n" % (i, c["keys"], c["target"]))
            fm.write("%d\t%s\t%s\t%s\n" % (i, c["kind"], c["domain"],
                                           c["context"]))
    print("saved", req_path, len(kept), flush=True)


if __name__ == "__main__":
    main()
