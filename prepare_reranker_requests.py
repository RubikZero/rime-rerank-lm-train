"""Prepare single-char pinyin reranker requests from CLUECorpusSmall.

- samples sentences, picks a char inside a known word, keys = its syllable
- splits train/val by disjoint corpus lines (strict)
- exact span dedup within pool; sentence-level dedup between val and train
Writes: reranker/requests.tsv (id, keys, target) and reranker/meta.tsv
(id, split, context, word).
"""

import argparse
import io
import os
import random

import zhconv
from pypinyin import Style, lazy_pinyin

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = r"D:\Workspace\rime\third-party\corpus\CLUECorpusSmall.txt"
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
OUT_DIR = os.path.join(HERE, "reranker")


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
        (0xF900 <= o <= 0xFAFF)


def load_vocab(path):
    chars = set()
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if len(tok) == 1 and is_han(tok):
                chars.add(tok)
    return chars


def load_words(top_n):
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
            if 2 <= len(w) <= 4 and all(is_han(c) for c in w):
                items.append((freq, w))
    items.sort(reverse=True)
    return [w for _, w in items[:top_n]]


def clean_syllable(s):
    s = s.replace("ü", "v")
    return "".join(c for c in s if "a" <= c <= "z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400000)
    ap.add_argument("--val-cases", type=int, default=5000)
    ap.add_argument("--sample-rate", type=float, default=0.05)
    ap.add_argument("--top-words", type=int, default=50000)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocab(VOCAB)
    words = load_words(args.top_words)
    word_set = set(words)
    prefix2 = set(w[:2] for w in words)
    print("words=%d vocab=%d" % (len(words), len(vocab)), flush=True)

    cases = []
    seen_span = set()
    line_id = 0
    scanned = 0
    with io.open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            scanned += 1
            if len(cases) >= args.target:
                break
            if rng.random() > args.sample_rate:
                continue
            sent = line.strip()
            if len(sent) < 16 or len(sent) > 300:
                continue
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
            idx = rng.randrange(len(word))
            syllables = lazy_pinyin(word, style=Style.NORMAL,
                                    errors="ignore")
            if len(syllables) != len(word):
                continue
            keys = clean_syllable(syllables[idx])
            target = word[idx]
            if not keys or target not in vocab:
                continue
            pos = i + idx
            context = sent[max(0, pos - args.ctx):pos]
            if len(context) < 8:
                continue
            span = context + target
            if span in seen_span:
                continue
            seen_span.add(span)
            cases.append({"line": line_id, "context": context,
                          "target": target, "keys": keys, "word": word,
                          "sentence": sent})
            line_id += 1
            if len(cases) % 50000 == 0:
                print("cases=%d scanned=%d" % (len(cases), scanned),
                      flush=True)

    # split by disjoint lines
    from collections import Counter

    line_counts = Counter(c["line"] for c in cases)
    lines = list(line_counts.keys())
    rng.shuffle(lines)
    val_lines = set()
    val_count = 0
    for ln in lines:
        if val_count >= args.val_cases:
            break
        val_lines.add(ln)
        val_count += line_counts[ln]
    train = [c for c in cases if c["line"] not in val_lines]
    val = [c for c in cases if c["line"] in val_lines]

    # strict: sentence-level dedup between val and train
    train_sent = set(c["sentence"] for c in train)
    val = [c for c in val if c["sentence"] not in train_sent]
    # exact span dedup inside val
    seen = set()
    val2 = []
    for c in val:
        span = c["context"] + c["target"]
        if span in seen:
            continue
        seen.add(span)
        val2.append(c)
    val = val2

    os.makedirs(OUT_DIR, exist_ok=True)
    with io.open(os.path.join(OUT_DIR, "requests.tsv"), "w",
                 encoding="utf-8", newline="\n") as fr, \
            io.open(os.path.join(OUT_DIR, "meta.tsv"), "w",
                    encoding="utf-8", newline="\n") as fm:
        for split, subset in (("train", train), ("val", val)):
            for k, c in enumerate(subset):
                rid = "%s-%d" % (split, k)
                fr.write("%s\t%s\t%s\n" % (rid, c["keys"], c["target"]))
                fm.write("%s\t%s\t%s\t%s\n" % (rid, split, c["context"],
                                               c["word"]))
    print("train=%d val=%d (file reranker/requests.tsv)" %
          (len(train), len(val)), flush=True)


if __name__ == "__main__":
    main()
