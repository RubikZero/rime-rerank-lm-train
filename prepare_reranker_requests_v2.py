"""Prepare single-char reranker requests with LONG, cross-sentence context.

Changes vs v1:
- target position sampled uniformly among all word hits (was: first hit only)
- context = previous sentence tail + current sentence prefix, up to 256 chars,
  min 32 chars; previous text resets at document boundaries (blank lines)
- train/val split by document id (context never crosses the split)
"""

import argparse
import io
import os
import random
from collections import Counter

import zhconv
from pypinyin import Style, lazy_pinyin

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = r"D:\Workspace\rime\third-party\corpus\CLUECorpusSmall.txt"
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
OUT_DIR = os.path.join(HERE, "reranker_v2")


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
    ap.add_argument("--target", type=int, default=320000)
    ap.add_argument("--val-cases", type=int, default=5000)
    ap.add_argument("--sample-rate", type=float, default=0.03)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--min-ctx", type=int, default=32)
    ap.add_argument("--top-words", type=int, default=50000)
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
    doc_id = -1
    tail = ""
    scanned = 0
    stats = Counter()
    with io.open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                doc_id += 1
                tail = ""
                continue
            scanned += 1
            if len(cases) >= args.target:
                break
            sent = line.strip()
            if len(sent) < 16 or len(sent) > 300:
                tail = (tail + sent)[-args.max_ctx:]
                continue
            # collect all word hits in this sentence
            hits = []
            for i in range(len(sent)):
                if i + 2 > len(sent) or sent[i:i + 2] not in prefix2:
                    continue
                for length in (4, 3, 2):
                    if i + length <= len(sent) and sent[i:i + length] in word_set:
                        hits.append((i, length))
                        break
            # sampling decides whether to emit, but the tail always advances
            if not hits or rng.random() > args.sample_rate:
                tail = (tail + sent)[-args.max_ctx:]
                continue
            stats["sent_with_hit"] += 1
            i, length = rng.choice(hits)
            word = sent[i:i + length]
            idx = rng.randrange(len(word))
            syllables = lazy_pinyin(word, style=Style.NORMAL, errors="ignore")
            if len(syllables) != len(word):
                tail = (tail + sent)[-args.max_ctx:]
                continue
            keys = clean_syllable(syllables[idx])
            target = word[idx]
            pos = i + idx
            if not keys or target not in vocab:
                tail = (tail + sent)[-args.max_ctx:]
                continue
            context = (tail + sent[:pos])[-args.max_ctx:]
            if len(context) < args.min_ctx:
                tail = (tail + sent)[-args.max_ctx:]
                continue
            span = context + target
            if span in seen_span:
                tail = (tail + sent)[-args.max_ctx:]
                continue
            seen_span.add(span)
            cases.append({"doc": doc_id, "context": context, "target": target,
                          "keys": keys, "word": word})
            tail = (tail + sent)[-args.max_ctx:]
            if len(cases) % 50000 == 0:
                print("cases=%d scanned=%d docs=%d" %
                      (len(cases), scanned, doc_id + 1), flush=True)

    # split by document ids (context cannot cross the split)
    doc_counts = Counter(c["doc"] for c in cases)
    docs = list(doc_counts.keys())
    rng.shuffle(docs)
    val_docs = set()
    val_count = 0
    for d in docs:
        if val_count >= args.val_cases:
            break
        val_docs.add(d)
        val_count += doc_counts[d]
    train = [c for c in cases if c["doc"] not in val_docs]
    val = [c for c in cases if c["doc"] in val_docs]
    # dedup within val
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
    print("train=%d val=%d (reranker_v2/requests.tsv)" %
          (len(train), len(val)), flush=True)


if __name__ == "__main__":
    main()
