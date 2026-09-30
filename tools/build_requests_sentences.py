"""Build sentence-level (整句) test requests from external corpora.

Target = a contiguous Han span (4-10 chars, whole-clause typing), keys = full
pinyin. Same context rule as v2: cross-sentence tail within a document, max 256
/ min 32. Strict dedup against CLUECorpusSmall (local window + full span).
"""

import argparse
import io
import os
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_requests_v2 import (DOMAINS, GROUP_DOMAIN, clean_keys,  # noqa
                               is_han, load_words)
from build_testset import GREP, TRAIN_CORPUS, load_vocab_chars  # noqa
from pypinyin import Style, lazy_pinyin  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "testset")
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
LENGTHS = (4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10)


def collect_sentences(reader, domain, vocab, rng, quota, ctx):
    cases = []
    tail = ""
    last_doc = None
    for doc, sent in reader:
        if doc != last_doc:
            last_doc = doc
            tail = ""
        if quota <= 0:
            break
        length = rng.choice(LENGTHS)
        if len(sent) < length + 4:
            tail = (tail + sent)[-ctx:]
            continue
        start = rng.randrange(0, len(sent) - length + 1)
        span = sent[start:start + length]
        if not all(is_han(c) for c in span):
            tail = (tail + sent)[-ctx:]
            continue
        syllables = lazy_pinyin(span, style=Style.NORMAL, errors="ignore")
        if len(syllables) != len(span):
            tail = (tail + sent)[-ctx:]
            continue
        keys = clean_keys(syllables)
        if not keys or any(c not in vocab for c in span):
            tail = (tail + sent)[-ctx:]
            continue
        context = (tail + sent[:start])[-ctx:]
        if len(context) < 32:
            tail = (tail + sent)[-ctx:]
            continue
        cases.append({"domain": domain, "kind": "sent", "context": context,
                      "target": span, "keys": keys})
        quota -= 1
        tail = (tail + sent)[-ctx:]
    return cases


def word_hits(sent, word_set):
    hits = []
    i = 0
    while i < len(sent):
        for wlen in (4, 3, 2):
            if i + wlen <= len(sent) and sent[i:i + wlen] in word_set:
                hits.append((i, wlen))
                i += wlen
                break
        else:
            i += 1
    return hits


def collect_word_spans(reader, domain, word_set, vocab, rng, quota, ctx,
                       min_len=4, max_len=10):
    cases = []
    tail = ""
    last_doc = None
    for doc, sent in reader:
        if doc != last_doc:
            last_doc = doc
            tail = ""
        if quota <= 0:
            break
        hits = word_hits(sent, word_set)
        if not hits:
            tail = (tail + sent)[-ctx:]
            continue
        s = rng.randrange(len(hits))
        start = hits[s][0]
        pos = start
        span_len = 0
        for i, wlen in hits[s:]:
            if i != pos:
                break
            if i + wlen - start > max_len:
                break
            pos = i + wlen
            span_len = pos - start
            if span_len >= min_len and (span_len >= max_len or
                                        rng.random() < 0.4):
                break
        if not (min_len <= span_len <= max_len):
            tail = (tail + sent)[-ctx:]
            continue
        span = sent[start:start + span_len]
        if not all(is_han(c) for c in span):
            tail = (tail + sent)[-ctx:]
            continue
        syllables = lazy_pinyin(span, style=Style.NORMAL, errors="ignore")
        if len(syllables) != len(span):
            tail = (tail + sent)[-ctx:]
            continue
        keys = clean_keys(syllables)
        if not keys or any(c not in vocab for c in span):
            tail = (tail + sent)[-ctx:]
            continue
        context = (tail + sent[:start])[-ctx:]
        if len(context) < 32:
            tail = (tail + sent)[-ctx:]
            continue
        cases.append({"domain": domain, "kind": "sent", "context": context,
                      "target": span, "keys": keys})
        quota -= 1
        tail = (tail + sent)[-ctx:]
    return cases


def strict_dedup(cases, prefix):
    def write_spans(name, span_fn):
        path = os.path.join(OUT_DIR, name)
        with io.open(path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(span_fn(c) + "\n")
        return path

    matched = set()
    for name, fn in (("%s_full.txt" % prefix,
                      lambda c: c["context"] + c["target"]),
                     ("%s_local.txt" % prefix,
                      lambda c: c["context"][-64:] + c["target"])):
        path = write_spans(name, fn)
        proc = subprocess.run([GREP, "-F", "-h", "-o", "-f", path,
                               TRAIN_CORPUS], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        got = set(x for x in proc.stdout.decode("utf-8", "ignore").splitlines()
                  if x)
        print("%s: matched=%d" % (name, len(got)), flush=True)
        matched |= got
    matched_list = sorted(matched, key=len, reverse=True)

    def contaminated(c):
        span = c["context"] + c["target"]
        local = c["context"][-64:] + c["target"]
        if span in matched or local in matched:
            return True
        for m in matched_list:
            if len(m) < len(local):
                break
            if local in m:
                return True
        return False

    kept = [c for c in cases if not contaminated(c)]
    print("dedup kept %d/%d" % (len(kept), len(cases)), flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("words", "sent"), default="words")
    ap.add_argument("--tag", default="sent")
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocab_chars(VOCAB)
    word_set = set(load_words(50000)) if args.mode == "words" else set()
    quota = {d: args.per_domain * args.oversample
             for d in set(GROUP_DOMAIN.values())}

    all_cases = []
    for name, (path, reader) in DOMAINS.items():
        if not os.path.exists(path):
            print("skip", path)
            continue
        domain = GROUP_DOMAIN[name]
        if args.mode == "words":
            cases = collect_word_spans(reader(path), domain, word_set, vocab,
                                       rng, quota[domain], args.ctx)
        else:
            cases = collect_sentences(reader(path), domain, vocab, rng,
                                      quota[domain], args.ctx)
        quota[domain] -= len(cases)
        all_cases.extend(cases)
        print("domain=%s collected=%d remaining=%d" %
              (domain, len(cases), quota[domain]), flush=True)

    kept = strict_dedup(all_cases, "spans_%s" % args.tag)

    by_bucket = defaultdict(list)
    for c in kept:
        by_bucket[c["domain"]].append(c)
    final = []
    for domain, cases in by_bucket.items():
        rng.shuffle(cases)
        final.extend(cases[:args.per_domain])
    rng.shuffle(final)

    with io.open(os.path.join(OUT_DIR, "requests_%s.tsv" % args.tag), "w",
                 encoding="utf-8", newline="\n") as fr, \
            io.open(os.path.join(OUT_DIR,
                                 "requests_%s_meta.tsv" % args.tag), "w",
                    encoding="utf-8", newline="\n") as fm:
        for i, c in enumerate(final):
            fr.write("%d\t%s\t%s\n" % (i, c["keys"], c["target"]))
            fm.write("%d\t%s\t%s\t%s\n" % (i, c["kind"], c["domain"],
                                           c["context"]))
    per = defaultdict(int)
    lens = defaultdict(int)
    for c in final:
        per[c["domain"]] += 1
        lens[len(c["target"])] += 1
    print("final=%d domains=%s lens=%s" %
          (len(final), dict(per), dict(sorted(lens.items()))), flush=True)


if __name__ == "__main__":
    main()
