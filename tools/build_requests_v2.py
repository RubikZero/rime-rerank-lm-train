"""Build long-context test requests (v2) from external corpora.

Same context rule as training v2: random word-hit position, cross-sentence
tail within a document, max 256 / min 32. Doc-aware readers keep sentence
order (wiki item / lccc dialogue / lcsts post / thucnews article).
Strict dedup against CLECorpusSmall: local window (target + 64 chars) and the
full span (context + target).
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
from collections import defaultdict

import zhconv
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_testset import GREP, TRAIN_CORPUS, load_vocab_chars  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_TEST = r"D:\Workspace\rime\third-party\corpus-test"
OUT_DIR = os.path.join(HERE, "testset")
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"

SENT_SPLIT = re.compile(r"[。！？!?；;\n\r]+")
HAN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def is_han(ch):
    return bool(HAN.match(ch))


def clean_keys(syllables):
    keys = []
    for s in syllables:
        s = s.replace("ü", "v")
        s = "".join(c for c in s if "a" <= c <= "z")
        if not s:
            return None
        keys.append(s)
    return "".join(keys)


def sentences(text):
    for piece in SENT_SPLIT.split(text):
        piece = piece.strip()
        if 16 <= len(piece) <= 300:
            yield piece


def iter_wiki(path):
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    for doc, item in enumerate(data):
        text = ""
        if isinstance(item, dict):
            text = item.get("completion") or item.get("text") or ""
        elif isinstance(item, str):
            text = item
        for sent in sentences(text):
            yield doc, sent


def iter_lccc(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for doc, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, list):
                continue
            for turn in obj:
                if isinstance(turn, str):
                    for sent in sentences(turn.replace(" ", "")):
                        yield doc, sent


def iter_lcsts(path):
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for doc, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sent in sentences(obj.get("text") or ""):
                yield doc, sent


def iter_thucnews(path):
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for doc, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sent in sentences(obj.get("title", "")):
                yield doc, sent
            for sent in sentences(obj.get("content", "")):
                yield doc, sent


DOMAINS = {
    "wiki": (os.path.join(CORPUS_TEST, "wiki.json"), iter_wiki),
    "lccc": (os.path.join(CORPUS_TEST, "lccc_valid.jsonl.gz"), iter_lccc),
    "lccc2": (os.path.join(CORPUS_TEST, "lccc_train.jsonl.gz"), iter_lccc),
    "lcsts": (os.path.join(CORPUS_TEST, "lcsts_valid.jsonl"), iter_lcsts),
    "lcsts2": (os.path.join(CORPUS_TEST, "lcsts_train.jsonl"), iter_lcsts),
    "thucnews": (os.path.join(CORPUS_TEST, "thucnews_keji.jsonl"),
                 iter_thucnews),
    "thucnews2": (os.path.join(CORPUS_TEST, "thucnews_shehui.jsonl"),
                  iter_thucnews),
}
GROUP_DOMAIN = {"wiki": "wiki", "lccc": "lccc", "lccc2": "lccc",
                "lcsts": "lcsts", "lcsts2": "lcsts",
                "thucnews": "thucnews", "thucnews2": "thucnews"}


def load_words(top_n):
    items = []
    with io.open(ESSAY, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            try:
                freq = float(p[1])
            except ValueError:
                continue
            w = zhconv.convert(p[0], "zh-cn")
            if 2 <= len(w) <= 4 and all(is_han(c) for c in w):
                items.append((freq, w))
    items.sort(reverse=True)
    return [w for _, w in items[:top_n]]


def collect(reader, domain, word_set, prefix2, vocab, rng, quota, ctx,
            max_scan=None):
    cases = []
    tail = ""
    last_doc = None
    for doc, sent in reader:
        if max_scan and doc > max_scan:
            break
        if doc != last_doc:
            last_doc = doc
            tail = ""
        hits = []
        for i in range(len(sent)):
            if i + 2 > len(sent) or sent[i:i + 2] not in prefix2:
                continue
            for length in (4, 3, 2):
                if i + length <= len(sent) and sent[i:i + length] in word_set:
                    hits.append((i, length))
                    break
        if hits:
            i, length = rng.choice(hits)
            word = sent[i:i + length]
            syllables = lazy_pinyin(word, style=Style.NORMAL, errors="ignore")
            if len(syllables) == len(word):
                for kind in ("char", "word"):
                    if quota.get((domain, kind), 0) <= 0:
                        continue
                    if kind == "char":
                        idx = rng.randrange(len(word))
                        target = word[idx]
                        keys = clean_keys([syllables[idx]])
                        pos = i + idx
                    else:
                        target = word
                        keys = clean_keys(syllables)
                        pos = i
                    if not keys or any(c not in vocab for c in target):
                        continue
                    context = (tail + sent[:pos])[-ctx:]
                    if len(context) < 32:
                        continue
                    cases.append({"domain": domain, "kind": kind,
                                  "context": context, "target": target,
                                  "keys": keys})
                    quota[(domain, kind)] -= 1
        tail = (tail + sent)[-ctx:]
    return cases


def strict_dedup(cases):
    def write_spans(name, span_fn):
        path = os.path.join(OUT_DIR, name)
        with io.open(path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(span_fn(c) + "\n")
        return path

    matched = set()
    for name, fn in (("spans_v2_full.txt", lambda c: c["context"] + c["target"]),
                     ("spans_v2_local.txt",
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
    ap.add_argument("--per-domain-kind", type=int, default=250)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--top-words", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocab_chars(VOCAB)
    words = load_words(args.top_words)
    word_set = set(words)
    prefix2 = set(w[:2] for w in words)
    print("words=%d vocab=%d" % (len(words), len(vocab)), flush=True)

    quota = {}
    for g in set(GROUP_DOMAIN.values()):
        for kind in ("char", "word"):
            quota[(g, kind)] = args.per_domain_kind * args.oversample

    all_cases = []
    for name, (path, reader) in DOMAINS.items():
        if not os.path.exists(path):
            print("skip", path)
            continue
        domain = GROUP_DOMAIN[name]
        q = {k: v for k, v in quota.items() if k[0] == domain}
        cases = collect(reader(path), domain, word_set, prefix2, vocab, rng, q,
                        args.ctx)
        for c in cases:
            all_cases.append(c)
        print("domain=%s collected=%d remaining=%s" %
              (domain, len(cases), q), flush=True)

    kept = strict_dedup(all_cases)

    by_bucket = defaultdict(list)
    for c in kept:
        by_bucket[(c["domain"], c["kind"])].append(c)
    final = []
    for key, cases in by_bucket.items():
        rng.shuffle(cases)
        final.extend(cases[:args.per_domain_kind])
    rng.shuffle(final)

    with io.open(os.path.join(OUT_DIR, "requests_v2.tsv"), "w",
                 encoding="utf-8", newline="\n") as fr, \
            io.open(os.path.join(OUT_DIR, "requests_v2_meta.tsv"), "w",
                    encoding="utf-8", newline="\n") as fm:
        for i, c in enumerate(final):
            fr.write("%d\t%s\t%s\n" % (i, c["keys"], c["target"]))
            fm.write("%d\t%s\t%s\t%s\n" % (i, c["kind"], c["domain"],
                                           c["context"]))
    per = defaultdict(int)
    for c in final:
        per[(c["domain"], c["kind"])] += 1
    print("final=%d buckets=%s" % (len(final), dict(per)), flush=True)


if __name__ == "__main__":
    main()
