"""Export the rime-based syllable->char:weight table for the C++ plugin.

Reads testset/char_readings.tsv (built by build_readings.py) and writes
one row per (syllable, char) with the char restricted to the MLM vocab:

    syllable \t char \t weight

Usage: python export_plugin_readings.py --out <model_dir>/syllable_readings.tsv
"""

import argparse
import io
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, load_char2id  # noqa

DEFAULT_IN = os.path.join(HERE, "testset", "char_readings.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab = set(load_char2id(VOCAB))
    rows = defaultdict(list)
    for line in io.open(args.inp, encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 3 or f[0] not in vocab:
            continue
        rows[f[1]].append((f[0], float(f[2])))

    with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for syl in sorted(rows):
            for ch, w in sorted(rows[syl], key=lambda x: -x[1]):
                f.write("%s\t%s\t%.4f\n" % (syl, ch, w))
    total = sum(len(v) for v in rows.values())
    print("syllables=%d rows=%d -> %s (%d bytes)"
          % (len(rows), total, args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
