"""Export the pinyin syllable -> characters table for plugin-side decoding.

Reads the syllable->chars JSON cache produced by lm_decode_sentences.py and
writes a compact TSV (syllable \\t chars) next to the ONNX model.
"""

import argparse
import io
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with io.open(args.cache, encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("map", {})
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for syl in sorted(m):
            chars = "".join(m[syl])
            if chars:
                f.write("%s\t%s\n" % (syl, chars))
    print("wrote %d syllables -> %s (%d bytes)"
          % (len(m), args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
