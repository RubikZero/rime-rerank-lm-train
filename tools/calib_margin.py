"""Sweep the smart-promotion margin threshold from plugin debug logs.

Reads extractor logs produced with `promote: smart, promote_margin: -100`
and RIME_LM_RANKER_DEBUG=1 (worker margin lines), joins with the targets,
and reports overall accuracy (promoted -> MLM result, else IME top) per
threshold, mirroring doc/smart-promote-calib.md.

Usage: python calib_margin.py --log testset/test_margin_h.log --requests ... \
    --cand testset/cand_margin_h.tsv --label hard
"""

import argparse
import io
import re

LINE = re.compile(
    r"\[lm_ranker\] worker margin=([+-][\d.]+) promote=(\d) "
    r"input=(\S*) text=(\S*) ime=(\S*)")


def load_requests(path):
    keys, target = {}, {}
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            keys[p[0]] = p[1]
            target[p[0]] = p[2]
    return keys, target


def load_cand(path):
    """rid -> (mlm_or_none, ime_top)."""
    out = {}
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        mlm = None
        ime = ""
        for c in p[2].split("\x1f"):
            f = c.split("\x1e")
            if len(f) < 2 or not f[0]:
                continue
            if f[-1] == "lm_ranker":
                mlm = f[0]
            elif not ime:
                ime = f[0]
        out[p[0]] = (mlm, ime)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--cand", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    keys, target = load_requests(args.requests)
    by_input = {}
    for line in io.open(args.log, encoding="utf-8", errors="replace"):
        m = LINE.search(line)
        if m:
            by_input[m.group(3)] = (float(m.group(1)), m.group(4), m.group(5))

    cands = load_cand(args.cand) if args.cand else {}
    recs = []
    for rid, inp in keys.items():
        tgt = target.get(rid, "")
        if not tgt or inp not in by_input:
            continue
        margin, mlm_log, ime_log = by_input[inp]
        mlm = mlm_log
        ime = ime_log
        if rid in cands:
            c_mlm, c_ime = cands[rid]
            if c_mlm:
                mlm = c_mlm
            if c_ime:
                ime = c_ime
        recs.append({"margin": margin,
                     "m_ok": mlm == tgt,
                     "i_ok": ime == tgt})
    n = len(recs)
    print("%s: n=%d  MLM on-promote acc=%.4f  IME acc=%.4f"
          % (args.label, n,
             sum(r["m_ok"] for r in recs) / max(1, n),
             sum(r["i_ok"] for r in recs) / max(1, n)))
    best = None
    for t10 in range(-10, 31):
        T = t10 / 10.0
        promo = sum(1 for r in recs if r["margin"] > T)
        acc = sum((r["m_ok"] if r["margin"] > T else r["i_ok"])
                  for r in recs)
        a = acc / max(1, n)
        if best is None or a > best[1]:
            best = (T, a, promo)
        print("T=%+4.1f  acc=%.4f (promo %.2f)" % (T, a, promo / max(1, n)))
    print("best: T=%+.1f acc=%.4f promo=%.2f"
          % (best[0], best[1], best[2] / max(1, n)))


if __name__ == "__main__":
    main()
