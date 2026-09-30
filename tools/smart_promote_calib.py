"""Calibrate the `smart` promotion rule.

Promote the MLM sentence only when its per-char log-likelihood exceeds the
IME top candidate's by a margin. Sweeps the threshold on both sentence sets
and reports the resulting accuracy, so the plugin default can be chosen.
"""

import argparse
import io
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lm_decode_sentences import (CACHE, VOCAB, build_cases,  # noqa
                                 build_syllable_map, load_char2id,
                                 load_ime_raw, load_requests)
from ort_decode_bench import decode_ort  # noqa


def score_candidate(sess, c, chars, ctx_len=256):
    n = len(chars)
    if n == 0:
        return 0.0
    ctx = np.asarray(c["ctx_ids"], dtype=np.int64)
    L = 1 + ctx_len + (n - 1) + 2
    ids = np.zeros((n, L), dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    ids[:, 0] = 101
    ids[:, 1:1 + ctx_len] = ctx
    att[:, 1:1 + ctx_len] = (ctx != 0).astype(np.int64)
    att[:, 0] = 1
    pos = []
    for k in range(n):
        for p in range(k):
            ids[k, 1 + ctx_len + p] = c["char2id"][chars[p]]
            att[k, 1 + ctx_len + p] = 1
        mp = 1 + ctx_len + k
        ids[k, mp] = 103
        att[k, mp] = 1
        ids[k, mp + 1] = 102
        att[k, mp + 1] = 1
        pos.append(mp)
    logits = sess.run(["logits"], {"input_ids": ids,
                                   "attention_mask": att})[0]
    x = logits[np.arange(n), np.asarray(pos)].astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    return sum(float(lp[k, c["char2id"][chars[k]]]) for k in range(n))


def analyse(sess, cases, ime, beam=4, topk=8, ctx_len=256):
    outs = decode_ort(sess, cases, beam, topk, ctx_len)
    recs = []
    for c, o in zip(cases, outs):
        rank, top = ime.get(c["rid"], (-1, ""))
        i_ok = rank == 0
        m_ok = o == c["target"]
        s_mlm = score_candidate(sess, c, o, ctx_len)
        s_ime = 0.0
        if top and all(ch in c["char2id"] for ch in top):
            s_ime = score_candidate(sess, c, top, ctx_len)
        margin = s_mlm / max(1, len(o)) - s_ime / max(1, len(top))
        recs.append({"i_ok": i_ok, "m_ok": m_ok, "margin": margin,
                     "top": top, "out": o, "target": c["target"]})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--hard-requests", required=True)
    ap.add_argument("--hard-meta", required=True)
    ap.add_argument("--hard-candidates", required=True)
    ap.add_argument("--words-requests", required=True)
    ap.add_argument("--words-meta", required=True)
    ap.add_argument("--words-candidates", required=True)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    rev = {cid: ch for ch, cid in char2id.items()}
    syl2ids = build_syllable_map(char2id, CACHE)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    sets = {}
    for name, req, meta, cand in (
            ("hard", args.hard_requests, args.hard_meta,
             args.hard_candidates),
            ("words", args.words_requests, args.words_meta,
             args.words_candidates)):
        keys, target, m = load_requests(req, meta)
        cases = build_cases(char2id, rev, syl2ids, keys, target, m, args.ctx)
        ime = load_ime_raw(cand)
        sets[name] = analyse(sess, cases, ime, ctx_len=args.ctx)
        n = len(sets[name])
        ime_acc = sum(r["i_ok"] for r in sets[name]) / n
        mlm_acc = sum(r["m_ok"] for r in sets[name]) / n
        print("%s: n=%d IME=%.4f MLM=%.4f" % (name, n, ime_acc, mlm_acc),
              flush=True)

    grid = [round(-1.0 + 0.1 * i, 2) for i in range(41)]
    lines = []
    best = (0.0, -1.0, {})
    for t in grid:
        accs = {}
        for name, recs in sets.items():
            hits = 0
            promo = 0
            for r in recs:
                if r["margin"] > t:
                    promo += 1
                    hits += r["m_ok"]
                else:
                    hits += r["i_ok"]
            accs[name] = (hits / len(recs), promo / len(recs))
        total = sum(a for a, _ in accs.values()) / len(accs)
        lines.append("T=%+.2f  hard=%.4f(promo %.2f)  words=%.4f(promo %.2f) "
                     "avg=%.4f" % (t, accs["hard"][0], accs["hard"][1],
                                   accs["words"][0], accs["words"][1], total))
        if total > best[1]:
            best = (t, total, accs)
    for l in lines:
        print(l, flush=True)
    print("best T=%+.2f avg=%.4f" % (best[0], best[1]), flush=True)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write("best T=%+.2f\n\n" % best[0])
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
