"""CPU (ONNX Runtime) benchmark of constrained sentence decoding.

Mirrors lm_decode_sentences.decode (left-padded 256-char context) and reports
exact match + per-sentence latency for beam x topk grid, as the reference for
the C++ plugin implementation.
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lm_decode_sentences import (CACHE, VOCAB, build_cases,  # noqa
                                 build_syllable_map, load_char2id,
                                 load_ime_raw, load_requests)


def decode_ort(sess, cases, beam, topk, ctx_len=256):
    outs = []
    for c in cases:
        ctx = np.asarray(c["ctx_ids"], dtype=np.int64)
        allowed = c["allowed"]
        beams = [([], 0.0)]
        for t, al in enumerate(allowed):
            rows = beams
            nrow = len(rows)
            L = 1 + ctx_len + t + 2
            ids = np.zeros((nrow, L), dtype=np.int64)
            att = np.zeros((nrow, L), dtype=np.int64)
            ids[:, 0] = 101
            ids[:, 1:1 + ctx_len] = ctx
            att[:, 1:1 + ctx_len] = (ctx != 0).astype(np.int64)
            for r, (g, _) in enumerate(rows):
                for p, ch in enumerate(g):
                    ids[r, 1 + ctx_len + p] = c["char2id"][ch]
                ids[r, L - 2] = 103
                ids[r, L - 1] = 102
                att[r, 0] = 1
                att[r, L - 2] = 1
                att[r, L - 1] = 1
                if t:
                    att[r, 1 + ctx_len:1 + ctx_len + t] = 1
            logits = sess.run(["logits"], {"input_ids": ids,
                                           "attention_mask": att})[0]
            x = logits[:, L - 2].astype(np.float64)
            x = x - x.max(axis=-1, keepdims=True)
            lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
            cands = []
            for r, (g, s) in enumerate(rows):
                sub = lp[r, np.asarray(al, dtype=np.int64)]
                k = min(topk, sub.size)
                idx = np.argpartition(-sub, k - 1)[:k]
                for i in idx:
                    cands.append((g + [c["rev"][al[int(i)]]],
                                  s + float(sub[int(i)])))
            cands.sort(key=lambda x: -x[1])
            beams = cands[:beam]
        outs.append("".join(beams[0][0][:len(allowed)]))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--beams", default="1,4,8,12")
    ap.add_argument("--topks", default="8,12")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    rev = {cid: ch for ch, cid in char2id.items()}
    syl2ids = build_syllable_map(char2id, CACHE)
    keys, target, meta = load_requests(args.requests, args.meta)
    cases = build_cases(char2id, rev, syl2ids, keys, target, meta, args.ctx)
    if args.limit:
        cases = cases[:args.limit]
    ime = load_ime_raw(args.candidates)
    n = len(cases)
    char_tot = sum(len(c["target"]) for c in cases)
    ime_exact = sum(1 for c in cases if ime.get(c["rid"], (-1, ""))[0] == 0)
    print("cases=%d IME exact=%.4f" % (n, ime_exact / n), flush=True)

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, sess_options=so,
                                providers=["CPUExecutionProvider"])
    decode_ort(sess, cases[:3], 2, 8, args.ctx)  # warmup

    rows = []
    for beam in [int(x) for x in args.beams.split(",")]:
        for topk in [int(x) for x in args.topks.split(",")]:
            t0 = time.perf_counter()
            outs = decode_ort(sess, cases, beam, topk, args.ctx)
            dt = time.perf_counter() - t0
            exact = sum(1 for c, o in zip(cases, outs) if o == c["target"])
            ch = sum(sum(1 for a, b in zip(o, c["target"]) if a == b)
                     for c, o in zip(cases, outs))
            rows.append((beam, topk, exact / n, ch / char_tot, dt,
                         dt / n * 1000))
            print("beam=%-2d topk=%-2d exact=%.4f char=%.4f "
                  "time=%7.1fs (%.1f ms/sent)"
                  % (beam, topk, exact / n, ch / char_tot, dt,
                     dt / n * 1000), flush=True)

    if args.out:
        lines = ["| beam | topk | 整句精确 | 字级 | 每句延迟 (CPU) |",
                 "|---:|---:|---:|---:|---:|",
                 "| IME | — | %.4f | — | — |" % (ime_exact / n)]
        for beam, topk, ex, ch, dt, ms in rows:
            lines.append("| %d | %d | %.4f | %.4f | %.1f ms |"
                         % (beam, topk, ex, ch, ms))
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("written:", args.out, flush=True)


if __name__ == "__main__":
    main()
