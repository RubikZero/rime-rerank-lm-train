"""Per-request comparison: C++ plugin top-1 vs Python ONNX top-1."""

import argparse
import io
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_plugin_parity import build_items, score  # noqa
from eval_reranker import VOCAB, load_candidates, load_char2id  # noqa
from eval_reranker import load_requests_meta  # noqa


def masked_z(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-9 else (x - x.mean()) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    target, meta = load_requests_meta(args.requests, args.meta)
    cands = load_candidates(args.candidates)
    cpp = {}
    for line in io.open(args.cpp, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            first = f[2].split("\x1f")[0].split("\x1e")[0]
            cpp[f[0]] = first
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    agree = total = 0
    shown = 0
    for rid, tgt in target.items():
        if rid not in meta or len(tgt) != 1:
            continue
        kind, domain, context = meta[rid]
        if kind != "char" or rid not in cpp:
            continue
        items = build_items(rid, tgt, cands, char2id, 0)
        if not items:
            continue
        lm = score(sess, context, items, char2id)
        logq = np.log1p(np.maximum([x["q"] for x in items], 0.0))
        fused = masked_z(lm) + args.beta * masked_z(logq)
        pick = items[int(np.argsort(-fused)[0])]["ch"]
        total += 1
        if pick == cpp[rid]:
            agree += 1
        elif shown < args.show:
            shown += 1
            print("--- rid=%s target=%s cpp=%s py=%s" %
                  (rid, tgt, cpp[rid], pick))
            print("ctx tail:", context[-40:])
            print("top5 py:", [(items[i]["ch"], round(float(fused[i]), 3))
                               for i in np.argsort(-fused)[:5]])
    print("agreement: %d/%d = %.4f" % (agree, total, agree / total))


if __name__ == "__main__":
    main()
