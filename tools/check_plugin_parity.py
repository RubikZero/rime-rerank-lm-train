"""Compare C++ plugin ranking with Python reference on the same requests.

Runs the exported ONNX (mask-index graph) over candidate lists built two ways
(capped top-K by prior vs full menu) to explain differences from the plugin.
"""

import argparse
import io
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, is_han, load_candidates, load_char2id  # noqa
from eval_reranker import load_requests_meta  # noqa

TESTSET = os.path.join(HERE, "testset")


def masked_z(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-9 else (x - x.mean()) / sd


def build_items(rid, target, cands, char2id, cap):
    by_ch = {}
    for rank, (text, q, ctype) in enumerate(cands.get(rid, [])):
        if len(text) != 1 or not is_han(text) or text not in char2id:
            continue
        prev = by_ch.get(text)
        if prev is None or q > prev[0]:
            by_ch[text] = {"ch": text, "q": q, "rank": rank}
    if target not in by_ch:
        return None
    items = sorted(by_ch.values(), key=lambda x: -x["q"])
    if cap and len(items) > cap:
        head = items[:cap]
        if not any(x["ch"] == target for x in head):
            head.append(by_ch[target])
        items = head
    return items


def score(sess, context, items, char2id, ctx_len=256):
    vals = [char2id[c] for c in context if c in char2id][-ctx_len:]
    ctx = np.zeros(ctx_len, dtype=np.int64)
    ctx[ctx_len - len(vals):] = vals
    b = len(items)
    ids = np.zeros((b, 1 + ctx_len + 2), dtype=np.int64)
    att = np.zeros_like(ids)
    ids[:, 0] = 101
    ids[:, 1:1 + ctx_len] = ctx
    att[:, 1:1 + ctx_len] = (ctx != 0)
    ids[:, 1 + ctx_len] = 103
    ids[:, 1 + ctx_len + 1] = 102
    att[:, 0] = 1
    att[:, 1 + ctx_len:] = 1
    mask_index = np.full(b, 1 + ctx_len, dtype=np.int64)
    logits = sess.run(["logits"], {"input_ids": ids, "attention_mask": att,
                                   "mask_index": mask_index})[0][:, 0]
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    return np.array([lp[i, char2id[items[i]["ch"]]]
                     for i in range(len(items))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--beta", type=float, default=0.25)
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    target, meta = load_requests_meta(args.requests, args.meta)
    cands = load_candidates(args.candidates)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    for cap in (20, 0):
        n = top1 = 0
        for rid, tgt in target.items():
            if rid not in meta or len(tgt) != 1:
                continue
            kind, domain, context = meta[rid]
            if kind != "char":
                continue
            items = build_items(rid, tgt, cands, char2id, cap)
            if not items:
                continue
            lm = score(sess, context, items, char2id)
            logq = np.log1p(np.maximum([x["q"] for x in items], 0.0))
            fused = masked_z(lm) + args.beta * masked_z(logq)
            pick = int(np.argsort(-fused)[0])
            n += 1
            top1 += items[pick]["ch"] == tgt
        print("cap=%s n=%d top1=%.4f" % (cap or "full", n, top1 / n))


if __name__ == "__main__":
    main()
