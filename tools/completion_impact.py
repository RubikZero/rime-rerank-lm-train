"""Measure the impact of predictive multi-char candidates on rerank accuracy.

Compares, on real menus: IME baseline, rerank with all candidates, and rerank
with multi-char (prediction/completion-like) candidates excluded.
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


def masked_z(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-9 else (x - x.mean()) / sd


def score_mixed(sess, context, texts, char2id, ctx_len=256):
    """Seq pseudo-likelihood for candidates of any length (mask-index ONNX)."""
    vals = [char2id[c] for c in context if c in char2id][-ctx_len:]
    ctx = np.zeros(ctx_len, dtype=np.int64)
    ctx[ctx_len - len(vals):] = vals
    ids_all = [[char2id[ch] for ch in text] for text in texts]
    scores = np.full(len(texts), -np.inf, dtype=np.float64)
    maxlen = max((len(x) for x in ids_all if x), default=0)
    for k in range(maxlen):
        rows = [i for i, x in enumerate(ids_all) if len(x) > k]
        if not rows:
            continue
        b = len(rows)
        seq = 1 + ctx_len + k + 2
        ids = np.zeros((b, seq), dtype=np.int64)
        att = np.zeros((b, seq), dtype=np.int64)
        ids[:, 0] = 101
        att[:, 0] = 1
        ids[:, 1:1 + ctx_len] = ctx
        att[:, 1:1 + ctx_len] = (ctx != 0)
        for r, i in enumerate(rows):
            for p in range(k):
                ids[r, 1 + ctx_len + p] = ids_all[i][p]
                att[r, 1 + ctx_len + p] = 1
            ids[r, 1 + ctx_len + k] = 103
            att[r, 1 + ctx_len + k] = 1
            ids[r, 1 + ctx_len + k + 1] = 102
            att[r, 1 + ctx_len + k + 1] = 1
        mask_index = np.full(b, 1 + ctx_len + k, dtype=np.int64)
        logits = sess.run(["logits"], {"input_ids": ids,
                                       "attention_mask": att,
                                       "mask_index": mask_index})[0][:, 0]
        x = logits.astype(np.float64)
        x = x - x.max(axis=-1, keepdims=True)
        lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
        for r, i in enumerate(rows):
            if not np.isfinite(scores[i]):
                scores[i] = 0.0
            scores[i] += lp[r, ids_all[i][k]]
    return scores


def build_items(rid, target, cands, char2id):
    by_text = {}
    for rank, (text, q, ctype) in enumerate(cands.get(rid, [])):
        if not text or not all(is_han(ch) and ch in char2id for ch in text):
            continue
        prev = by_text.get(text)
        if prev is None or q > prev[0]:
            by_text[text] = {"ch": text, "q": q, "rank": rank,
                             "type": ctype}
    if target not in by_text:
        return None
    return sorted(by_text.values(), key=lambda x: x["q"])


def evaluate(cases, args):
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    stats = {"n": 0, "ime": 0, "all": 0, "exact": 0, "multi_ranked": 0,
             "multi_total": 0}
    for rid, case in cases.items():
        target = case["target"]
        context = case["context"]
        items = build_items(rid, target, args.cands, args.char2id)
        if not items:
            continue
        multi = [x for x in items if len(x["ch"]) > 1]
        stats["multi_total"] += len(multi)
        ime_pick = max(items, key=lambda x: (x["q"], -x["rank"]))
        stats["n"] += 1
        stats["ime"] += ime_pick["ch"] == target

        def rerank(subset):
            if len(subset) < 2:
                return max(subset, key=lambda x: x["q"])["ch"]
            lm = score_mixed(sess, context, [x["ch"] for x in subset],
                             args.char2id)
            logq = np.log1p(np.maximum([x["q"] for x in subset], 0.0))
            fused = masked_z(lm) + args.beta * masked_z(logq)
            return subset[int(np.argsort(-fused)[0])]["ch"]

        all_pick = rerank(items)
        exact_subset = [x for x in items if len(x["ch"]) == 1]
        exact_pick = rerank(exact_subset)
        stats["all"] += all_pick == target
        stats["exact"] += exact_pick == target
        if len(multi) > 0 and all_pick in [m["ch"] for m in multi]:
            stats["multi_ranked"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--beta", type=float, default=0.25)
    args = ap.parse_args()

    args.char2id = load_char2id(VOCAB)
    args.cands = load_candidates(args.candidates)
    target, meta = load_requests_meta(args.requests, args.meta)
    cases = {}
    for rid, tgt in target.items():
        if rid not in meta or len(tgt) != 1:
            continue
        kind, domain, context = meta[rid]
        cases[rid] = {"target": tgt, "context": context}
    stats = evaluate(cases, args)
    n = stats["n"]
    print("n=%d  IME=%.4f  rerank_all=%.4f  rerank_exact_only=%.4f  "
          "multi_total=%d  multi_picked=%d"
          % (n, stats["ime"] / n, stats["all"] / n, stats["exact"] / n,
             stats["multi_total"], stats["multi_ranked"]))


if __name__ == "__main__":
    main()
