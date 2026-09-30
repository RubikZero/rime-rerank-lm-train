"""Verify the prefix-cache graphs against the shipping model.onnx.

For N real validation cases it compares the candidate ranking of
  * model.onnx        : [CLS] ctx [MASK] [SEP], mask_index -> logits    (today)
  * encode_prefix + score_tail : frozen prefix + recomputed tail        (cache)
Both run in ONNX Runtime, on the very files the plugin loads.

Usage: python s1_verify_graphs.py [--limit N] [--tail T] [--dir DIR]
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

MODEL_DIR = r"D:\Workspace\rime\third-party\lm-test\s1-models"
DATA = r"D:\Workspace\rime\third-party\lm-train\reranker_v2\data.npz"
CLS, MASK, SEP, PAD = 101, 103, 102, 0


def load_val(limit):
    d = np.load(DATA)
    sel = np.flatnonzero(d["split"] == 1)
    if limit:
        sel = sel[:limit]
    return (d["ctx"][sel].astype(np.int64), d["cand"][sel].astype(np.int64),
            d["mask"][sel].astype(np.int64), d["target_pos"][sel].astype(np.int64))


def restricted(logits, cand, mask):
    if logits.ndim == 3:          # model.onnx returns [batch, 1, vocab]
        logits = logits[:, 0, :]
    s = np.take_along_axis(logits, cand, axis=1)
    return np.where(mask == 0, -1e9, s)


def metrics(scores, target):
    order = np.argsort(-scores, axis=1)
    rank = (order == target[:, None]).nonzero()[1] + 1
    return {"top1": float((rank == 1).mean()), "top3": float((rank <= 3).mean()),
            "mrr": float((1.0 / rank).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--dir", default=MODEL_DIR)
    args = ap.parse_args()

    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    opt.intra_op_num_threads = 4
    base = ort.InferenceSession(os.path.join(args.dir, "model.onnx"), opt,
                                providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(os.path.join(args.dir, "encode_prefix.onnx"), opt,
                               providers=["CPUExecutionProvider"])
    step = ort.InferenceSession(os.path.join(args.dir, "score_tail.onnx"), opt,
                                providers=["CPUExecutionProvider"])
    layers, heads, head_dim = 4, 4, 64

    ctx, cand, mask, target = load_val(args.limit)
    n, W = ctx.shape
    ids = np.concatenate([np.full((n, 1), CLS, dtype=np.int64), ctx,
                          np.full((n, 1), MASK, dtype=np.int64),
                          np.full((n, 1), SEP, dtype=np.int64)], axis=1)
    att = np.concatenate([np.ones((n, 1), dtype=np.int64), (ctx != PAD).astype(np.int64),
                          np.ones((n, 2), dtype=np.int64)], axis=1)
    mpos = np.full((n,), 1 + W, dtype=np.int64)
    ref = restricted(base.run(None, {"input_ids": ids, "attention_mask": att,
                                     "mask_index": mpos})[0], cand, mask)

    cut = W - args.tail
    # One prefix per example here: validation rows have different contexts (the
    # plugin feeds a single shared prefix with batch 1, which the graph expands;
    # that path is covered by the plugin-like check).
    p_ids, p_att = ids[:, :1 + cut], att[:, :1 + cut]
    kv = enc.run(None, {"input_ids": p_ids, "attention_mask": p_att,
                        "position_ids": np.arange(1 + cut, dtype=np.int64)})
    chunk = np.concatenate([ctx[:, cut:],
                            np.full((n, 1), MASK, dtype=np.int64),
                            np.full((n, 1), SEP, dtype=np.int64)], axis=1)
    feed = {"input_ids": chunk,
            "attention_mask": np.ones_like(chunk),
            "position_ids": np.arange(1 + cut, 1 + cut + chunk.shape[1], dtype=np.int64),
            "past_attention_mask": p_att}
    names = enc.get_outputs()
    for i, o in enumerate(names):
        feed[o.name.replace("present.", "past.")] = kv[i]
    got_logits = step.run(None, feed)[0][:, 0, :]
    got = restricted(got_logits, cand, mask)

    pa = np.exp(ref - ref.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True)
    pb = np.exp(got - got.max(1, keepdims=True)); pb /= pb.sum(1, keepdims=True)
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    kl = float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(1).mean())
    mb, mg = metrics(ref, target), metrics(got, target)
    print("n=%d  tail=%d  prefix=%d tokens" % (n, args.tail, 1 + cut))
    print("  model.onnx          top1=%.4f top3=%.4f mrr=%.4f" %
          (mb["top1"], mb["top3"], mb["mrr"]))
    print("  prefix-cache graphs top1=%.4f top3=%.4f mrr=%.4f" %
          (mg["top1"], mg["top3"], mg["mrr"]))
    print("  per-case top1 agreement=%.2f%%  KL=%.5f  dtop1=%+.2fpt" %
          (agree * 100, kl, (mg["top1"] - mb["top1"]) * 100))
    return 0 if agree > 0.97 else 1


if __name__ == "__main__":
    raise SystemExit(main())
