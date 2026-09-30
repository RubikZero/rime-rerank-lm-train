"""ONNX Runtime CPU evaluation of char-level reranking (fp32 vs int8).

Reuses the real-candidate char test set (capped K=20 + target, ctx 256) and
reports top1 at beta=0.25 plus single-case latency, so the plugin's
implementation can be validated against PyTorch numbers.
"""

import argparse
import io
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, build_cases, load_char2id  # noqa

TESTSET = os.path.join(HERE, "testset")
K = 20


def masked_z(x, mask):
    x = np.asarray(x, dtype=np.float64)
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(x)
    mu = x[m].mean()
    sd = x[m].std()
    return np.where(m, (x - mu) / sd, 0.0) if sd > 1e-9 else np.zeros_like(x)


def run_eval(sess, cases, beta=0.25, batch=32, limit=0):
    n = len(cases) if not limit else min(limit, len(cases))
    hits1 = 0
    for i in range(0, n, batch):
        group = cases[i:i + batch]
        b = len(group)
        ids = np.zeros((b, 259), dtype=np.int64)
        att = np.zeros((b, 259), dtype=np.int64)
        for j, c in enumerate(group):
            ids[j, 0] = 101
            ids[j, 1:257] = c["ctx"]
            att[j, 1:257] = (c["ctx"] != 0).astype(np.int64)
            ids[j, 257] = 103
            ids[j, 258] = 102
            att[j, [0, 257, 258]] = 1
        logits = sess.run(["logits"], {"input_ids": ids,
                                       "attention_mask": att})[0]
        x = logits[:, 257].astype(np.float64)
        x = x - x.max(axis=-1, keepdims=True)
        lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
        for j, c in enumerate(group):
            n_i = c["n"]
            cand = c["cand"][:n_i]
            logq = c["logq"][:n_i]
            lm = lp[j, cand]
            zlm = masked_z(lm, np.ones(n_i))
            zpr = masked_z(logq, np.ones(n_i))
            order = np.argsort(-(zlm + beta * zpr))
            hits1 += int(order[0] == c["target_pos"])
    return hits1 / n


def time_latency(sess, cases, n=300):
    times = []
    for c in cases[:n]:
        ids = np.zeros((1, 259), dtype=np.int64)
        att = np.zeros((1, 259), dtype=np.int64)
        ids[0, 1:257] = c["ctx"]
        att[0, 1:257] = (c["ctx"] != 0).astype(np.int64)
        ids[0, 0] = 101
        ids[0, 257] = 103
        ids[0, 258] = 102
        att[0, [0, 257, 258]] = 1
        t0 = time.perf_counter()
        sess.run(["logits"], {"input_ids": ids, "attention_mask": att})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": statistics.mean(times),
            "p50_ms": times[len(times) // 2],
            "p90_ms": times[int(len(times) * 0.9)],
            "max_ms": max(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--models", default="model.onnx,model.int8.onnx")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--latency-n", type=int, default=300)
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    cases = build_cases(char2id, {}, os.path.join(TESTSET,
                                                  "requests_v2.tsv"),
                        os.path.join(TESTSET, "requests_v2_meta.tsv"),
                        os.path.join(TESTSET,
                                     "candidates_v2_luna_pinyin_simp.tsv"),
                        256)
    print("char cases:", len(cases), flush=True)

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    out = {}
    for name in args.models.split(","):
        path = os.path.join(args.onnx_dir, name)
        if not os.path.exists(path):
            print("skip missing", path, flush=True)
            continue
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        top1 = run_eval(sess, cases)
        lat = time_latency(sess, cases, args.latency_n)
        out[name] = {"top1_beta0.25": top1, **lat}
        print("%-18s top1=%.4f  mean=%.1fms p50=%.1f p90=%.1f max=%.1f"
              % (name, top1, lat["mean_ms"], lat["p50_ms"], lat["p90_ms"],
                 lat["max_ms"]), flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
