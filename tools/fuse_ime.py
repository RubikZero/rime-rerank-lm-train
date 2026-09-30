"""Test conservative fusion with the IME order prior and top-K reranking."""

import argparse
import io
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_real import build_schema_cases  # noqa
from eval_testset import evaluate  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TOK = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"


def z(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return np.zeros_like(a) if s < 1e-9 else (a - a.mean()) / s


def rank_of(case, scores):
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    return order.index(case["candidates"].index(case["target"])) + 1


def summarize(ranks, cases):
    out = {}
    for tag in ("char", "word", "ALL"):
        rs = [r for c, r in zip(cases, ranks) if tag == "ALL" or
              c["kind"] == tag]
        if not rs:
            continue
        n = len(rs)
        out[tag] = {"n": n, "top1": sum(r == 1 for r in rs) / n,
                    "top3": sum(r <= 3 for r in rs) / n,
                    "mrr": sum(1.0 / r for r in rs) / n,
                    "avg_rank": sum(rs) / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="luna_pinyin_simp")
    ap.add_argument("--quota", type=int, default=300)
    args = ap.parse_args()

    cases = build_schema_cases(args.schema, {"char": args.quota,
                                             "word": args.quota}, 42)
    hits = [c for c in cases if c["ime_rank"] is not None]
    ime = [c["ime_rank"] + 1 for c in hits]
    print("IME baseline:", summarize(ime, hits), flush=True)

    models = []
    with io.open(os.path.join(HERE, "models_real.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name, path = line.split("\t", 1)
                models.append((name, path))

    tok = AutoTokenizer.from_pretrained(BASE_TOK)
    records = []
    for ci, case in enumerate(cases):
        ctx_ids = tuple(tok.encode(case["context"], add_special_tokens=False))
        for yi, cand in enumerate(case["candidates"]):
            cand_ids = tuple(tok.convert_tokens_to_ids(ch) for ch in cand)
            records.append((ctx_ids, cand_ids, ci, yi))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefer = {"v1_step-5000", "v1_final", "v2_step-5000"}
    selected = [m for m in models if m[0] in prefer] or models[:2]
    for name, path in selected:
        scores = evaluate(cases, records, tok, path, device, 512, "seq")
        for beta in (1.0, 2.0, 4.0, 8.0):
            ranks = []
            for ci, case in enumerate(cases):
                if case["ime_rank"] is None:
                    continue
                sc = scores[ci]
                prior = np.arange(len(sc), 0, -1, dtype=np.float64)
                comb = z(sc) + beta * z(prior)
                ranks.append(rank_of(case, list(comb)))
            summary = summarize(ranks, hits)
            print("%-12s ime_prior beta=%.1f ALL=%s char=%s word=%s" %
                  (name, beta, summary["ALL"], summary["char"],
                   summary["word"]), flush=True)
        for k in (3, 5, 10):
            ranks = []
            for ci, case in enumerate(cases):
                if case["ime_rank"] is None:
                    continue
                sc = scores[ci]
                head = sorted(range(min(k, len(sc))), key=lambda i: -sc[i])
                order = head + list(range(min(k, len(sc)), len(sc)))
                rank = order.index(case["candidates"].index(
                    case["target"])) + 1
                ranks.append(rank)
            print("%-12s topk=%d %s" %
                  (name, k, summarize(ranks, hits)["ALL"]), flush=True)


if __name__ == "__main__":
    main()
