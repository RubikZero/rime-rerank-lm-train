"""Diagnose real-candidate reranking: LM vs frequency prior vs fusion."""

import argparse
import io
import json
import math
import os
import sys

import numpy as np
import torch
import zhconv
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_real import build_schema_cases  # noqa
from eval_testset import evaluate  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
BASE_TOK = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"


def load_freq():
    freq = {}
    with io.open(ESSAY, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    v = float(parts[1])
                except ValueError:
                    continue
                k = zhconv.convert(parts[0], "zh-cn")
                freq[k] = freq.get(k, 0.0) + v
    return freq


def z(arr):
    a = np.asarray(arr, dtype=np.float64)
    s = a.std()
    return np.zeros_like(a) if s < 1e-9 else (a - a.mean()) / s


def rank_of(case, score_list):
    order = sorted(range(len(score_list)), key=lambda k: -score_list[k])
    return order.index(case["candidates"].index(case["target"])) + 1


def summarize(ranks, cases):
    out = {}
    for tag in ("char", "word", "ALL"):
        rs = [r for c, r in zip(cases, ranks) if tag == "ALL" or
              c["kind"] == tag]
        if not rs:
            continue
        n = len(rs)
        out[tag] = {"n": n,
                    "top1": sum(r == 1 for r in rs) / n,
                    "top3": sum(r <= 3 for r in rs) / n,
                    "mrr": sum(1.0 / r for r in rs) / n,
                    "avg_rank": sum(rs) / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="luna_pinyin_simp")
    ap.add_argument("--list", default="models_real.txt")
    ap.add_argument("--quota", type=int, default=300)
    ap.add_argument("--betas", default="0,0.25,0.5,1.0,2.0,4.0")
    args = ap.parse_args()

    cases = build_schema_cases(args.schema, {"char": args.quota,
                                             "word": args.quota}, 42)
    hits = [c for c in cases if c["ime_rank"] is not None]
    print("cases", len(cases), "hits", len(hits), flush=True)
    freq = load_freq()

    freq_ranks = [rank_of(c, [math.log1p(freq.get(x, 0.0))
                              for x in c["candidates"]]) for c in hits]
    ime_ranks = [c["ime_rank"] + 1 for c in hits]
    print("baselines: IME", summarize(ime_ranks, hits), flush=True)
    print("baselines: freq", summarize(freq_ranks, hits), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_TOK)
    unk = tok.unk_token_id
    cases_all = cases
    records = []
    for ci, case in enumerate(cases_all):
        ctx_ids = tuple(tok.encode(case["context"], add_special_tokens=False))
        for yi, cand in enumerate(case["candidates"]):
            cand_ids = tuple(tok.convert_tokens_to_ids(ch) for ch in cand)
            records.append((ctx_ids, cand_ids, ci, yi))

    models = []
    with io.open(args.list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name, path = line.split("\t", 1)
                models.append((name, path))

    betas = [float(b) for b in args.betas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, path in models[:4]:
        scores = evaluate(cases_all, records, tok, path, device, 512, "seq")
        logf = [[math.log1p(freq.get(x, 0.0)) for x in c["candidates"]]
                for c in cases_all]
        for beta in betas:
            ranks, used = [], []
            for ci, case in enumerate(cases_all):
                if case["ime_rank"] is None:
                    continue
                sc = scores[ci]
                if sc is None or any(s is None for s in sc):
                    continue
                comb = z(sc) + beta * z(logf[ci])
                ranks.append(rank_of(case, list(comb)))
                used.append(case)
            print("%-12s beta=%.2f %s" %
                  (name, beta, summarize(ranks, used)["ALL"]), flush=True)


if __name__ == "__main__":
    main()
