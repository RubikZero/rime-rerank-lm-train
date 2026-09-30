"""Evaluate checkpoints on real librime candidates.

Builds testset_real_<schema>.json from extraction output (candidates converted
to simplified), then measures IME baseline vs LM re-ranking (seq scoring).
"""

import argparse
import io
import json
import os
import random
import sys
import time

import torch
import zhconv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_testset import evaluate  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "testset")
BASE_TOK = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"


def load_requests(path):
    target = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                target[int(parts[0])] = parts[2]
    return target


def load_meta(path):
    meta = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                meta[int(parts[0])] = (parts[1], parts[2], parts[3])
    return meta


def load_candidates(path):
    result = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            rid = int(parts[0])
            cands = []
            if len(parts) > 2 and parts[2]:
                for c in parts[2].split("\x1f"):
                    text = c.split("\x1e")[0]
                    if text:
                        cands.append(text)
            result[rid] = cands
    return result


def build_schema_cases(schema, quotas, seed):
    req = load_requests(os.path.join(OUT_DIR, "requests.tsv"))
    meta = load_meta(os.path.join(OUT_DIR, "requests_meta.tsv"))
    cands = load_candidates(os.path.join(OUT_DIR,
                                         "candidates_%s.tsv" % schema))
    rng = random.Random(seed)
    by_kind = {"char": [], "word": []}
    not_found = {"char": 0, "word": 0}
    total = {"char": 0, "word": 0}
    for rid, target in req.items():
        kind, domain, context = meta[rid]
        total[kind] += 1
        simp = [zhconv.convert(c, "zh-cn") for c in cands.get(rid, [])]
        target_s = zhconv.convert(target, "zh-cn")
        rank = None
        for i, c in enumerate(simp):
            if c == target_s:
                rank = i
                break
        if rank is None:
            not_found[kind] += 1
        case = {"id": rid, "kind": kind, "domain": domain,
                "context": context, "target": target_s,
                "candidates": simp, "ime_rank": rank}
        by_kind[kind].append(case)
    selected = []
    stats = {"schema": schema, "not_found": not_found, "total": total}
    for kind in ("char", "word"):
        hits = [c for c in by_kind[kind] if c["ime_rank"] is not None]
        rng.shuffle(hits)
        selected.extend(hits[:quotas[kind]])
        stats["%s_hits" % kind] = len(hits)
        stats["%s_selected" % kind] = min(len(hits), quotas[kind])
    rng.shuffle(selected)
    out = os.path.join(OUT_DIR, "testset_real_%s.json" % schema)
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": selected, "stats": stats}, f, ensure_ascii=False,
                  indent=1)
    print("saved", out, "cases", len(selected), "stats", stats, flush=True)
    return selected


def metrics(cases, scores):
    per = {}
    base = {}
    for ci, case in enumerate(cases):
        if case["ime_rank"] is None:
            continue
        sc = scores[ci]
        if sc is None or any(s is None for s in sc):
            continue
        hit = case["candidates"].index(case["target"])
        ime_rank = case["ime_rank"] + 1
        order = sorted(range(len(sc)), key=lambda k: -sc[k])
        lm_rank = order.index(hit) + 1
        for tag in (case["kind"], case["domain"], "ALL"):
            for store, rank in ((base, ime_rank), (per, lm_rank)):
                d = store.setdefault(tag, {"n": 0, "top1": 0, "top3": 0,
                                           "mrr": 0.0, "rank_sum": 0})
                d["n"] += 1
                d["top1"] += rank == 1
                d["top3"] += rank <= 3
                d["mrr"] += 1.0 / rank
                d["rank_sum"] += rank
    for store in (base, per):
        for d in store.values():
            d["top1"] /= d["n"]
            d["top3"] /= d["n"]
            d["mrr"] /= d["n"]
            d["avg_rank"] = d["rank_sum"] / d["n"]
            del d["rank_sum"]
    base_overall = base.pop("ALL")
    lm_overall = per.pop("ALL")
    return base_overall, lm_overall, base, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--quota", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(OUT_DIR,
                                                  "results_real.jsonl"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    quotas = {"char": args.quota, "word": args.quota}
    cases = build_schema_cases(args.schema, quotas, args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_TOK)
    unk = tok.unk_token_id
    records = []
    for ci, case in enumerate(cases):
        ctx_ids = tuple(tok.encode(case["context"], add_special_tokens=False))
        for yi, cand in enumerate(case["candidates"]):
            cand_ids = tuple(tok.convert_tokens_to_ids(ch) for ch in cand)
            records.append((ctx_ids, cand_ids, ci, yi))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    with io.open(args.list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name, path = line.split("\t", 1)
                models.append((name, path))

    with io.open(args.out, "a", encoding="utf-8") as out:
        for name, path in models:
            t0 = time.perf_counter()
            scores = evaluate(cases, records, tok, path, device, args.batch,
                              "seq")
            base, lm, base_d, lm_d = metrics(cases, scores)
            dt = time.perf_counter() - t0
            result = {"schema": args.schema, "model": name,
                      "ime": base, "lm": lm, "seconds": round(dt, 1),
                      "ime_kind": {k: v for k, v in base_d.items()
                                   if k in ("char", "word")},
                      "lm_kind": {k: v for k, v in lm_d.items()
                                  if k in ("char", "word")}}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print("%-8s %-16s IME top1=%.3f mrr=%.3f avg_rank=%.2f | "
                  "LM top1=%.3f mrr=%.3f avg_rank=%.2f | d_top1=%+.3f" %
                  (args.schema, name, base["top1"], base["mrr"],
                   base["avg_rank"], lm["top1"], lm["mrr"], lm["avg_rank"],
                   lm["top1"] - base["top1"]), flush=True)


if __name__ == "__main__":
    main()
