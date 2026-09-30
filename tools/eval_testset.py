"""Batched GPU evaluation of checkpoints on IME test sets (1..N char candidates)."""

import argparse
import io
import json
import os
import time

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TOK = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"
DATA = os.path.join(HERE, "testset", "testset.json")
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"


def load_freq():
    import zhconv

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


def evaluate(cases, records, tokenizer, model_path, device, batch_size,
             scoring):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    scores = [None] * len(cases)
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    mask = tokenizer.mask_token_id
    pad = tokenizer.pad_token_id

    tasks = []
    for r in records:
        ctx_ids, cand_ids, ci, yi = r
        if cand_ids is None:
            continue
        if scoring == "maskall":
            body = list(ctx_ids) + [mask] * len(cand_ids)
            tasks.append((body, [1 + len(ctx_ids) + k
                                 for k in range(len(cand_ids))],
                          list(cand_ids), ci, yi))
        else:
            for k in range(len(cand_ids)):
                body = list(ctx_ids) + [cand_ids[j] if j != k else mask
                                        for j in range(len(cand_ids))]
                tasks.append((body, [1 + len(ctx_ids) + k],
                              [cand_ids[k]], ci, yi))
    tasks.sort(key=lambda t: len(t[0]))
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks),
                                                     batch_size)]
    print("scoring=%s forward tasks=%d batches=%d" %
          (scoring, len(tasks), len(batches)), flush=True)

    with torch.inference_mode():
        for batch in batches:
            max_len = max(len(t[0]) + 2 for t in batch)
            ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
            att = torch.zeros((len(batch), max_len), dtype=torch.long)
            rows, poss, cids, metas = [], [], [], []
            for bi, (body, positions, target_ids, ci, yi) in enumerate(batch):
                row = [cls] + body + [sep]
                ids[bi, :len(row)] = torch.tensor(row, dtype=torch.long)
                att[bi, :len(row)] = 1
                for pos, tid in zip(positions, target_ids):
                    rows.append(bi)
                    poss.append(pos)
                    cids.append(tid)
                    metas.append((ci, yi))
            ids = ids.to(device)
            att = att.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits = model(input_ids=ids, attention_mask=att).logits
            sel = logits[torch.tensor(rows, device=device),
                         torch.tensor(poss, device=device)].float()
            logp = sel - torch.logsumexp(sel, dim=-1, keepdim=True)
            vals = logp[torch.arange(len(cids), device=device),
                        torch.tensor(cids, device=device)]
            agg = {}
            for (ci, yi), v in zip(metas, vals.tolist()):
                agg[(ci, yi)] = agg.get((ci, yi), 0.0) + v
            for (ci, yi), v in agg.items():
                if scores[ci] is None:
                    scores[ci] = [None] * len(cases[ci]["candidates"])
                scores[ci][yi] = v

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def compute_metrics(cases, scores):
    per_domain = {}
    for ci, case in enumerate(cases):
        sc = scores[ci]
        if sc is None or any(s is None for s in sc):
            continue
        order = sorted(range(len(sc)), key=lambda k: -sc[k])
        rank = order.index(case["candidates"].index(case["target"])) + 1
        tags = [case["domain"]]
        if "length" in case:
            tags.append("len%d" % case["length"])
        for tag in tags + ["ALL"]:
            d = per_domain.setdefault(tag, {"n": 0, "top1": 0, "top3": 0,
                                            "mrr": 0.0})
            d["n"] += 1
            d["top1"] += rank == 1
            d["top3"] += rank <= 3
            d["mrr"] += 1.0 / rank
    for d in per_domain.values():
        d["top1"] /= d["n"]
        d["top3"] /= d["n"]
        d["mrr"] /= d["n"]
    overall = per_domain.pop("ALL")
    return overall, per_domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "testset",
                                                  "results.jsonl"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--scoring", choices=("maskall", "seq"), default="maskall")
    args = ap.parse_args()

    with io.open(args.data, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    print("cases:", len(cases), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_TOK)
    unk = tok.unk_token_id

    records = []
    for ci, case in enumerate(cases):
        ctx_ids = tuple(tok.encode(case["context"], add_special_tokens=False))
        for yi, cand in enumerate(case["candidates"]):
            cand_ids = tuple(tok.convert_tokens_to_ids(ch) for ch in cand)
            ok = all(c is not None and c != unk for c in cand_ids)
            records.append((ctx_ids, cand_ids if ok else None, ci, yi))
    valid = [i for i, r in enumerate(records) if r[1] is not None]
    print("candidate records:", len(valid), "/", len(records), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    freq = load_freq()
    fb = {"n": 0, "top1": 0, "top3": 0, "mrr": 0.0}
    for case in cases:
        cands = case["candidates"]
        order = sorted(range(len(cands)),
                       key=lambda k: -freq.get(cands[k], 0.0))
        rank = order.index(cands.index(case["target"])) + 1
        fb["n"] += 1
        fb["top1"] += rank == 1
        fb["top3"] += rank <= 3
        fb["mrr"] += 1.0 / rank
    fb["top1"] /= fb["n"]
    fb["top3"] /= fb["n"]
    fb["mrr"] /= fb["n"]
    print("freq baseline: top1=%.3f top3=%.3f mrr=%.3f" %
          (fb["top1"], fb["top3"], fb["mrr"]), flush=True)

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
                              args.scoring)
            overall, per_domain = compute_metrics(cases, scores)
            dt = time.perf_counter() - t0
            result = {"name": name, "path": path, "scoring": args.scoring,
                      "overall": overall,
                      "domains": per_domain, "seconds": round(dt, 1)}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print("%-18s top1=%.4f top3=%.4f mrr=%.4f (%.1fs) %s" %
                  (name, overall["top1"], overall["top3"], overall["mrr"], dt,
                   " ".join("%s=%.3f" % (k, v["top1"])
                            for k, v in sorted(per_domain.items()))),
                  flush=True)


if __name__ == "__main__":
    main()
