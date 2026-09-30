"""Large-scale evaluation of LM re-ranking with frequency prior fusion.

Re-ranks homophone candidates using z-normalized LM scores plus a weighted
frequency prior, which mirrors how the filter will run on top of the existing
IME candidate order.
"""

import argparse
import io
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from eval_models import (DATA, MODELS, build_cases, load_vocab_chars,
                         parse_dict, parse_freq, to_simplified)


def build_freq(essay_path, simp):
    freq = parse_freq(essay_path)
    if not simp:
        return freq
    import zhconv

    merged = {}
    for k, v in freq.items():
        k2 = zhconv.convert(k, "zh-cn")
        merged[k2] = merged.get(k2, 0.0) + v
    return merged


def zscore(values):
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELS[1][1])
    ap.add_argument("--name", default="model")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--simp", action="store_true")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--betas", default="0,0.25,0.5,1.0,2.0,4.0")
    args = ap.parse_args()

    vocab = load_vocab_chars(os.path.join(MODELS[0][1], "vocab.txt"))
    words = parse_dict(os.path.join(DATA, "luna_pinyin.dict.yaml"))
    freq = build_freq(os.path.join(DATA, "essay.txt"), args.simp)
    cases = build_cases(words, vocab, n=args.n)
    if args.simp:
        cases = to_simplified(cases)
    transform = None
    if args.cross:
        import zhconv

        transform = lambda s: zhconv.convert(s, "zh-cn")
    print("model=%s cases=%d simp=%s cross=%s" %
          (args.name, len(cases), args.simp, args.cross), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    mask_id = tok.mask_token_id
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    unk_id = tok.unk_token_id

    def score(context, cand):
        if transform is not None:
            context = transform(context)
            cand = transform(cand)
        ctx_ids = tok.encode(context, add_special_tokens=False)
        cand_ids = [tok.convert_tokens_to_ids(ch) for ch in cand]
        if any(i in (unk_id, None) for i in cand_ids):
            return None
        ids = [cls_id] + ctx_ids + [mask_id] * len(cand_ids) + [sep_id]
        start = 1 + len(ctx_ids)
        with torch.inference_mode():
            logits = model(torch.tensor([ids])).logits[0]
            logp = torch.log_softmax(logits, dim=-1)
        return sum(float(logp[start + k, c]) for k, c in enumerate(cand_ids))

    betas = [float(b) for b in args.betas.split(",")]
    stats = {b: {"top1": 0, "top3": 0, "mrr": 0.0} for b in betas}
    base_stats = {"top1": 0, "top3": 0, "mrr": 0.0}
    kept = 0
    for case in cases:
        cands = case["candidates"]
        if transform is not None:
            tf = [transform(c) for c in cands]
            if len(set(tf)) != len(tf):
                continue
        lm = [score(case["context"], c) for c in cands]
        if any(s is None for s in lm):
            continue
        logf = [math.log1p(freq.get(c, 0.0)) for c in cands]
        zlm = zscore(lm)
        zf = zscore(logf)
        kept += 1
        freq_order = sorted(range(len(cands)), key=lambda i: -logf[i])
        rank = freq_order.index(cands.index(case["target"])) + 1
        base_stats["top1"] += rank == 1
        base_stats["top3"] += rank <= 3
        base_stats["mrr"] += 1.0 / rank
        for b in betas:
            comb = zlm + b * zf
            order = list(np.argsort(-comb))
            rank = order.index(cands.index(case["target"])) + 1
            stats[b]["top1"] += rank == 1
            stats[b]["top3"] += rank <= 3
            stats[b]["mrr"] += 1.0 / rank

    def report(tag, s):
        print("%-12s top1=%.3f top3=%.3f mrr=%.3f" %
              (tag, s["top1"] / kept, s["top3"] / kept, s["mrr"] / kept),
              flush=True)

    print("evaluated cases:", kept)
    report("freq_only", base_stats)
    for b in betas:
        report("zlm+%.2ff" % b, stats[b])


if __name__ == "__main__":
    main()
