"""Sweep alpha (prior trust) and override threshold tau on the reranker val set."""

import argparse
import io
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_reranker import Reranker, build_sequences, masked_z  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
RERANK = os.path.join(HERE, "reranker")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(RERANK, "data.npz"))
    ap.add_argument("--model", default=os.path.join(RERANK, "model",
                                                    "model.pt"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--alphas", default="0.5,1,2,4,8")
    ap.add_argument("--taus", default="0,0.25,0.5,1,2,4")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    base = ckpt["base"]
    type_vocab = ckpt["type_vocab"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = Reranker(base, len(type_vocab),
                     feat_dim=1 + (1 if ckpt.get("concat_score") else 0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    z = np.load(args.data)
    va = np.where(z["split"] == 1)[0]
    ds = TensorDataset(
        torch.from_numpy(z["ctx"][va].astype(np.int64)),
        torch.from_numpy(z["cand"][va].astype(np.int64)),
        torch.from_numpy(z["logq"][va].astype(np.float32)),
        torch.from_numpy(z["rankn"][va].astype(np.float32)),
        torch.from_numpy(z["types"][va].astype(np.int64)),
        torch.from_numpy(z["mask"][va].astype(np.float32)),
        torch.from_numpy(z["target_pos"][va].astype(np.int64)))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    all_scores, all_priors, all_targets, all_masks = [], [], [], []
    for (ctx, cand, logq, rankn, types, mask, target_pos) in dl:
        ctx = ctx.to(device)
        cand = cand.to(device)
        mask_d = mask.to(device)
        zz = masked_z(logq.to(device), mask_d)
        ids, att = build_sequences(ctx, cand, tok.cls_token_id,
                                   tok.sep_token_id)
        out = model(ids, att, types.reshape(-1).to(device),
                    rankn.reshape(-1).to(device),
                    zz.reshape(-1) if ckpt.get("concat_score") else None)
        out = out.reshape(ctx.shape[0], -1)
        all_scores.append(out.cpu())
        all_priors.append(logq)
        all_targets.append(target_pos)
        all_masks.append(mask)
    out = torch.cat(all_scores)
    logq = torch.cat(all_priors)
    target = torch.cat(all_targets)
    mask = torch.cat(all_masks)
    valid_n = mask.sum(dim=1)
    # target index in the packed (quality-sorted) list; prior top1 = 0
    prior_top1 = (target == 0)
    print("val n=%d prior_top1=%.4f" %
          (out.shape[0], prior_top1.float().mean().item()), flush=True)

    for alpha in [float(x) for x in args.alphas.split(",")]:
        zz = masked_z(logq, mask)
        score = alpha * zz + out
        score = score.masked_fill(mask == 0, -1e9)
        for tau in [float(x) for x in args.taus.split(",")]:
            # candidate to promote: best non-prior candidate
            srt = score.argsort(dim=1, descending=True)
            best = srt[:, 0]
            margin = score.gather(1, best.unsqueeze(1)) - \
                score.gather(1, torch.zeros_like(best).unsqueeze(1))
            apply_ = ((best != 0) & (margin.squeeze(1) > tau))
            final = torch.where(apply_, best, torch.zeros_like(best))
            o = (final == target)
            churn = float((final != 0).float().mean())
            over = int((final != 0).sum())
            over_ok = int(((final != 0) & o).sum())
            print("alpha=%.2f tau=%.2f top1=%.4f churn=%.3f "
                  "overrides=%d precision=%.3f" %
                  (alpha, tau, o.float().mean().item(), churn, over,
                   (over_ok / over) if over else 1.0), flush=True)


if __name__ == "__main__":
    main()
