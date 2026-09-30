"""Diagnose the trained reranker: context-only, per-prior-rank accuracy, samples."""

import argparse
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
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--split", type=int, default=1,
                    help="0=train, 1=val")
    ap.add_argument("--limit", type=int, default=0)
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
    va = np.where(z["split"] == args.split)[0]
    if args.limit:
        va = va[:args.limit]
    ctx_np = z["ctx"][va]
    cand_np = z["cand"][va]
    logq_np = z["logq"][va]
    rankn_np = z["rankn"][va]
    types_np = z["types"][va]
    mask_np = z["mask"][va]
    tgt_np = z["target_pos"][va]

    ds = TensorDataset(
        torch.from_numpy(ctx_np.astype(np.int64)),
        torch.from_numpy(cand_np.astype(np.int64)),
        torch.from_numpy(logq_np.astype(np.float32)),
        torch.from_numpy(rankn_np.astype(np.float32)),
        torch.from_numpy(types_np.astype(np.int64)),
        torch.from_numpy(mask_np.astype(np.float32)),
        torch.from_numpy(tgt_np.astype(np.int64)))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    outs, priors, targets, masks = [], [], [], []
    for (ctx, cand, logq, rankn, types, mask, target_pos) in dl:
        ctx = ctx.to(device)
        cand = cand.to(device)
        m = mask.to(device)
        zz = masked_z(logq.to(device), m)
        ids, att = build_sequences(ctx, cand, tok.cls_token_id,
                                   tok.sep_token_id)
        out = model(ids, att, types.reshape(-1).to(device),
                    rankn.reshape(-1).to(device),
                    zz.reshape(-1) if ckpt.get("concat_score") else None)
        outs.append(out.reshape(ctx.shape[0], -1).cpu())
        priors.append(logq)
        targets.append(target_pos)
        masks.append(mask)
    out = torch.cat(outs)
    logq = torch.cat(priors)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    zz = masked_z(logq, mask)

    def top1(score):
        score = score.masked_fill(mask == 0, -1e9)
        return (score.argmax(dim=1) == target).float().mean().item()

    print("n=%d  prior_top1=%.4f  context_only(alpha=0)=%.4f  "
          "with_prior(alpha=1)=%.4f" %
          (out.shape[0], (target == 0).float().mean().item(),
           top1(out), top1(zz + out)), flush=True)

    score = (zz + out).masked_fill(mask == 0, -1e9)
    arg = score.argmax(dim=1)
    buckets = [(0, 0), (1, 2), (3, 9), (10, 20)]
    for lo, hi in buckets:
        sel = (target >= lo) & (target <= hi)
        if sel.sum() == 0:
            continue
        acc = (arg[sel] == target[sel]).float().mean().item()
        changed = (arg[sel] != 0).float().mean().item()
        print("prior_rank %d-%d: n=%d model_top1=%.4f override=%.3f" %
              (lo, hi, int(sel.sum()), acc, changed), flush=True)

    # examples where model overrides
    print("\n-- override examples (context | target <- model) --")
    picks = ((arg != 0)).nonzero().squeeze(1)[:args.examples]
    idmap = {v: k for k, v in tok.get_vocab().items()}
    for i in picks.tolist():
        ctx_ids = [int(x) for x in ctx_np[i] if x != 0]
        ctx = "".join(idmap.get(x, "?") for x in ctx_ids)[-24:]
        cands = [idmap.get(int(c), "?") for c in cand_np[i][mask_np[i] == 1]]
        tgt = idmap.get(int(cand_np[i][target[i]]), "?")
        pick = idmap.get(int(cand_np[i][arg[i]]), "?")
        print("%s | %s <- %s   (prior=%s)" % (ctx, tgt, pick, cands[0]))


if __name__ == "__main__":
    main()
