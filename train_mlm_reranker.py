"""Candidate-constrained softmax fine-tuning of the MLM for pinyin reranking.

Forward: [CLS] ctx(256) [MASK] [SEP] -> logits at the mask position.
Loss: cross entropy over the candidate set only (target vs top-20 candidates).
One forward per case (no cross-encoder expansion). Mid-epoch validation.
"""

import argparse
import io
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "reranker_v2", "data.npz")


def build_inputs(ctx, mask_tok, cls_id, sep_id):
    b = ctx.shape[0]
    one = torch.ones(b, 1, dtype=torch.long, device=ctx.device)
    ids = torch.cat([one * cls_id, ctx, one * mask_tok, one * sep_id], dim=1)
    att = torch.cat([one, (ctx != 0).long(), one, one], dim=1)
    return ids, att


@torch.inference_mode()
def evaluate(model, loader, device, cls_id, sep_id, mask_tok):
    model.eval()
    n = top1 = top3 = 0
    mrr = 0.0
    prior_top1 = 0
    buckets = {}
    for ctx, cand, mask, target_pos, logq in loader:
        ctx = ctx.to(device)
        cand = cand.to(device)
        mask = mask.to(device)
        target_pos = target_pos.to(device)
        ids, att = build_inputs(ctx, mask_tok, cls_id, sep_id)
        logits = model(input_ids=ids, attention_mask=att).logits[:, -2, :]
        score = logits.gather(1, cand).masked_fill(mask == 0, -1e9)
        order = score.argsort(dim=1, descending=True)
        rank = (order == target_pos.unsqueeze(1)).nonzero()[:, 1] + 1
        n += ctx.shape[0]
        top1 += int((rank == 1).sum())
        top3 += int((rank <= 3).sum())
        mrr += float((1.0 / rank.float()).sum())
        prior_rank = target_pos + 1
        prior_top1 += int((prior_rank == 1).sum())
        for lo, hi, name in ((0, 0, "rank1"), (1, 2, "rank2-3"),
                             (3, 8, "rank4-9"), (9, 99, "rank10up")):
            sel = (target_pos >= lo) & (target_pos <= hi)
            k = int(sel.sum())
            if k == 0:
                continue
            b = buckets.setdefault(name, [0, 0])
            b[0] += k
            b[1] += int((rank[sel] == 1).sum())
    out = {"n": n, "top1": top1 / n, "top3": top3 / n, "mrr": mrr / n,
           "prior_top1": prior_top1 / n}
    for name, (k, ok) in buckets.items():
        out["top1_" + name] = ok / k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    mlflow = None
    if args.mlflow:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri("sqlite:///" + os.path.join(
            os.path.dirname(HERE), "mlflow.db").replace("\\", "/"))
        mlflow.set_experiment("lm-ranker")
        mlflow.start_run(run_name=args.run_name or
                         os.path.basename(os.path.abspath(args.out)))
        mlflow.log_params({"base": os.path.basename(os.path.dirname(args.base))
                           + "/" + os.path.basename(args.base),
                           "epochs": args.epochs, "batch": args.batch,
                           "lr": args.lr, "objective": "candidate_softmax"})

    z = np.load(args.data)
    split = z["split"]
    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    C = z["cand"].shape[1]

    def make(idx):
        return TensorDataset(
            torch.from_numpy(z["ctx"][idx].astype(np.int64)),
            torch.from_numpy(z["cand"][idx].astype(np.int64)),
            torch.from_numpy(z["mask"][idx].astype(np.float32)),
            torch.from_numpy(z["target_pos"][idx].astype(np.int64)),
            torch.from_numpy(z["logq"][idx].astype(np.float32)))

    train_dl = DataLoader(make(tr), batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(make(va), batch_size=64, shuffle=False, num_workers=0)
    print("train=%d val=%d C=%d" % (len(tr), len(va), C), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    steps_per_epoch = len(train_dl)
    total_steps = int(steps_per_epoch * args.epochs)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.05), total_steps)
    eval_every = max(1, int(steps_per_epoch * args.eval_frac))
    accum = max(1, args.accum)

    os.makedirs(args.out, exist_ok=True)
    best = -1.0
    step = 0
    running = 0.0
    count = 0
    t0 = time.perf_counter()
    stop = False
    while not stop:
        for ctx, cand, mask, target_pos, logq in train_dl:
            ctx = ctx.to(device)
            cand = cand.to(device)
            mask = mask.to(device)
            target_pos = target_pos.to(device)
            ids, att = build_inputs(ctx, tok.mask_token_id, tok.cls_token_id,
                                    tok.sep_token_id)
            logits = model(input_ids=ids, attention_mask=att).logits[:, -2, :]
            score = logits.gather(1, cand).masked_fill(mask == 0, -1e9)
            loss = F.cross_entropy(score, target_pos)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            running += float(loss.detach())
            count += 1
            if step % 200 == 0:
                print("step %d/%d loss=%.4f lr=%.2e %.0f tok/s" %
                      (step, total_steps, running / count,
                       scheduler.get_last_lr()[0],
                       count * args.batch * ids.shape[1] /
                       (time.perf_counter() - t0)), flush=True)
                if mlflow:
                    mlflow.log_metric("train_loss", running / count, step=step)
                running, count = 0.0, 0
                t0 = time.perf_counter()
            if step % eval_every == 0 or step >= total_steps:
                metrics = evaluate(model, val_dl, device, tok.cls_token_id,
                                   tok.sep_token_id, tok.mask_token_id)
                print("step %d metrics: %s" % (step, metrics), flush=True)
                if mlflow:
                    for k, v in metrics.items():
                        if k != "n":
                            mlflow.log_metric("val_" + k, v, step=step)
                if metrics["top1"] > best:
                    best = metrics["top1"]
                    model.save_pretrained(args.out)
                    tok.save_pretrained(args.out)
                    print("  new best top1=%.4f saved" % best, flush=True)
                model.train()
            if step >= total_steps:
                stop = True
                break

    print("done. best val top1=%.4f -> %s" % (best, args.out), flush=True)
    if mlflow:
        mlflow.log_metric("best_val_top1", best)
        mlflow.end_run()


if __name__ == "__main__":
    main()
