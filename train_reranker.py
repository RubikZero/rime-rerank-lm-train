"""Train a single-char neural reranker for pinyin candidates.

Design: the librime score is NOT part of the prompt. It enters as a standalone
float feature:
    final_score = alpha * z_case(log_quality) + head([CLS]_repr, feat)
where feat = [rank_norm, type_embedding] (score itself can also be optionally
concatenated via --concat-score). The head is zero-initialized so training
starts from the IME prior order.
"""

import argparse
import io
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RERANK = os.path.join(HERE, "reranker")
BASE = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"


class Reranker(nn.Module):
    def __init__(self, base_dir, n_types, hidden=128, feat_dim=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_dir)
        self.type_emb = nn.Embedding(n_types, 8)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size + 8 + feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, ids, att, types, rankn, feat_extra):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        cls = h[:, 0]
        feats = [self.type_emb(types), rankn.unsqueeze(-1)]
        if feat_extra is not None:
            feats.append(feat_extra.unsqueeze(-1))
        x = torch.cat([cls] + feats, dim=-1)
        return self.head(x).squeeze(-1)


def build_sequences(ctx, cand, cls_id, sep_id):
    B, C = cand.shape
    ctx_e = ctx.unsqueeze(1).expand(B, C, ctx.shape[1]).reshape(B * C, -1)
    cand_f = cand.reshape(B * C, 1)
    one = torch.ones(B * C, 1, dtype=torch.long, device=ctx.device)
    ids = torch.cat([one * cls_id, ctx_e, one * sep_id, cand_f,
                     one * sep_id], dim=1)
    att = torch.cat([one, (ctx_e != 0).long(), one, one, one], dim=1)
    return ids, att


def masked_z(x, mask):
    x = x * mask
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = x.sum(dim=1, keepdim=True) / n
    var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / n
    std = var.sqrt().clamp(min=1e-6)
    return (x - mean) / std * mask


def run_epoch(model, loader, optimizer, scheduler, device, cfg, epoch,
              mlflow_step=None):
    model.train()
    total, count = 0.0, 0
    accum = max(1, cfg.get("accum", 1))
    t0 = time.perf_counter()
    step = -1
    for step, batch in enumerate(loader):
        (ctx, cand, logq, rankn, types, mask, target_pos) = [
            t.to(device) for t in batch]
        z = masked_z(logq, mask)
        ids, att = build_sequences(ctx, cand, cfg["cls_id"], cfg["sep_id"])
        out = model(ids, att, types.reshape(-1), rankn.reshape(-1),
                    z.reshape(-1) if cfg["concat_score"] else None)
        out = out.reshape(ctx.shape[0], -1)
        scores = cfg["alpha"] * z + out
        scores = scores.masked_fill(mask == 0, -1e9)
        loss = F.cross_entropy(scores, target_pos)
        loss = loss + cfg["shrink"] * (out ** 2).mean()
        (loss / accum).backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
        count += 1
        if (step + 1) % 200 == 0:
            speed = (step + 1) * ctx.shape[0] * cand.shape[1] * ids.shape[1] / (
                time.perf_counter() - t0)
            print("epoch %d step %d loss=%.4f %.0f tok/s" %
                  (epoch + 1, step + 1, total / count, speed), flush=True)
            if mlflow_step:
                mlflow_step(total / count, (epoch * len(loader)) + step + 1)
            total, count = 0.0, 0
            t0 = time.perf_counter()
    if (step + 1) % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, cfg):
    model.eval()
    n_cases = top1 = top3 = 0
    mrr = 0.0
    override = override_ok = 0
    prior_top1 = prior_top3 = 0
    for batch in loader:
        (ctx, cand, logq, rankn, types, mask, target_pos) = [
            t.to(device) for t in batch]
        z = masked_z(logq, mask)
        ids, att = build_sequences(ctx, cand, cfg["cls_id"], cfg["sep_id"])
        out = model(ids, att, types.reshape(-1), rankn.reshape(-1),
                    z.reshape(-1) if cfg["concat_score"] else None)
        scores = (cfg["alpha"] * z + out.reshape(ctx.shape[0], -1))
        scores = scores.masked_fill(mask == 0, -1e9)
        order = scores.argsort(dim=1, descending=True)
        tgt = target_pos.unsqueeze(1)
        rank = (order == tgt).nonzero()[:, 1] + 1
        n_cases += ctx.shape[0]
        top1 += int((rank == 1).sum())
        top3 += int((rank <= 3).sum())
        mrr += float((1.0 / rank.float()).sum())
        prior_rank = target_pos + 1
        prior_top1 += int((prior_rank == 1).sum())
        prior_top3 += int((prior_rank <= 3).sum())
        changed = order[:, 0] != 0
        override += int(changed.sum())
        correct = (order[:, 0] == target_pos)
        override_ok += int((changed & correct).sum())
    return {
        "n": n_cases,
        "top1": top1 / n_cases,
        "top3": top3 / n_cases,
        "mrr": mrr / n_cases,
        "prior_top1": prior_top1 / n_cases,
        "prior_top3": prior_top3 / n_cases,
        "override_rate": override / n_cases,
        "override_precision": (override_ok / override) if override else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(RERANK, "data.npz"))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=os.path.join(RERANK, "model"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--val-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train-samples", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=1e-4)
    ap.add_argument("--concat-score", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--run-name", default="reranker-v1")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    mlflow = None
    if not args.no_mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri("sqlite:///" + os.path.join(
                os.path.dirname(HERE), "mlflow.db").replace("\\", "/"))
            mlflow.set_experiment("lm-ranker")
            mlflow.start_run(run_name=args.run_name)
            mlflow.log_params({
                "base": os.path.basename(args.base), "epochs": args.epochs,
                "batch": args.batch, "lr": args.lr,
                "train_samples": args.train_samples, "alpha": args.alpha,
                "shrink": args.shrink, "concat_score": args.concat_score,
                "K": int(np.load(args.data)["cand"].shape[1])})
        except Exception as exc:  # noqa: BLE001
            print("mlflow disabled:", exc, flush=True)
            mlflow = None

    z = np.load(args.data)
    type_vocab = json.load(io.open(
        os.path.join(os.path.dirname(args.data), "type_vocab.json"),
        encoding="utf-8"))
    split = z["split"]
    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(tr)
    tr = tr[:args.train_samples]
    print("train=%d val=%d C=%d" %
          (len(tr), len(va), z["cand"].shape[1]), flush=True)

    def make(idx):
        return TensorDataset(
            torch.from_numpy(z["ctx"][idx].astype(np.int64)),
            torch.from_numpy(z["cand"][idx].astype(np.int64)),
            torch.from_numpy(z["logq"][idx].astype(np.float32)),
            torch.from_numpy(z["rankn"][idx].astype(np.float32)),
            torch.from_numpy(z["types"][idx].astype(np.int64)),
            torch.from_numpy(z["mask"][idx].astype(np.float32)),
            torch.from_numpy(z["target_pos"][idx].astype(np.int64)))

    train_dl = DataLoader(make(tr), batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(make(va), batch_size=args.val_batch, shuffle=False,
                        num_workers=0)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = Reranker(args.base, len(type_vocab),
                     feat_dim=1 + (1 if args.concat_score else 0)).to(device)
    cfg = {"cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id,
           "alpha": args.alpha, "shrink": args.shrink,
           "concat_score": args.concat_score, "accum": args.accum}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = (len(train_dl) // max(1, args.accum)) * args.epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.05), total_steps)

    mlflow_step = None
    if mlflow:
        mlflow_step = lambda loss, step: mlflow.log_metric(
            "train_loss", loss, step=step)  # noqa: E731

    best_top1 = -1.0
    metrics = {}
    for epoch in range(args.epochs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            run_epoch(model, train_dl, optimizer, scheduler, device, cfg,
                      epoch, mlflow_step)
        metrics = evaluate(model, val_dl, device, cfg)
        print("epoch %d: %s" % (epoch + 1, metrics), flush=True)
        if mlflow:
            for k, v in metrics.items():
                mlflow.log_metric("val_" + k, v, step=epoch + 1)
        os.makedirs(args.out, exist_ok=True)
        payload = {"state_dict": model.state_dict(), "base": args.base,
                   "type_vocab": type_vocab, "alpha": args.alpha,
                   "concat_score": args.concat_score,
                   "K": int(z["cand"].shape[1]), "metrics": metrics,
                   "epoch": epoch + 1}
        torch.save(payload, os.path.join(args.out, "model_last.pt"))
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save(payload, os.path.join(args.out, "model_best.pt"))
            print("  new best top1=%.4f" % best_top1, flush=True)

    torch.save({"state_dict": model.state_dict(), "base": args.base,
                "type_vocab": type_vocab, "alpha": args.alpha,
                "concat_score": args.concat_score,
                "K": int(z["cand"].shape[1]),
                "metrics": metrics}, os.path.join(args.out, "model.pt"))
    print("saved", args.out, "(best top1=%.4f)" % best_top1, flush=True)
    if mlflow:
        mlflow.log_param("model_path", args.out)
        mlflow.end_run()


if __name__ == "__main__":
    main()
