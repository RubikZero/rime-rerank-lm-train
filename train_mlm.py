"""Task-adaptive MLM continued training for small Chinese BERT models."""

import argparse
import io
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "train.npz"))
    ap.add_argument("--val", default=os.path.join(HERE, "data", "val.npz"))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default=os.path.join(HERE, "train.log"))
    ap.add_argument("--mlflow-uri", default="")
    ap.add_argument("--mlflow-experiment", default="lm-ranker")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    mlog = lambda *a, **k: None  # noqa: E731
    mlflow_run = False
    if not args.no_mlflow:
        try:
            import mlflow
            uri = args.mlflow_uri or ("sqlite:///" + os.path.join(
                os.path.dirname(HERE), "mlflow.db").replace("\\", "/"))
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(args.mlflow_experiment)
            run_name = args.run_name or os.path.basename(
                os.path.abspath(args.out))
            mlflow.start_run(run_name=run_name)
            mlflow_run = True
            mlflow.log_params({
                "base": os.path.basename(args.base),
                "data": os.path.basename(os.path.dirname(args.data)) + "/" +
                        os.path.basename(args.data),
                "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                "seed": args.seed, "max_steps": args.max_steps})
            mlog = mlflow.log_metric
            print("mlflow tracking:", uri, "run:", run_name, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("mlflow disabled:", exc, flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base)
    model.to(device)
    model.train()

    def load_split(path):
        z = np.load(path)
        return TensorDataset(torch.from_numpy(z["ids"].astype(np.int32)),
                             torch.from_numpy(z["mask"].astype(np.int32)),
                             torch.from_numpy(z["labels"].astype(np.int32)))

    train_ds = load_split(args.data)
    val_ds = load_split(args.val)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    log_f = io.open(args.log, "a", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    log("train samples=%d steps_per_epoch=%d total_steps=%d batch=%d lr=%g"
        % (len(train_ds), steps_per_epoch, total_steps, args.batch, args.lr))

    os.makedirs(args.out, exist_ok=True)
    global_step = 0
    running = 0.0
    t0 = time.perf_counter()
    done = False
    for epoch in range(args.epochs):
        for ids, mask, labels in train_dl:
            ids = ids.to(device, non_blocking=True).long()
            mask = mask.to(device, non_blocking=True).long()
            labels = labels.to(device, non_blocking=True).long()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            running += float(loss.detach())

            if global_step % 50 == 0:
                elapsed = time.perf_counter() - t0
                lr_now = scheduler.get_last_lr()[0]
                log("step %d/%d loss=%.4f lr=%.2e %.1f tok/s"
                    % (global_step, total_steps, running / 50, lr_now,
                       50 * args.batch * ids.shape[1] / elapsed))
                mlog("train_loss", running / 50, step=global_step)
                mlog("lr", lr_now, step=global_step)
                running = 0.0
                t0 = time.perf_counter()

            if args.save_steps and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.out, "step-%d" % global_step)
                save_checkpoint(model, tok, save_dir)
                log("saved checkpoint: " + save_dir)

            if args.max_steps and global_step >= args.max_steps:
                done = True
                break
        if done:
            break

    final_dir = os.path.join(args.out, "final")
    save_checkpoint(model, tok, final_dir)
    log("training done, final checkpoint: " + final_dir)

    model.eval()
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    total, count = 0.0, 0
    with torch.inference_mode():
        for ids, mask, labels in val_dl:
            ids = ids.to(device).long()
            mask = mask.to(device).long()
            labels = labels.to(device).long()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss = model(input_ids=ids, attention_mask=mask,
                             labels=labels).loss
            total += float(loss)
            count += 1
    val_loss = total / max(1, count)
    log("validation loss: %.4f" % val_loss)
    mlog("val_loss", val_loss, step=global_step)
    if mlflow_run:
        try:
            import mlflow
            mlflow.log_param("final_checkpoint", final_dir)
            mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass


def save_checkpoint(model, tok, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
