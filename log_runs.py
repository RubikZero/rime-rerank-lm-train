"""Retro-log the v1-v4 fine-tuning runs into local MLflow.

Checkpoint metrics are logged as series with step = training step, so the run
page shows real curves and the Compare view can align runs.
"""

import io
import json
import os
import re
import sys

import mlflow
from mlflow.tracking import MlflowClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTSET = os.path.join(ROOT, "lm-test", "testset")
EXPERIMENT = "lm-ranker"

RUNS = [
    {"name": "v1", "log": "train-full.log", "out": "checkpoints",
     "data": "data/train.npz", "epochs": 2, "batch": 64, "lr": 3e-5,
     "notes": "long context (ctx>=8), 1.5M samples",
     "ckpts": ["v1_step-5000", "v1_final"]},
    {"name": "v2", "log": "train-v2.log", "out": "checkpoints_v2",
     "data": "data/train.npz", "epochs": 2, "batch": 64, "lr": 3e-5,
     "notes": "short-ratio 0.5, 1.81M samples",
     "ckpts": ["v2_step-5000", "v2_final"]},
    {"name": "v3", "log": "train-v3.log", "out": "checkpoints_v3",
     "data": "data/train.npz", "epochs": 1, "batch": 64, "lr": 1e-5,
     "notes": "continuation from v1 final",
     "ckpts": ["v3_final"]},
    {"name": "v4", "log": "train-v4.log", "out": "checkpoints_v4",
     "data": "data5m/train.npz", "epochs": 2, "batch": 64, "lr": 3e-5,
     "notes": "5M samples, short-ratio 0.3",
     "ckpts": ["v4_step-70000", "v4_final"]},
]

SCHEMAS = ("luna_pinyin_simp", "luna_quanpin", "luna_pinyin_fluency")


def parse_loss_curve(path):
    points = []
    if not os.path.exists(path):
        return points
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"step (\d+)/\d+ loss=([0-9.]+) lr=", line.strip())
            if m:
                points.append((int(m.group(1)), float(m.group(2))))
    return points


def parse_val_loss(path):
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"validation loss: ([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def load_jsonl_by_name(path):
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["name"]] = r
    return out


def load_real():
    out = {}
    path = os.path.join(TESTSET, "results_real.jsonl")
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["schema"], r["model"])] = r
    return out


def ckpt_step(name, final_step):
    m = re.search(r"step-(\d+)", name)
    return int(m.group(1)) if m else final_step


def reset_experiment():
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        return 0
    runs = client.search_runs([exp.experiment_id], max_results=10000)
    for r in runs:
        client.delete_run(r.info.run_id)
    return len(runs)


def main():
    reset = "--reset" in sys.argv
    mlflow.set_tracking_uri("sqlite:///" +
                            os.path.join(ROOT, "mlflow.db").replace("\\", "/"))
    if reset:
        n = reset_experiment()
        print("deleted %d existing runs" % n, flush=True)
    mlflow.set_experiment(EXPERIMENT)
    res5000 = load_jsonl_by_name(os.path.join(TESTSET, "results.jsonl"))
    resword = load_jsonl_by_name(os.path.join(TESTSET,
                                              "results_words_seq.jsonl"))
    resreal = load_real()

    for run in RUNS:
        curve = parse_loss_curve(os.path.join(HERE, run["log"]))
        final_step = curve[-1][0] if curve else 0
        with mlflow.start_run(run_name=run["name"]):
            mlflow.set_tag("notes", run["notes"])
            mlflow.log_params({"data": run["data"], "epochs": run["epochs"],
                               "batch": run["batch"], "lr": run["lr"],
                               "base": "uer/chinese_roberta_L-4_H-256"})
            for step, loss in curve:
                mlflow.log_metric("train_loss", loss, step=step)
            val = parse_val_loss(os.path.join(HERE, run["log"]))
            if val is not None:
                mlflow.log_metric("val_loss", val, step=final_step)
            for ckpt in run["ckpts"]:
                step = ckpt_step(ckpt, final_step)
                if ckpt in res5000:
                    r = res5000[ckpt]["overall"]
                    for key in ("top1", "top3", "mrr"):
                        mlflow.log_metric("test5000_%s" % key, r[key],
                                          step=step)
                if ckpt in resword:
                    r = resword[ckpt]["overall"]
                    for key in ("top1", "top3", "mrr"):
                        mlflow.log_metric("word_%s" % key, r[key], step=step)
                for schema in SCHEMAS:
                    r = resreal.get((schema, ckpt))
                    if not r:
                        continue
                    mlflow.log_metric("real_%s_lm_top1" % schema,
                                      r["lm"]["top1"], step=step)
                    mlflow.log_metric("real_%s_ime_top1" % schema,
                                      r["ime"]["top1"], step=step)
    print("logged runs:", [r["name"] for r in RUNS], flush=True)


if __name__ == "__main__":
    main()
