"""Pure baselines on the real-candidate char test set: MLM only / MLM+prior."""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, build_cases, load_char2id  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lm-train"))
from train_reranker import masked_z  # noqa


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm", default=r"D:\Workspace\rime\third-party\lm-train\checkpoints\final")
    ap.add_argument("--requests", default=os.path.join(HERE, "testset",
                                                       "requests_v2.tsv"))
    ap.add_argument("--meta", default=os.path.join(HERE, "testset",
                                                   "requests_v2_meta.tsv"))
    ap.add_argument("--candidates", default=os.path.join(
        HERE, "testset", "candidates_v2_luna_pinyin_simp.tsv"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--betas", default="0,0.25,0.5,1,2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    char2id = load_char2id(VOCAB)
    cases = build_cases(char2id, {}, args.requests, args.meta,
                        args.candidates, args.ctx)
    print("char cases:", len(cases), flush=True)
    ctx = torch.tensor(np.stack([c["ctx"] for c in cases]))
    cand = torch.tensor(np.stack([c["cand"] for c in cases]))
    logq = torch.tensor(np.stack([c["logq"] for c in cases]))
    tgt = torch.tensor([c["target_pos"] for c in cases])
    ns = torch.tensor([c["n"] for c in cases])
    mask = (torch.arange(cand.shape[1])[None, :] < ns[:, None]).float()
    print("prior top1=%.4f" % (tgt == 0).float().mean().item(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.mlm)
    model = AutoModelForMaskedLM.from_pretrained(args.mlm).to(device)
    model.eval()
    lm = torch.zeros_like(logq)
    for i in range(0, len(cases), args.batch):
        cb = ctx[i:i + args.batch].to(device)
        kb = cand[i:i + args.batch].to(device)
        B = cb.shape[0]
        one = torch.ones(B, 1, dtype=torch.long, device=device)
        ids = torch.cat([one * tok.cls_token_id, cb,
                         one * tok.mask_token_id, one * tok.sep_token_id],
                        dim=1)
        att = torch.cat([one, (cb != 0).long(), one, one], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits = model(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, -2].float(), dim=-1)
        rows = torch.arange(B, device=device).unsqueeze(1).expand(
            B, kb.shape[1])
        lm[i:i + B] = lp[rows, kb].cpu()

    zlm = masked_z(lm, mask)
    zpr = masked_z(logq, mask)
    rank = {}
    for beta in [float(x) for x in args.betas.split(",")]:
        score = (zlm + beta * zpr).masked_fill(mask == 0, -1e9)
        order = score.argsort(dim=1, descending=True)
        r = (order == tgt.unsqueeze(1)).nonzero()[:, 1] + 1
        top1 = (r == 1).float().mean().item()
        top3 = (r <= 3).float().mean().item()
        mrr = (1.0 / r.float()).mean().item()
        over = int((order[:, 0] != 0).sum())
        ok_over = int(((order[:, 0] != 0) & (order[:, 0] == tgt)).sum())
        print("beta=%.2f top1=%.4f top3=%.4f mrr=%.4f churn=%.3f "
              "precision=%.3f" %
              (beta, top1, top3, mrr, over / len(cases),
               (ok_over / over) if over else 1.0), flush=True)


if __name__ == "__main__":
    main()
