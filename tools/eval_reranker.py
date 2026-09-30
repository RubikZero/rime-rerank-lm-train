"""Evaluate the trained reranker on the real-candidate char test set."""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lm-train"))
from train_reranker import Reranker, build_sequences, masked_z  # noqa

TESTSET = os.path.join(HERE, "testset")
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
CTX = 64
K = 20


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
        (0xF900 <= o <= 0xFAFF)


def load_char2id(path):
    d = {}
    with io.open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\n")
            if len(tok) == 1:
                d.setdefault(tok, idx)
    return d


def load_requests_meta(req_path, meta_path):
    target = {}
    with io.open(req_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                target[p[0]] = p[2]
    meta = {}
    with io.open(meta_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                meta[p[0]] = (p[1], p[2], p[3])  # kind, domain, context
    return target, meta


def load_candidates(path):
    out = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t", 2)
            cands = []
            if len(p) > 2 and p[2]:
                for c in p[2].split("\x1f"):
                    f4 = c.split("\x1e")
                    if len(f4) < 4:
                        continue
                    try:
                        q = float(f4[2])
                    except ValueError:
                        q = 0.0
                    cands.append((f4[0], q, f4[3]))
            out[p[0]] = cands
    return out


def build_cases(char2id, type_vocab, req_path, meta_path, cand_path,
                ctx_len):
    target, meta = load_requests_meta(req_path, meta_path)
    cands = load_candidates(cand_path)
    cases = []
    for rid, tgt in target.items():
        if rid not in meta or rid not in cands:
            continue
        kind, domain, context = meta[rid]
        if kind != "char" or len(tgt) != 1 or tgt not in char2id:
            continue
        by_char = {}
        for rank, (text, q, ctype) in enumerate(cands[rid]):
            if len(text) != 1 or not is_han(text) or text not in char2id:
                continue
            prev = by_char.get(text)
            if prev is None or q > prev[0]:
                by_char[text] = (q, rank, type_vocab.get(ctype, 0))
        if tgt not in by_char:
            continue
        items = [(ch, q, r, t) for ch, (q, r, t) in by_char.items()]
        items.sort(key=lambda x: -x[1])
        if len(items) > K:
            head = items[:K]
            ti = next(x for x in items if x[0] == tgt)
            if ti not in head:
                head.append(ti)
            items = head
        ctx_ids = np.zeros(ctx_len, dtype=np.int64)
        ctx = context[-ctx_len:] if len(context) > ctx_len else context
        vals = [char2id[c] for c in ctx if c in char2id]
        ctx_ids[ctx_len - len(vals):] = vals
        n = len(items)
        cand = np.zeros(K + 1, dtype=np.int64)
        logq = np.zeros(K + 1, dtype=np.float32)
        rankn = np.zeros(K + 1, dtype=np.float32)
        types = np.zeros(K + 1, dtype=np.int64)
        for k, (ch, q, r, t) in enumerate(items):
            cand[k] = char2id[ch]
            logq[k] = np.log1p(max(q, 0.0))
            rankn[k] = r / max(1, len(cands[rid]) - 1)
            types[k] = t
        cases.append({"rid": rid, "domain": domain,
                      "target_pos": next(i for i, x in enumerate(items)
                                         if x[0] == tgt),
                      "ctx": ctx_ids, "cand": cand, "logq": logq,
                      "rankn": rankn, "types": types, "n": n})
    return cases


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        ROOT, "lm-train", "reranker_v2", "model", "model_best.pt"))
    ap.add_argument("--requests", default=os.path.join(
        TESTSET, "requests_v2.tsv"))
    ap.add_argument("--meta", default=os.path.join(
        TESTSET, "requests_v2_meta.tsv"))
    ap.add_argument("--candidates", default=os.path.join(
        TESTSET, "candidates_v2_luna_pinyin_simp.tsv"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--alphas", default="1,2,4,8")
    ap.add_argument("--taus", default="0,0.5,1,2,4")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mlm", default="")
    ap.add_argument("--mlm-weight", default="0,0.5,1,2")
    ap.add_argument("--prior-weight", default="4")
    ap.add_argument("--rr-weights", default="1")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    type_vocab = ckpt["type_vocab"]
    base = ckpt["base"]
    tok = AutoTokenizer.from_pretrained(base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Reranker(base, len(type_vocab),
                     feat_dim=1 + (1 if ckpt.get("concat_score") else 0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # pad cases to batch
    cases = build_cases(char2id, type_vocab, args.requests, args.meta,
                        args.candidates, args.ctx)
    print("char cases:", len(cases), flush=True)
    ctx = torch.tensor(np.stack([c["ctx"] for c in cases]), device=device)
    cand = torch.tensor(np.stack([c["cand"] for c in cases]), device=device)
    logq = torch.tensor(np.stack([c["logq"] for c in cases]), device=device)
    rankn = torch.tensor(np.stack([c["rankn"] for c in cases]), device=device)
    types = torch.tensor(np.stack([c["types"] for c in cases]), device=device)
    tgt = torch.tensor([c["target_pos"] for c in cases], device=device)
    ns = torch.tensor([c["n"] for c in cases], device=device)
    mask = (torch.arange(cand.shape[1], device=device)[None, :] <
            ns[:, None]).float()

    out = []
    for i in range(0, len(cases), args.batch):
        ctx_b = ctx[i:i + args.batch]
        cand_b = cand[i:i + args.batch]
        mask_b = mask[i:i + args.batch]
        zz = masked_z(logq[i:i + args.batch], mask_b)
        ids, att = build_sequences(ctx_b, cand_b, tok.cls_token_id,
                                   tok.sep_token_id)
        o = model(ids, att, types[i:i + args.batch].reshape(-1),
                  rankn[i:i + args.batch].reshape(-1),
                  zz.reshape(-1) if ckpt.get("concat_score") else None)
        out.append(o.reshape(ctx_b.shape[0], -1))
    out = torch.cat(out)
    zz = masked_z(logq, mask)
    prior = (tgt == 0).float().mean().item()
    print("IME(prior) top1=%.4f" % prior, flush=True)

    lm_scores = None
    if args.mlm:
        from transformers import AutoModelForMaskedLM
        mtok = AutoTokenizer.from_pretrained(args.mlm)
        mmod = AutoModelForMaskedLM.from_pretrained(args.mlm).to(device)
        mmod.eval()
        lm_scores = torch.zeros_like(logq)
        with torch.inference_mode():
            for i in range(0, len(cases), args.batch):
                ctx_b = ctx[i:i + args.batch]
                cand_b = cand[i:i + args.batch]
                mask_b = mask[i:i + args.batch]
                B = ctx_b.shape[0]
                one = torch.ones(B, 1, dtype=torch.long, device=device)
                ids = torch.cat([one * mtok.cls_token_id, ctx_b,
                                 one * mtok.mask_token_id,
                                 one * mtok.sep_token_id], dim=1)
                att = torch.cat([one, (ctx_b != 0).long(), one, one], dim=1)
                with torch.autocast(device_type="cuda",
                                    dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                    logits = mmod(input_ids=ids, attention_mask=att).logits
                lp = torch.log_softmax(logits[:, -2].float(), dim=-1)
                rows = torch.arange(B, device=device).unsqueeze(1).expand(
                    B, cand_b.shape[1])
                lm_scores[i:i + B] = lp[rows, cand_b]
        del mmod
        torch.cuda.empty_cache()

    for alpha in [float(x) for x in args.alphas.split(",")]:
        if lm_scores is not None:
            zlm = masked_z(lm_scores, mask)
            combos = [(float(w), alpha, float(g))
                      for w in args.mlm_weight.split(",")
                      for g in args.rr_weights.split(",")]
        else:
            combos = [(0.0, alpha, 1.0)]
        for w, a, g in combos:
            score = (a * zz + out) if lm_scores is None else \
                (a * zz + w * zlm + g * out)
            score = score.masked_fill(mask == 0, -1e9)
            best = score.argmax(dim=1)
            final = best
            ok = (final == tgt).float().mean().item()
            over = int((final != 0).sum())
            over_ok = int(((final != 0) & (final == tgt)).sum())
            print("prior=%.2f mlm=%.2f rr=%.2f top1=%.4f churn=%.3f "
                  "overrides=%d precision=%.3f" %
                  (a, w, g, ok, over / len(cases), over,
                   (over_ok / over) if over else 1.0), flush=True)


if __name__ == "__main__":
    main()
