"""Word-level evaluation: seq pseudo-likelihood scoring of word candidates.

P(cand | ctx) = prod_k P(c_k | ctx, c_1..c_{k-1}) via one masked forward per
position: input = [CLS] ctx c_1..c_{k-1} [MASK] [SEP].
"""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, is_han, load_candidates, load_requests_meta  # noqa
from lm_only_eval import build_cases  # noqa

TESTSET = os.path.join(HERE, "testset")
K = 20


def load_char2id(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\n")
            if len(tok) == 1:
                d.setdefault(tok, idx)
    return d


def build_word_cases(char2id, req_path, meta_path, cand_path, ctx_len,
                     kind="word"):
    target, meta = load_requests_meta(req_path, meta_path)
    cands = load_candidates(cand_path)
    cases = []
    for rid, tgt in target.items():
        if rid not in meta or rid not in cands:
            continue
        ckind, domain, context = meta[rid]
        if ckind != kind or len(tgt) < 1:
            continue
        if not all(ch in char2id for ch in tgt):
            continue
        by_text = {}
        for rank, (text, q, ctype) in enumerate(cands[rid]):
            if len(text) != len(tgt) or not all(ch in char2id for ch in text):
                continue
            prev = by_text.get(text)
            if prev is None or q > prev[0]:
                by_text[text] = {"text": text, "q": q, "rank": rank}
        if tgt not in by_text:
            continue
        items = sorted(by_text.values(), key=lambda x: -x["q"])
        if len(items) > K:
            head = items[:K]
            if not any(x["text"] == tgt for x in head):
                head.append(by_text[tgt])
            items = head
        ctx_ids = np.zeros(ctx_len, dtype=np.int64)
        ctx = context[-ctx_len:] if len(context) > ctx_len else context
        vals = [char2id[c] for c in ctx if c in char2id]
        ctx_ids[ctx_len - len(vals):] = vals
        cases.append({"rid": rid, "domain": domain, "target": tgt,
                      "items": items, "ctx": ctx_ids})
    return cases


@torch.inference_mode()
def score_word_cases(cases, model_path, char2id, ctx_len, batch=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()
    for i in range(0, len(cases), batch):
        group = cases[i:i + batch]
        cb = torch.tensor(np.stack([c["ctx"] for c in group])).to(device)
        b = cb.shape[0]
        maxlen = max(len(x["text"]) for c in group for x in c["items"])
        sums = [[0.0] * len(c["items"]) for c in group]
        for k in range(maxlen):
            rows = []
            for j, c in enumerate(group):
                for m, x in enumerate(c["items"]):
                    if len(x["text"]) > k:
                        rows.append((j, m, x["text"]))
            if not rows:
                continue
            seqs = []
            for j, m, text in rows:
                pre = [char2id[ch] for ch in text[:k]]
                seqs.append((j, m, pre))
            maxpre = max(len(p) for _, _, p in seqs)
            L = 1 + ctx_len + maxpre + 1 + 1
            ids = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
            att = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
            pos = []
            for r, (j, m, pre) in enumerate(seqs):
                ids[r, 0] = tok.cls_token_id
                ids[r, 1:1 + ctx_len] = cb[j]
                att[r, 1:1 + ctx_len] = (cb[j] != 0).long()
                off = 1 + ctx_len
                for p, v in enumerate(pre):
                    ids[r, off + p] = v
                    att[r, off + p] = 1
                mp = off + len(pre)
                ids[r, mp] = tok.mask_token_id
                att[r, mp] = 1
                ids[r, mp + 1] = tok.sep_token_id
                att[r, mp + 1] = 1
                pos.append(mp)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits = model(input_ids=ids, attention_mask=att).logits
            pidx = torch.tensor(pos, device=device)
            sel = logits[torch.arange(len(seqs), device=device), pidx]
            lp = torch.log_softmax(sel.float(), dim=-1)
            for r, (j, m, pre) in enumerate(seqs):
                tgt_id = char2id[group[j]["items"][m]["text"][k]]
                sums[j][m] += float(lp[r, tgt_id])
        for j, c in enumerate(group):
            c["lm"] = np.array(sums[j])
    del model
    return cases


def masked_z(x, mask):
    x = np.asarray(x, dtype=np.float64)
    m = mask.astype(bool)
    mu = x[m].mean()
    sd = x[m].std()
    if sd < 1e-9:
        return np.zeros_like(x)
    out = (x - mu) / sd
    return np.where(m, out, 0.0)


def evaluate(cases, betas):
    n = len(cases)
    prior_top1 = np.mean([c["items"][0]["text"] == c["target"]
                          for c in cases])
    print("word cases=%d  prior(IME) top1=%.4f" % (n, prior_top1), flush=True)
    for c in cases:
        c["logq"] = np.array([np.log1p(max(x["q"], 0.0)) for x in c["items"]])
        c["mask"] = np.ones(len(c["items"]), dtype=np.float64)
        c["zlm"] = masked_z(c["lm"], c["mask"])
        c["zpr"] = masked_z(c["logq"], c["mask"])
    for beta in betas:
        hits1 = hits3 = mrr = over_all = ok_over = 0
        for i, c in enumerate(cases):
            s = np.where(c["mask"] > 0, c["zlm"] + beta * c["zpr"], -1e9)
            order = np.argsort(-s)
            texts = [x["text"] for x in c["items"]]
            ti = texts.index(c["target"])
            r = int((order == ti).nonzero()[0][0]) + 1
            hits1 += r == 1
            hits3 += r <= 3
            mrr += 1.0 / r
            if order[0] != 0:
                over_all += 1
                if r == 1:
                    ok_over += 1
        print("beta=%.2f top1=%.4f top3=%.4f mrr=%.4f churn=%.3f precision=%.3f"
              % (beta, hits1 / n, hits3 / n, mrr / n, over_all / n,
                 (ok_over / over_all) if over_all else 1.0), flush=True)
    by_len = {}
    for i, c in enumerate(cases):
        L = len(c["target"])
        ti = [x["text"] for x in c["items"]].index(c["target"])
        r = int((np.argsort(-c["zlm"]) == ti).nonzero()[0][0]) + 1
        d = by_len.setdefault(L, [0, 0])
        d[0] += r == 1
        d[1] += 1
    print("  MLM-only top1 by target len: %s" % "  ".join(
        "%dchar=%.3f(n=%d)" % (L, v[0] / v[1], v[1])
        for L, v in sorted(by_len.items())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm", required=True)
    ap.add_argument("--requests", default=os.path.join(TESTSET,
                                                       "requests_v2.tsv"))
    ap.add_argument("--meta", default=os.path.join(TESTSET,
                                                   "requests_v2_meta.tsv"))
    ap.add_argument("--candidates", default=os.path.join(
        TESTSET, "candidates_v2_luna_pinyin_simp.tsv"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--kind", default="word")
    ap.add_argument("--betas", default="0,0.25,0.5,1")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    cases = build_word_cases(char2id, args.requests, args.meta,
                             args.candidates, args.ctx, args.kind)
    print("word cases:", len(cases), flush=True)
    cases = score_word_cases(cases, args.mlm, char2id, args.ctx, args.batch)
    evaluate(cases, [float(x) for x in args.betas.split(",")])


if __name__ == "__main__":
    main()
