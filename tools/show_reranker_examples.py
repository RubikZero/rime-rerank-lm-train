"""Render visual examples of reranker decisions on the real char test set."""

import argparse
import io
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lm-train"))
from eval_reranker import (CTX, K, VOCAB, is_han, load_candidates,  # noqa
                           load_char2id, load_requests_meta)
from train_reranker import Reranker, build_sequences, masked_z  # noqa

TESTSET = os.path.join(HERE, "testset")
OUT = r"D:\Workspace\rime\doc\reranker-examples.md"
PRIOR_BETA = 0.25


def zs(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s


def build_display_cases(char2id, type_vocab):
    target, meta = load_requests_meta()
    cands = load_candidates(os.path.join(TESTSET,
                                         "candidates_luna_pinyin_simp.tsv"))
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
                by_char[text] = (q, rank, ctype)
        if tgt not in by_char:
            continue
        items = [{"ch": ch, "q": q, "rank": r, "type": t}
                 for ch, (q, r, t) in by_char.items()]
        items.sort(key=lambda x: x["rank"])
        ctx = context[-CTX:] if len(context) > CTX else context
        cases.append({"rid": rid, "domain": domain, "context": ctx,
                      "target": tgt, "items": items})
    return cases


def pack(char2id, cases):
    ctx = np.zeros((len(cases), CTX), dtype=np.int64)
    cand = np.zeros((len(cases), K + 1), dtype=np.int64)
    logq = np.zeros((len(cases), K + 1), dtype=np.float32)
    rankn = np.zeros((len(cases), K + 1), dtype=np.float32)
    types = np.zeros((len(cases), K + 1), dtype=np.int64)
    vals = np.zeros((len(cases), K + 1), dtype=np.int8)
    target_pos = np.zeros(len(cases), dtype=np.int64)
    for i, c in enumerate(cases):
        ids = [char2id[ch] for ch in c["context"] if ch in char2id]
        ctx[i, CTX - len(ids):] = ids
        chosen = sorted(c["items"], key=lambda x: -x["q"])[:K]
        ti = next(x for x in c["items"] if x["ch"] == c["target"])
        if ti not in chosen:
            chosen.append(ti)
        # display keeps IME order but scores follow the (quality-sorted) pack
        c["packed"] = chosen
        target_pos[i] = chosen.index(ti)
        n = len(chosen)
        for k, x in enumerate(chosen):
            cand[i, k] = char2id[x["ch"]]
            logq[i, k] = np.log1p(max(x["q"], 0.0))
            rankn[i, k] = x["rank"] / max(1, len(c["items"]) - 1)
            types[i, k] = 0
        vals[i, :n] = 1
    return ctx, cand, logq, rankn, types, vals, target_pos


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--mlm", default=os.path.join(ROOT, "lm-train",
                                                  "checkpoints", "final"))
    ap.add_argument("--reranker", default=os.path.join(
        ROOT, "lm-train", "reranker", "model6", "model_best.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    char2id = load_char2id(VOCAB)
    cases = build_display_cases(char2id, {})
    print("cases:", len(cases), flush=True)
    ctx, cand, logq, rankn, types, mask, tpos = pack(char2id, cases)
    t_ctx = torch.tensor(ctx)
    t_cand = torch.tensor(cand)
    t_logq = torch.tensor(logq)
    t_rankn = torch.tensor(rankn)
    t_types = torch.tensor(types)
    t_mask = torch.tensor(mask, dtype=torch.float32)
    t_tgt = torch.tensor(tpos)

    # LM scores
    mtok = AutoTokenizer.from_pretrained(args.mlm)
    mmod = AutoModelForMaskedLM.from_pretrained(args.mlm).to(device)
    mmod.eval()
    lm = np.zeros_like(logq)
    B = 64
    for i in range(0, len(cases), B):
        cb = t_ctx[i:i + B].to(device)
        kb = t_cand[i:i + B].to(device)
        b = cb.shape[0]
        one = torch.ones(b, 1, dtype=torch.long, device=device)
        ids = torch.cat([one * mtok.cls_token_id, cb,
                         one * mtok.mask_token_id, one * mtok.sep_token_id],
                        dim=1)
        att = torch.cat([one, (cb != 0).long(), one, one], dim=1)
        logits = mmod(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, -2].float(), dim=-1)
        rows = torch.arange(b, device=device).unsqueeze(1).expand(b, kb.shape[1])
        lm[i:i + b] = lp[rows, kb].cpu().numpy()
    del mmod

    # reranker scores
    ckpt = torch.load(args.reranker, map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(ckpt["base"])
    model = Reranker(ckpt["base"], len(ckpt["type_vocab"]),
                     feat_dim=1 + (1 if ckpt.get("concat_score") else 0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    rr = np.zeros_like(logq)
    for i in range(0, len(cases), B):
        cb = t_ctx[i:i + B].to(device)
        kb = t_cand[i:i + B].to(device)
        mb = t_mask[i:i + B].to(device)
        zz = masked_z(t_logq[i:i + B].to(device), mb)
        ids, att = build_sequences(cb, kb, tok.cls_token_id,
                                   tok.sep_token_id)
        o = model(ids, att, t_types[i:i + B].reshape(-1).to(device),
                  t_rankn[i:i + B].reshape(-1).to(device),
                  zz.reshape(-1) if ckpt.get("concat_score") else None)
        rr[i:i + B] = o.reshape(cb.shape[0], -1).cpu().numpy()

    lines = []
    lines.append("# 重排器可视化样例（拼音单字，真实候选）\n")
    lines.append("- 方案：**z(MLM 掩码得分) + 0.25·z(IME 先验)**（真实测试集最优配置，"
                 "top1 0.5234 vs IME 0.4542）\n")
    lines.append("- 列含义：IME名次=原菜单序；quality=librime 候选分；"
                 "LM(z)=掩码伪似然（case 内 z）；先验(z)=log quality（z）；"
                 "融合分=最终排序分；★=方案选择；✓=正确答案\n")
    lines.append("\n---\n")

    groups = {"corrected": [], "kept": [], "mistake": []}
    for i, c in enumerate(cases):
        m = t_mask[i].numpy() > 0
        fused = (zs(lm[i][m]) + PRIOR_BETA * zs(logq[i][m]))
        order = np.argsort(-fused)
        pick_idx = int(order[0])
        prior_idx = int(np.argmax(logq[i][m]))
        correct = pick_idx == int(tpos[i])
        if correct and prior_idx != pick_idx:
            groups["corrected"].append((i, m, fused, pick_idx))
        elif correct and prior_idx == pick_idx:
            groups["kept"].append((i, m, fused, pick_idx))
        else:
            groups["mistake"].append((i, m, fused, pick_idx))
    print({k: len(v) for k, v in groups.items()}, flush=True)

    for tag, title, note in (
            ("corrected", "修正成功", "模型推翻 IME 首选并选中正确字"),
            ("kept", "保持正确", "IME 首选已正确，模型保持不变"),
            ("mistake", "改错样例", "模型推翻后选错（诚实展示）")):
        picked = groups[tag][:args.per_group]
        lines.append("\n## %s（%s）\n" % (title, note))
        for n, (i, m, fused, pick_idx) in enumerate(picked, 1):
            c = cases[i]
            items = [x for k, x in enumerate(c["packed"]) if m[k]]
            order = np.argsort(-fused)
            rank_of = {int(k): r + 1 for r, k in enumerate(order)}
            lines.append("\n### %s-%d（域：%s）\n" % (title, n, c["domain"]))
            ctx_disp = c["context"].replace("\n", " ")
            lines.append("- 上文：`…%s` ｜ 待定位置：▢\n" % ctx_disp[-24:])
            pick = items[pick_idx]["ch"]
            target = c["target"]
            prior_ch = items[int(np.argmax(logq[i][m]))]["ch"]
            lines.append("- 正确字：**%s** ｜ IME 首选：`%s` ｜ 方案选择：**%s**\n"
                         % (target, prior_ch, pick))
            lines.append("\n| 候选 | IME名次 | quality | LM(z) | 先验(z) | 融合分 | 标记 |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---|\n")
            show = sorted(range(len(items)),
                          key=lambda k: min(items[k]["rank"],
                                            rank_of.get(k, 99)))[:args.top]
            zl, zp = zs(lm[i][m]), zs(logq[i][m])
            for k in sorted(show, key=lambda k: fused[k], reverse=True):
                x = items[k]
                marks = []
                if x["ch"] == target:
                    marks.append("✓")
                if k == pick_idx:
                    marks.append("★")
                lines.append("| %s | %d | %.3g | %+.2f | %+.2f | %+.2f | %s |\n"
                             % (x["ch"], x["rank"] + 1, x["q"], zl[k], zp[k],
                                fused[k], " ".join(marks)))

    with io.open(OUT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("written:", OUT, flush=True)
    # console preview
    for line in lines[:60]:
        print(line, end="")


if __name__ == "__main__":
    main()
