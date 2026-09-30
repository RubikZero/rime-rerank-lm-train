"""Render visual examples for the long-context test set (MLM + prior blend).

Shows exactly what the model sees: the full 256-char context and candidate
scores. Output: D:\\Workspace\\rime\\doc\\reranker-examples-v2.md
"""

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
from eval_reranker import VOCAB, is_han, load_candidates  # noqa

TESTSET = os.path.join(HERE, "testset")
OUT = r"D:\Workspace\rime\doc\reranker-examples-v2.md"
K = 20
PRIOR_BETA = 0.25


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
                meta[p[0]] = (p[1], p[2], p[3])
    return target, meta


def zs(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm", default=os.path.join(ROOT, "lm-train",
                                                  "checkpoints", "final"))
    ap.add_argument("--requests", default=os.path.join(TESTSET,
                                                       "requests_v2.tsv"))
    ap.add_argument("--meta", default=os.path.join(TESTSET,
                                                   "requests_v2_meta.tsv"))
    ap.add_argument("--candidates", default=os.path.join(
        TESTSET, "candidates_v2_luna_pinyin_simp.tsv"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    char2id = load_char2id(VOCAB)
    target, meta = load_requests_meta(args.requests, args.meta)
    cands = load_candidates(args.candidates)
    cases = []
    for rid, tgt in target.items():
        if rid not in meta or rid not in cands:
            continue
        kind, domain, context = meta[rid]
        if kind != "char" or len(tgt) != 1 or tgt not in char2id:
            continue
        items = []
        for rank, (text, q, ctype) in enumerate(cands[rid]):
            if len(text) != 1 or not is_han(text) or text not in char2id:
                continue
            items.append({"ch": text, "q": q, "rank": rank})
        by_ch = {}
        for x in items:
            prev = by_ch.get(x["ch"])
            if prev is None or x["q"] > prev["q"]:
                by_ch[x["ch"]] = x
        items = sorted(by_ch.values(), key=lambda x: x["rank"])
        if not any(x["ch"] == tgt for x in items):
            continue
        cases.append({"rid": rid, "domain": domain, "context": context,
                      "target": tgt, "items": items})
    print("char cases:", len(cases), flush=True)

    tok = AutoTokenizer.from_pretrained(args.mlm)
    model = AutoModelForMaskedLM.from_pretrained(args.mlm).to(device)
    model.eval()
    results = []
    B = 32
    for i in range(0, len(cases), B):
        batch = cases[i:i + B]
        b = len(batch)
        ctx_ids = torch.zeros(b, args.ctx, dtype=torch.long)
        for j, c in enumerate(batch):
            vals = [char2id[x] for x in c["context"] if x in char2id][-args.ctx:]
            ctx_ids[j, args.ctx - len(vals):] = torch.tensor(vals)
        ctx_ids = ctx_ids.to(device)
        one = torch.ones(b, 1, dtype=torch.long, device=device)
        ids = torch.cat([one * tok.cls_token_id, ctx_ids,
                         one * tok.mask_token_id, one * tok.sep_token_id],
                        dim=1)
        att = torch.cat([one, (ctx_ids != 0).long(), one, one], dim=1)
        logits = model(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, -2].float(), dim=-1).cpu().numpy()
        for j, c in enumerate(batch):
            c["lm"] = {x["ch"]: float(lp[j, char2id[x["ch"]]])
                       for x in c["items"]}
        results.extend(batch)
    cases = results

    groups = {"corrected": [], "kept": [], "mistake": []}
    for c in cases:
        chs = [x["ch"] for x in c["items"]]
        lm = np.array([c["lm"][ch] for ch in chs])
        lq = np.array([np.log1p(max(x["q"], 0.0)) for x in c["items"]])
        fused = zs(lm) + PRIOR_BETA * zs(lq)
        order = np.argsort(-fused)
        pick = int(order[0])
        prior = 0  # items sorted by menu rank; rank 0 is IME's first
        correct = chs[pick] == c["target"]
        if correct and pick != prior:
            groups["corrected"].append((c, fused, pick))
        elif correct and pick == prior:
            groups["kept"].append((c, fused, pick))
        else:
            groups["mistake"].append((c, fused, pick))
    print({k: len(v) for k, v in groups.items()}, flush=True)

    lines = ["# 重排可视化（长上下文测试集 v2）\n",
             "- 方案：**z(MLM 掩码得分) + 0.25·z(IME 先验)**\n",
             "- 上下文：模型实际看到的完整上下文（最多 %d 字，左 PAD 略）\n"
             % args.ctx,
             "- 列：quality=librime 候选分；LM(z)/先验(z)/融合分=case 内 z 分数；"
             "★=选中；✓=正确\n", "\n---\n"]
    for tag, title in (("corrected", "修正成功（推翻 IME 首选）"),
                       ("kept", "保持正确（IME 首选已正确）"),
                       ("mistake", "改错样例")):
        lines.append("\n## %s\n" % title)
        for n, (c, fused, pick) in enumerate(
                groups[tag][:args.per_group], 1):
            chs = [x["ch"] for x in c["items"]]
            lm = np.array([c["lm"][ch] for ch in chs])
            lq = np.array([np.log1p(max(x["q"], 0.0)) for x in c["items"]])
            zlm, zpr = zs(lm), zs(lq)
            lines.append("\n### %s-%d（域：%s，上下文 %d 字）\n" %
                         (title, n, c["domain"], len(c["context"])))
            lines.append("- 模型输入：`%s` ｜ 待定位置：▢\n" %
                         c["context"].replace("\n", " "))
            lines.append("- 正确字：**%s** ｜ IME 首选：`%s` ｜ 选择：**%s**\n"
                         % (c["target"], chs[0], chs[pick]))
            lines.append("\n| 候选 | IME名次 | quality | LM(z) | 先验(z) | 融合分 | 标记 |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---|\n")
            order = np.argsort(-fused)
            rank_of = {int(k): r + 1 for r, k in enumerate(order)}
            show = sorted(range(len(chs)),
                          key=lambda k: min(k, rank_of.get(k, 99)))[:args.top]
            for k in sorted(show, key=lambda k: fused[k], reverse=True):
                marks = ("✓" if chs[k] == c["target"] else "")
                if k == pick:
                    marks += "★"
                lines.append("| %s | %d | %.3g | %+.2f | %+.2f | %+.2f | %s |\n"
                             % (chs[k], k + 1, c["items"][k]["q"], zlm[k],
                                zpr[k], fused[k], marks))

    with io.open(args.out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("written:", args.out, flush=True)


if __name__ == "__main__":
    main()
