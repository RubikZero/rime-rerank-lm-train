"""Random-sampled visual comparison: base MLM vs fine-tuned MLM (+prior).

Header shows model names and group counts; each case lists candidates with
scores from both models side by side. Output file name identifies the model.
"""

import argparse
import io
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, is_han, load_candidates  # noqa

TESTSET = os.path.join(HERE, "testset")
K = 20
BETA = 0.25


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
def score_cases(cases, model_path, char2id, ctx_len, key, batch=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()
    for i in range(0, len(cases), batch):
        batch_cases = cases[i:i + batch]
        b = len(batch_cases)
        ctx = torch.zeros(b, ctx_len, dtype=torch.long)
        for j, c in enumerate(batch_cases):
            vals = [char2id[x] for x in c["context"] if x in char2id][-ctx_len:]
            ctx[j, ctx_len - len(vals):] = torch.tensor(vals)
        ctx = ctx.to(device)
        one = torch.ones(b, 1, dtype=torch.long, device=device)
        ids = torch.cat([one * tok.cls_token_id, ctx,
                         one * tok.mask_token_id, one * tok.sep_token_id],
                        dim=1)
        att = torch.cat([one, (ctx != 0).long(), one, one], dim=1)
        logits = model(input_ids=ids, attention_mask=att).logits
        lp = torch.log_softmax(logits[:, -2].float(), dim=-1).cpu().numpy()
        for j, c in enumerate(batch_cases):
            c[key] = np.array([float(lp[j, char2id[x["ch"]]])
                               for x in c["items"]])
    del model
    return cases


def classify(c, key):
    chs = [x["ch"] for x in c["items"]]
    pick = int(np.argsort(-c[key])[0])
    if chs[pick] != c["target"]:
        return "mistake"
    return "kept" if pick == 0 else "corrected"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "lm-train",
                                                   "checkpoints", "final"))
    ap.add_argument("--ft", default=os.path.join(ROOT, "lm-train",
                                                 "reranker_v2", "mlm_ft_v2"))
    ap.add_argument("--requests", default=os.path.join(TESTSET,
                                                       "requests_v2.tsv"))
    ap.add_argument("--meta", default=os.path.join(TESTSET,
                                                   "requests_v2_meta.tsv"))
    ap.add_argument("--candidates", default=os.path.join(
        TESTSET, "candidates_v2_luna_pinyin_simp.tsv"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(
        r"D:\Workspace\rime\doc", "reranker-examples-ftv2-compare.md"))
    args = ap.parse_args()

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
        by_ch = {}
        for rank, (text, q, ctype) in enumerate(cands[rid]):
            if len(text) != 1 or not is_han(text) or text not in char2id:
                continue
            prev = by_ch.get(text)
            if prev is None or q > prev[0]:
                by_ch[text] = {"ch": text, "q": q, "rank": rank}
        if tgt not in by_ch:
            continue
        items = sorted(by_ch.values(), key=lambda x: -x["q"])
        if len(items) > K:
            head = items[:K]
            if not any(x["ch"] == tgt for x in head):
                head.append(by_ch[tgt])
            items = head
        items.sort(key=lambda x: -x["q"])
        cases.append({"rid": rid, "domain": domain, "context": context,
                      "target": tgt, "items": items})
    print("char cases:", len(cases), flush=True)

    score_cases(cases, args.base, char2id, args.ctx, "lm_base")
    score_cases(cases, args.ft, char2id, args.ctx, "lm_ft")

    for c in cases:
        lq = np.array([np.log1p(max(x["q"], 0.0)) for x in c["items"]])
        zp = zs(lq)
        c["fused_base"] = zs(c["lm_base"]) + BETA * zp
        c["fused_ft"] = zs(c["lm_ft"]) + BETA * zp
        c["pick_base"] = int(np.argsort(-c["fused_base"])[0])
        c["pick_ft"] = int(np.argsort(-c["fused_ft"])[0])

    counts = {"base": {"corrected": 0, "kept": 0, "mistake": 0},
              "ft": {"corrected": 0, "kept": 0, "mistake": 0}}
    groups = {"corrected": [], "kept": [], "mistake": []}
    for c in cases:
        tag_b = classify(c, "fused_base")
        tag_f = classify(c, "fused_ft")
        counts["base"][tag_b] += 1
        counts["ft"][tag_f] += 1
        groups[tag_f].append(c)

    def top1(key):
        return sum(1 for c in cases
                   if c["items"][int(np.argsort(-c[key])[0])]["ch"]
                   == c["target"]) / len(cases)

    lines = ["# 重排可视化：base vs 微调（随机抽样，长上下文测试集）\n",
             "- base = `lm-train/checkpoints/final`（v1 MLM）；"
             "FT = `lm-train/reranker_v2/mlm_ft_v2`"
             "（候选集 constrained softmax 微调 3 epochs）\n",
             "- 融合：`z(LM) + 0.25·z(IME 先验)`；案例随机抽样（seed=%d）\n"
             % args.seed,
             "- 总体 top1：base %.3f ｜ FT **%.3f**（测试集 %d 例）\n"
             % (top1("fused_base"), top1("fused_ft"), len(cases)),
             "- 分组计数（base → FT）：修正 %d→%d ｜ 保持 %d→%d ｜ 改错 %d→%d\n"
             % (counts["base"]["corrected"], counts["ft"]["corrected"],
                counts["base"]["kept"], counts["ft"]["kept"],
                counts["base"]["mistake"], counts["ft"]["mistake"]),
             "- 标记：✓=正确；☆=base 选择；★=FT 选择\n", "\n---\n"]
    rng = random.Random(args.seed)
    for tag, title in (("corrected", "修正成功（FT 推翻 IME 首选）"),
                       ("kept", "保持正确"),
                       ("mistake", "改错样例")):
        picks = rng.sample(groups[tag], min(args.per_group, len(groups[tag])))
        picks.sort(key=lambda c: c["domain"])
        lines.append("\n## %s（随机抽取 %d 例，共 %d 例）\n"
                     % (title, len(picks), len(groups[tag])))
        for n, c in enumerate(picks, 1):
            chs = [x["ch"] for x in c["items"]]
            lq = np.array([np.log1p(max(x["q"], 0.0)) for x in c["items"]])
            zl_b, zl_f, zp = zs(c["lm_base"]), zs(c["lm_ft"]), zs(lq)
            lines.append("\n### %s-%d（域：%s，上下文 %d 字）\n"
                         % (title, n, c["domain"], len(c["context"])))
            lines.append("- 模型输入：`%s` ｜ 待定位置：▢\n"
                         % c["context"].replace("\n", " "))
            lines.append("- 正确字：**%s** ｜ IME 首选：`%s` ｜ base 选择：`%s`"
                         " ｜ FT 选择：**`%s`**\n"
                         % (c["target"], chs[0], chs[c["pick_base"]],
                            chs[c["pick_ft"]]))
            lines.append("\n| 候选 | IME名次 | quality | LMbase(z) | LMft(z) | 先验(z) | 融合base | 融合ft | 标记 |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            keep = set(range(min(args.top, len(chs)))) | {c["pick_base"],
                                                          c["pick_ft"]}
            for k in sorted(keep, key=lambda k: c["fused_ft"][k], reverse=True):
                marks = "✓" if chs[k] == c["target"] else ""
                if k == c["pick_base"]:
                    marks += "☆"
                if k == c["pick_ft"]:
                    marks += "★"
                lines.append("| %s | %d | %.3g | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %s |\n"
                             % (chs[k], c["items"][k]["rank"] + 1,
                                c["items"][k]["q"], zl_b[k],
                                zl_f[k], zp[k], c["fused_base"][k],
                                c["fused_ft"][k], marks))
    with io.open(args.out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("written:", args.out, flush=True)


if __name__ == "__main__":
    main()
