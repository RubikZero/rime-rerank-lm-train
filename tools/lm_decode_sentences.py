"""Sentence generation by constrained MLM decoding, compared with IME.

Question: can the MLM act like an LLM, generating one character at a time
(each step appended to the context) under the typed pinyin constraints, and
beat IME's single best whole-sentence guess?

Allowed chars per position = all vocab chars having a reading (heteronym set,
from pypinyin) equal to the requested syllable at that position.
"""

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from pypinyin import Style, lazy_pinyin, pinyin
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, is_han, load_candidates, load_char2id  # noqa

TESTSET = os.path.join(HERE, "testset")
CACHE = os.path.join(TESTSET, "syllable2chars.json")


def clean_syl(s):
    s = s.replace("ü", "v")
    return "".join(c for c in s if "a" <= c <= "z")


def load_readings(path):
    char_w = {}
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            try:
                char_w.setdefault(f[0], {})[f[1]] = float(f[2])
            except ValueError:
                pass
    return char_w


def build_allowed_from_readings(char2id, char_w):
    syl2ids = defaultdict(list)
    for ch, readings in char_w.items():
        cid = char2id.get(ch)
        if cid is None:
            continue
        for syl in readings:
            syl2ids[syl].append(cid)
    return {k: sorted(set(v)) for k, v in syl2ids.items()}


def load_word_table(path, char2id):
    by_seq = defaultdict(set)
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2 and f[0] and all(c in char2id for c in f[0]):
            by_seq[f[1]].add(f[0])
    return dict(by_seq)


def build_syllable_map(char2id, cache):
    if os.path.exists(cache):
        with io.open(cache, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("size") == len(char2id):
            return {k: [char2id[c] for c in v if c in char2id]
                    for k, v in data["map"].items()}
    chars = list(char2id.items())
    m = defaultdict(list)
    for ch, cid in chars:
        try:
            readings = pinyin(ch, style=Style.NORMAL, heteronym=True,
                              errors="ignore")
        except Exception:
            continue
        if not readings or not readings[0]:
            continue
        for r in readings[0]:
            syl = clean_syl(r)
            if syl:
                m[syl].append(cid)
            elif r == "ê":
                m.setdefault("e", []).append(cid)
    dump = {k: sorted({c for c in v})
            for k, v in m.items()}
    rev = {cid: ch for ch, cid in char2id.items()}
    with io.open(cache, "w", encoding="utf-8") as f:
        json.dump({"size": len(char2id),
                   "map": {k: [rev[c] for c in v]
                           for k, v in dump.items()}}, f, ensure_ascii=False)
    return dump


def load_requests(req_path, meta_path):
    keys, target, meta = {}, {}, {}
    for line in io.open(req_path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            keys[p[0]] = p[1]
            target[p[0]] = p[2]
    for line in io.open(meta_path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            meta[p[0]] = (p[1], p[2], p[3])
    return keys, target, meta


@torch.inference_mode()
def decode(cases, model, tok, device, ctx_len, beam=4, topk=8,
           lam_pron=0.0, word_seqs=None, lam_word=0.0):
    import math
    results = []
    for ci, c in enumerate(cases):
        ctx = c["ctx_ids"]
        allowed = c["allowed"]  # list of lists of token ids
        syls = c["syls"]
        beams = [([], 0.0)]  # (generated ids, score)
        for t, al in enumerate(allowed):
            if not al:
                beams = [(g, s) for g, s in beams]
                break
            rows = []
            for g, s in beams:
                pre = [c["char2id"][x] for x in g]
                ids = ([tok.cls_token_id] + ctx + pre +
                       [tok.mask_token_id, tok.sep_token_id])
                att = ([1] + [1 if v != 0 else 0 for v in ctx] +
                       [1] * (len(pre) + 2))
                rows.append((g, s, ids, att))
            maxlen0 = max(len(r[2]) for r in rows)
            padded = torch.zeros(len(rows), maxlen0, dtype=torch.long,
                                 device=device)
            att = torch.zeros(len(rows), maxlen0, dtype=torch.long,
                              device=device)
            pos = []
            for r, (g, s, ids, a) in enumerate(rows):
                padded[r, :len(ids)] = torch.tensor(ids, device=device)
                att[r, :len(a)] = torch.tensor(a, device=device)
                pos.append(len(ids) - 2)
            logits = model(input_ids=padded, attention_mask=att).logits
            pidx = torch.tensor(pos, device=device)
            sel = logits[torch.arange(len(rows), device=device), pidx]
            lp = torch.log_softmax(sel.float(), dim=-1)
            cands = []
            al_t = torch.tensor(al, device=device)
            lw_t = None
            if lam_pron:
                lw_t = torch.tensor(
                    [math.log(max(w, 1e-6)) for w in c["logw"][t]],
                    device=device)
            for r, (g, s, ids, a) in enumerate(rows):
                sub = lp[r, al_t]
                if lw_t is not None:
                    sub = sub + lam_pron * lw_t
                k = min(topk, sub.numel())
                vals, idxs = torch.topk(sub, k)
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    ch = c["rev"][al[i]]
                    score = s + v
                    if word_seqs is not None and lam_word:
                        score += lam_word * match_word(g, ch, t, syls,
                                                       word_seqs)
                    cands.append((g + [ch], score))
            cands.sort(key=lambda x: -x[1])
            beams = cands[:beam]
        best = beams[0][0] if beams else []
        results.append("".join(best[:len(allowed)]))
    return results


def match_word(g, ch, t, syls, word_seqs):
    """1.0 if emitting ch closes a table word whose syllables match."""
    for k in range(min(4, t + 1), 0, -1):
        word = ("".join(g[-(k - 1):]) if k > 1 else "") + ch
        seq = " ".join(syls[t - k + 1:t + 1])
        if word in word_seqs.get(seq, ()):
            return 1.0
    return 0.0


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tok, device


def build_cases(char2id, rev, syl2ids, keys, target, meta, ctx_len,
                char_w=None, min_w=0.0):
    cases = []
    for rid, tgt in target.items():
        if rid not in meta or len(tgt) < 1 or not all(
                ch in char2id for ch in tgt):
            continue
        kind, domain, context = meta[rid]
        ctx = context[-ctx_len:]
        vals = [char2id[c] for c in ctx if c in char2id][-ctx_len:]
        ctx_ids = [0] * (ctx_len - len(vals)) + vals
        syls = [clean_syl(s) for s in
                lazy_pinyin(tgt, style=Style.NORMAL, errors="ignore")]
        if len(syls) != len(tgt):
            syls = []
            for ch in tgt:
                py = pinyin(ch, style=Style.NORMAL, heteronym=True,
                            errors="ignore")
                syls.append(clean_syl(py[0][0]) if py and py[0] else "")
        if not syls or any(not s for s in syls):
            continue
        allowed = [sorted(set(syl2ids.get(s, []))) for s in syls]
        if char_w and min_w > 0:
            allowed = [[tid for tid in al
                        if char_w.get(rev[tid], {}).get(s, 1.0) >= min_w]
                       for s, al in zip(syls, allowed)]
        reach = all(any(rid_c == char2id[ch] for rid_c in al)
                    for ch, al in zip(tgt, allowed))
        logw = []
        for s, al in zip(syls, allowed):
            row = []
            for tid in al:
                ch = rev[tid]
                w = (char_w or {}).get(ch, {}).get(s, 1.0)
                row.append(w)
            logw.append(row)
        cases.append({"rid": rid, "target": tgt, "domain": domain,
                      "context": ctx, "keys": keys.get(rid, ""), "syls": syls,
                      "ctx_ids": ctx_ids, "allowed": allowed, "logw": logw,
                      "char2id": char2id, "rev": rev, "reachable": reach})
    return cases


def load_ime_raw(path):
    ime = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                rank = int(p[1])
            except ValueError:
                rank = -1
            first = p[2].split("\x1f")[0].split("\x1e")[0]
            ime[p[0]] = (rank, first)
    return ime


def first_diff(gen, tgt):
    for i, (a, b) in enumerate(zip(gen, tgt)):
        if a != b:
            return i
    return min(len(gen), len(tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--beam", default="4")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--examples", default="")
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", default="")
    ap.add_argument("--readings", default="")
    ap.add_argument("--allowed", default="pypinyin",
                    choices=["pypinyin", "rime"])
    ap.add_argument("--lambda-pron", type=float, default=0.0,
                    dest="lam_pron")
    ap.add_argument("--word-table", default="", dest="word_table")
    ap.add_argument("--lambda-word", type=float, default=0.0,
                    dest="lam_word")
    ap.add_argument("--min-reading-weight", type=float, default=0.0,
                    dest="min_rw")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    rev = {cid: ch for ch, cid in char2id.items()}
    char_w = load_readings(args.readings) if args.readings else None
    if args.allowed == "rime":
        assert char_w, "--allowed rime requires --readings"
        syl2ids = build_allowed_from_readings(char2id, char_w)
    else:
        syl2ids = build_syllable_map(char2id, CACHE)
    word_seqs = load_word_table(args.word_table, char2id) \
        if args.word_table else None
    keys, target, meta = load_requests(args.requests, args.meta)
    cases = build_cases(char2id, rev, syl2ids, keys, target, meta, args.ctx,
                        char_w=char_w, min_w=args.min_rw)
    print("config: allowed=%s readings=%s lam_pron=%s word_table=%s "
          "lam_word=%s min_rw=%s" % (args.allowed, bool(args.readings),
                                     args.lam_pron, bool(args.word_table),
                                     args.lam_word, args.min_rw), flush=True)
    print("sentences:", len(cases),
          "reachable:", sum(1 for c in cases if c["reachable"]), flush=True)

    model, tok, device = load_model(args.mlm)
    best_beams = [int(x) for x in str(args.beam).split(",")]
    outs_by_beam = {}
    for b in best_beams:
        outs = decode(cases, model, tok, device, args.ctx, beam=b,
                      topk=args.topk, lam_pron=args.lam_pron,
                      word_seqs=word_seqs, lam_word=args.lam_word)
        outs_by_beam[b] = outs
        exact = sum(1 for c, o in zip(cases, outs) if o == c["target"])
        reach = [c for c in cases if c["reachable"]]
        outs_r = [o for c, o in zip(cases, outs) if c["reachable"]]
        exact_r = sum(1 for c, o in zip(reach, outs_r)
                      if o == c["target"])
        char_ok = sum(sum(1 for a, b2 in zip(o, c["target"]) if a == b2)
                      for c, o in zip(cases, outs))
        char_tot = sum(len(c["target"]) for c in cases)
        print("beam=%d  exact_all=%d/%d (%.3f)  exact_reachable=%d/%d "
              "(%.3f)  char_acc=%.4f"
              % (b, exact, len(cases), exact / len(cases), exact_r,
                 len(reach), exact_r / max(1, len(reach)),
                 char_ok / char_tot), flush=True)
        if args.dump:
            path = args.dump if len(best_beams) == 1 else \
                args.dump.replace(".tsv", "_b%d.tsv" % b)
            with io.open(path, "w", encoding="utf-8", newline="\n") as f:
                for c, o in zip(cases, outs):
                    f.write("%s\t%s\t%s\t%s\n" % (c["rid"], c["target"], o,
                                                  c["domain"]))
            print("dump ->", path, flush=True)

    if args.examples:
        ime = load_ime_raw(args.candidates)
        n = len(cases)
        ime_exact = sum(1 for c in cases
                        if ime.get(c["rid"], (-1, ""))[0] == 0)
        ime_char_ok = 0
        for c in cases:
            top = ime.get(c["rid"], (-1, ""))[1]
            ime_char_ok += sum(1 for i, ch in enumerate(c["target"])
                               if i < len(top) and top[i] == ch)
        char_tot = sum(len(c["target"]) for c in cases)
        bmin, bmax = min(best_beams), max(best_beams)
        o_min, o_max = outs_by_beam[bmin], outs_by_beam[bmax]
        both = ime_only = mlm_only = neither = 0
        for c, o in zip(cases, o_max):
            i_ok = ime.get(c["rid"], (-1, ""))[0] == 0
            m_ok = o == c["target"]
            if i_ok and m_ok:
                both += 1
            elif i_ok:
                ime_only += 1
            elif m_ok:
                mlm_only += 1
            else:
                neither += 1
        lines = ["# 整句生成对比：MLM 逐字解码 vs IME\n",
                 "- 模型：`%s`；逐位拼音约束（heteronym 全读音），"
                 "beam=%s topk=%d，上下文上限 %d 字\n"
                 % (args.mlm, args.beam, args.topk, args.ctx),
                 "- 数据：`%s`（%d 例，4-10 字随机切片整句）\n"
                 % (os.path.basename(args.requests), n),
                 "- 聚合（整句精确匹配 / 字级）：IME %.3f（%d/%d）｜ "
                 "贪心 %.3f（%d/%d）｜ beam=%d **%.3f（%d/%d）**\n"
                 % (ime_exact / n, ime_exact, n,
                    sum(1 for o, c in zip(o_min, cases)
                        if o == c["target"]) / n,
                    sum(1 for o, c in zip(o_min, cases)
                        if o == c["target"]), n,
                    bmax,
                    sum(1 for o, c in zip(o_max, cases)
                        if o == c["target"]) / n,
                    sum(1 for o, c in zip(o_max, cases)
                        if o == c["target"]), n),
                 "- 字级准确率：IME %.3f ｜ 贪心 %.3f ｜ beam=%d %.3f\n"
                 % (ime_char_ok / char_tot,
                    sum(sum(1 for a, b in zip(o, c["target"]) if a == b)
                        for c, o in zip(cases, o_min)) / char_tot,
                    bmax,
                    sum(sum(1 for a, b in zip(o, c["target"]) if a == b)
                        for c, o in zip(cases, o_max)) / char_tot),
                 "- 交叉（beam=%d vs IME）：都对 %d ｜ MLM 独对 **%d** ｜ "
                 "IME 独对 %d ｜ 都错 %d\n" % (bmax, both, mlm_only,
                                               ime_only, neither),
                 "- 每例展示：模型实际看到的完整上下文（▢ = 生成起点）、"
                 "键入拼音、目标、IME 首选及名次、贪心与 beam=%d 输出；"
                 "随机抽样 seed=%d\n" % (bmax, args.seed), "\n---\n"]
        rng = random.Random(args.seed)
        cats = [("MLM 独对：IME 未生成正确整句", []),
                ("beam=%d 修正贪心" % bmax, []),
                ("IME 也对（beam 保持正确）", []),
                ("beam=%d 失败样例" % bmax, [])]
        for c, og, ob in zip(cases, o_min, o_max):
            i_ok = ime.get(c["rid"], (-1, ""))[0] == 0
            g_ok = og == c["target"]
            b_ok = ob == c["target"]
            if b_ok and not i_ok:
                cats[0][1].append((c, og, ob))
            elif b_ok and not g_ok:
                cats[1][1].append((c, og, ob))
            elif b_ok and i_ok:
                cats[2][1].append((c, og, ob))
            else:
                cats[3][1].append((c, og, ob))
        for title, pool in cats:
            picks = rng.sample(pool, min(args.per_group, len(pool)))
            lines.append("\n## %s（随机抽取 %d 例，共 %d 例）\n"
                         % (title, len(picks), len(pool)))
            for k, (c, og, ob) in enumerate(picks, 1):
                rank, top = ime.get(c["rid"], (-1, ""))
                ctx_n = sum(1 for ch in c["context"] if ch in c["char2id"])
                lines.append("\n### 例 %d（域：%s，目标 %d 字，"
                             "上下文 %d 字）\n"
                             % (k, c["domain"], len(c["target"]), ctx_n))
                lines.append("- 模型输入（%d 字）：`%s` ｜ 待定位置：▢\n"
                             % (ctx_n, c["context"]))
                lines.append("- 键入拼音：`%s`\n" % " ".join(c["syls"]))
                lines.append("- 目标：**%s** ｜ IME 首选：`%s`（目标名次：%s）\n"
                             % (c["target"], top,
                                rank + 1 if rank >= 0 else "未出现"))
                lines.append("- 贪心：`%s`（%s） ｜ beam=%d：`%s`（%s）\n"
                             % (og, "✓" if og == c["target"] else "✗",
                                bmax, ob,
                                "✓" if ob == c["target"] else "✗"))
                if ob != c["target"]:
                    d = first_diff(ob, c["target"])
                    lines.append("- beam=%d 首个错误位置：第 %d 字"
                                 "（目标 `%s` / 生成 `%s`）\n"
                                 % (bmax, d + 1, c["target"][d:d + 1],
                                    ob[d:d + 1]))
        with io.open(args.examples, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("examples ->", args.examples, flush=True)


if __name__ == "__main__":
    main()
