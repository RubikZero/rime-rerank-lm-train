"""Mirror the C++ plugin decode exactly (ORT CPU, mask_index model,
syllable_readings.tsv) for a single case, to compare with Python/torch.

Usage: python ort_parity_one.py --requests ... --meta ... --case-dir <model_dir>
"""

import argparse
import io
import math
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lm_decode_sentences import (  # noqa
    VOCAB, build_cases, clean_syl, load_char2id, load_requests, load_readings,
    build_allowed_from_readings)
from pypinyin import Style, lazy_pinyin

DATA = r"D:\Workspace\rime\weasel\output\data"


def load_plugin_table(path):
    """syllable_readings.tsv (syllable, char, weight) -> ids/logw."""
    char2id = load_char2id(VOCAB)
    ids, logw = {}, {}
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        syl, ch = f[0], f[1]
        cid = char2id.get(ch)
        if cid is None:
            continue
        ids.setdefault(syl, []).append(cid)
        logw.setdefault(syl, []).append(math.log(max(float(f[2]), 1e-6)))
    return ids, logw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--plugin-table", default="")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lam-pron", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--cxx-context", action="store_true",
                    help="encode OOV context chars as [UNK] like the plugin")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    rev = {cid: ch for ch, cid in char2id.items()}
    if args.plugin_table:
        syl_ids, syl_logw = load_plugin_table(args.plugin_table)
    else:
        char_w = load_readings(os.path.join(HERE, "testset",
                                            "char_readings.tsv"))
        syl_ids = build_allowed_from_readings(char2id, char_w)
        syl_logw = {}
        for ch, readings in char_w.items():
            cid = char2id.get(ch)
            if cid is None:
                continue
            for syl, w in readings.items():
                syl_logw.setdefault(syl, []).append(math.log(max(w, 1e-6)))

    keys, target, meta = load_requests(args.requests, args.meta)
    sess = ort.InferenceSession(
        os.path.join(args.model_dir, "model.onnx"),
        providers=["CPUExecutionProvider"])

    for rid, tgt in target.items():
        kind, domain, context = meta[rid]
        if args.cxx_context:
            raw = list(context)[-args.ctx:]
            vals = [char2id.get(c, 100) for c in raw]
        else:
            raw = list(context)[-args.ctx:]
            vals = [char2id[c] for c in raw if c in char2id]
        ctx = np.asarray([0] * (args.ctx - len(vals)) + vals, dtype=np.int64)
        syls = [clean_syl(s) for s in
                lazy_pinyin(tgt, style=Style.NORMAL, errors="ignore")]
        beams = [([], 0.0)]
        for t, s in enumerate(syls):
            al = syl_ids.get(s, [])
            lw = syl_logw.get(s, [])
            if not al:
                print("no chars for", s)
                return
            rows = []
            for g, sc in beams:
                pre = [char2id[x] for x in g]
                ids = [101] + list(ctx) + pre + [103, 102]
                att = [1] + [1 if v != 0 else 0 for v in ctx] + [1] * (len(pre) + 2)
                rows.append((g, sc, ids, att))
            L = max(len(r[2]) for r in rows)
            inp = np.zeros((len(rows), L), dtype=np.int64)
            att = np.zeros_like(inp)
            pos = []
            for i, (g, sc, ids, at) in enumerate(rows):
                inp[i, :len(ids)] = ids
                att[i, :len(at)] = at
                pos.append(len(ids) - 2)
            mi = np.asarray(pos, dtype=np.int64)
            logits = sess.run(["logits"], {"input_ids": inp,
                                           "attention_mask": att,
                                           "mask_index": mi})[0]
            logits = np.asarray(logits)
            if logits.ndim == 3:
                logits = logits[:, 0, :]
            x = logits.astype(np.float64)
            x -= x.max(axis=-1, keepdims=True)
            lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
            cands = []
            lw_a = np.asarray(lw) if lw else np.zeros(len(al))
            for r in range(len(rows)):
                sub = lp[r, np.asarray(al)] + args.lam_pron * lw_a
                idx = np.argsort(-sub)[:args.topk]
                for i in idx:
                    cands.append((rows[r][0] + [rev[al[i]]],
                                  rows[r][1] + float(sub[i])))
            cands.sort(key=lambda p: -p[1])
            beams = cands[:args.beam]
        out = "".join(beams[0][0])
        print("rid=%s ORT-CPU output=%s target=%s %s"
              % (rid, out, tgt, "OK" if out == tgt else "MISS"))


if __name__ == "__main__":
    main()
