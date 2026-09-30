"""Pack extracted pinyin candidates into reranker training tensors.

Filters to single Han-char candidates, caps to top-K by quality (+ target),
and stores context ids, candidate ids, numeric features and masks.
"""

import argparse
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RERANK = os.path.join(HERE, "reranker")
VOCAB = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256\vocab.txt"
CTX = 64
K = 20


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
        (0xF900 <= o <= 0xFAFF)


def load_char2id(path):
    char2id = {}
    with io.open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\n")
            if len(tok) == 1:
                char2id.setdefault(tok, idx)
    return char2id


def parse_candidates(path):
    out = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            rid = parts[0]
            cands = []
            if len(parts) > 2 and parts[2]:
                for c in parts[2].split("\x1f"):
                    f4 = c.split("\x1e")
                    if len(f4) < 4:
                        continue
                    text, comment, quality, ctype = f4[0], f4[1], f4[2], f4[3]
                    try:
                        q = float(quality)
                    except ValueError:
                        q = 0.0
                    cands.append((text, q, ctype))
            out[rid] = cands
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RERANK)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--requests", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = args.dir
    args.requests = args.requests or os.path.join(d, "requests.tsv")
    args.meta = args.meta or os.path.join(d, "meta.tsv")
    args.candidates = args.candidates or os.path.join(
        d, "candidates_luna_pinyin_simp.tsv")
    args.out = args.out or os.path.join(d, "data.npz")
    context_len = args.ctx

    char2id = load_char2id(VOCAB)
    req = {}
    with io.open(args.requests, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                req[p[0]] = p[2]      # id -> target
    meta = {}
    with io.open(args.meta, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                meta[p[0]] = (p[1], p[2])  # id -> (split, context)
    cands = parse_candidates(args.candidates)
    print("requests=%d meta=%d cands=%d" % (len(req), len(meta), len(cands)),
          flush=True)

    type_vocab = {"<unk>": 0}
    rows = []
    dropped_no_target = 0
    dropped_oov = 0
    for rid, target in req.items():
        if rid not in meta or rid not in cands:
            continue
        split, context = meta[rid]
        tid = char2id.get(target)
        if tid is None:
            dropped_oov += 1
            continue
        # single Han-char candidates, dedup by char keeping best quality
        by_char = {}
        for rank, (text, q, ctype) in enumerate(cands[rid]):
            if len(text) != 1 or not is_han(text) or text not in char2id:
                continue
            prev = by_char.get(text)
            if prev is None or q > prev[0]:
                by_char[text] = (q, rank, ctype)
        if target not in by_char:
            dropped_no_target += 1
            continue
        items = [(ch, q, rank, ctype) for ch, (q, rank, ctype)
                 in by_char.items()]
        items.sort(key=lambda x: -x[1])
        if len(items) > K:
            head = items[:K]
            target_item = next(x for x in items if x[0] == target)
            if target_item not in head:
                head.append(target_item)
            items = head
        for _, _, _, ctype in items:
            if ctype not in type_vocab:
                type_vocab[ctype] = len(type_vocab)

        n = len(items)
        ctx_ids = np.zeros(context_len, dtype=np.int16)
        ctx = context[-context_len:] if len(context) > context_len else context
        ctx_vals = [char2id[c] for c in ctx if c in char2id]
        pad = context_len - len(ctx_vals)
        ctx_ids[pad:] = ctx_vals
        cand_ids = np.zeros(n, dtype=np.int16)
        logq = np.zeros(n, dtype=np.float32)
        rankn = np.zeros(n, dtype=np.float32)
        types = np.zeros(n, dtype=np.int16)
        for k, (ch, q, rank, ctype) in enumerate(items):
            cand_ids[k] = char2id[ch]
            logq[k] = np.log1p(max(q, 0.0))
            rankn[k] = rank / max(1, len(cands[rid]) - 1)
            types[k] = type_vocab[ctype]
        rows.append({"split": 0 if split == "train" else 1,
                     "ctx": ctx_ids, "cand": cand_ids, "logq": logq,
                     "rankn": rankn, "types": types,
                     "target_pos": next(i for i, x in enumerate(items)
                                        if x[0] == target)})
        if len(rows) % 50000 == 0:
            print("packed=%d" % len(rows), flush=True)

    N = len(rows)
    C = max(len(r["cand"]) for r in rows)
    C = min(C, K + 1)
    print("cases=%d C=%d dropped(no_target)=%d dropped(oov)=%d" %
          (N, C, dropped_no_target, dropped_oov), flush=True)

    def zeros():
        return np.zeros((N, C), dtype=np.float32)

    data = {
        "ctx": np.zeros((N, context_len), dtype=np.int16),
        "cand": np.zeros((N, C), dtype=np.int16),
        "logq": zeros(),
        "rankn": zeros(),
        "types": np.zeros((N, C), dtype=np.int16),
        "mask": np.zeros((N, C), dtype=np.int8),
        "target_pos": np.zeros(N, dtype=np.int16),
        "split": np.zeros(N, dtype=np.int8),
    }
    for i, r in enumerate(rows):
        n = len(r["cand"])
        data["ctx"][i] = r["ctx"]
        data["cand"][i, :n] = r["cand"]
        data["logq"][i, :n] = r["logq"]
        data["rankn"][i, :n] = r["rankn"]
        data["types"][i, :n] = r["types"]
        data["mask"][i, :n] = 1
        data["target_pos"][i] = r["target_pos"]
        data["split"][i] = r["split"]

    np.savez_compressed(args.out, **data)
    with io.open(os.path.join(d, "type_vocab.json"), "w",
                 encoding="utf-8") as f:
        json.dump(type_vocab, f, ensure_ascii=False, indent=1)
    n_train = int((data["split"] == 0).sum())
    n_val = int((data["split"] == 1).sum())
    print("saved %s: train=%d val=%d C=%d types=%d" %
          (args.out, n_train, n_val, C, len(type_vocab)), flush=True)


if __name__ == "__main__":
    main()
