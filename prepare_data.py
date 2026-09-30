"""Build task-adaptive MLM samples for IME candidate re-ranking.

Each sample mimics the inference format: a context (committed text) followed by
a masked target word of 1-4 characters. Targets are chosen at word boundaries
from the luna_pinyin dictionary when possible.
"""

import argparse
import io
import os
import random

import numpy as np
import zhconv

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"
CORPUS = r"D:\Workspace\rime\third-party\corpus\CLUECorpusSmall.txt"
DICT = r"D:\Workspace\rime\weasel\output\data\luna_pinyin.dict.yaml"
OUT_DIR = os.path.join(HERE, "data")

CTX = 64
SEQ = CTX + 8
CLS, SEP, MASK, PAD = 101, 102, 103, 0


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or (0xF900 <= o <= 0xFAFF)


def load_char2id(path):
    char2id = {}
    with io.open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.rstrip("\n")
            if len(token) == 1:
                char2id.setdefault(token, idx)
    return char2id


def load_words(path):
    words = set()
    started = False
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not started:
                if line.strip() == "...":
                    started = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0].strip():
                continue
            word = zhconv.convert(parts[0].strip(), "zh-cn")
            if 1 <= len(word) <= 4 and all(is_han(c) for c in word):
                words.add(word)
    return words


def pick_target(sent, words, words2, rng, limit=None):
    n = min(len(sent), limit) if limit else len(sent)
    hits = []
    for i in range(n):
        if i + 2 > n or sent[i:i + 2] not in words2:
            continue
        for length in (4, 3, 2):
            if i + length <= n and sent[i:i + length] in words:
                hits.append((i, length))
                break
    if hits:
        return rng.choice(hits)
    for _ in range(10):
        i = rng.randrange(n)
        if is_han(sent[i]):
            return (i, 1)
    return None


def make_sample(sent, words, words2, char2id, rng, short_ratio=0.5):
    limit = min(len(sent), 12) if rng.random() < short_ratio else None
    picked = pick_target(sent, words, words2, rng, limit)
    if picked is None:
        return None
    pos, length = picked
    target = sent[pos:pos + length]
    context = sent[max(0, pos - CTX):pos]
    ctx_ids = [char2id.get(c) for c in context]
    tgt_ids = [char2id.get(c) for c in target]
    if any(i is None for i in ctx_ids) or any(i is None for i in tgt_ids):
        return None
    if len(ctx_ids) < 2:
        return None
    pad_len = CTX - len(ctx_ids)
    ids = [PAD] * pad_len + [CLS] + ctx_ids + [MASK] * len(tgt_ids) + [SEP]
    labels = [-100] * (pad_len + 1 + len(ctx_ids)) + tgt_ids + [-100]
    mask = [0] * pad_len + [1] * (len(ids) - pad_len)
    ids += [PAD] * (SEQ - len(ids))
    labels += [-100] * (SEQ - len(labels))
    mask += [0] * (SEQ - len(mask))
    return ids, mask, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=1_500_000)
    ap.add_argument("--val-samples", type=int, default=5000)
    ap.add_argument("--sample-rate", type=float, default=0.05)
    ap.add_argument("--max-scan-lines", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--short-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    char2id = load_char2id(os.path.join(BASE, "vocab.txt"))
    words = load_words(DICT)
    words2 = {w[:2] for w in words if len(w) >= 2}
    print("vocab chars:", len(char2id), "dict words:", len(words), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_ids = np.zeros((args.max_samples, SEQ), dtype=np.int16)
    train_mask = np.zeros((args.max_samples, SEQ), dtype=np.int16)
    train_labels = np.full((args.max_samples, SEQ), -100, dtype=np.int16)
    val_ids = np.zeros((args.val_samples, SEQ), dtype=np.int16)
    val_mask = np.zeros((args.val_samples, SEQ), dtype=np.int16)
    val_labels = np.full((args.val_samples, SEQ), -100, dtype=np.int16)
    n_train = n_val = 0
    scanned = 0
    kept = 0
    with io.open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            scanned += 1
            if args.max_scan_lines and scanned > args.max_scan_lines:
                break
            if n_train >= args.max_samples:
                break
            if rng.random() > args.sample_rate:
                continue
            sent = line.strip()
            if len(sent) < 16 or len(sent) > 300:
                continue
            sample = make_sample(sent, words, words2, char2id, rng,
                                 args.short_ratio)
            if sample is None:
                continue
            ids, mask, labels = sample
            if n_val < args.val_samples:
                val_ids[n_val] = ids
                val_mask[n_val] = mask
                val_labels[n_val] = labels
                n_val += 1
            else:
                train_ids[n_train] = ids
                train_mask[n_train] = mask
                train_labels[n_train] = labels
                n_train += 1
            kept += 1
            if kept % 100000 == 0:
                print("scanned=%d kept=%d" % (scanned, kept), flush=True)

    def save(name, ids, mask, labels, count):
        path = os.path.join(args.out_dir, name)
        np.savez(path, ids=ids[:count], mask=mask[:count],
                 labels=labels[:count])
        print("saved", path, (count, SEQ), flush=True)

    save("train.npz", train_ids, train_mask, train_labels, n_train)
    save("val.npz", val_ids, val_mask, val_labels, n_val)


if __name__ == "__main__":
    main()
