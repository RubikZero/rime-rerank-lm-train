"""Classify sentence-decode failures (Step 0 of the pronunciation-weight plan).

For every failed case (output != target) assign a primary cause:

- coverage : some target char is not allowed by its requested syllable
             (decoder literally cannot produce the target)
- rare     : output uses a char admitted via a rare reading (< 10%, from
             luna_pinyin.dict.yaml percentages) while the target char has a
             common reading for the same syllable
- word     : output is much less dictionary-word-supported than the target
             (target - output word coverage >= 2 chars), i.e. word-level
             constraints / penalties could plausibly fix it
- other    : none of the above (pure LM ranking / context error)

Usage:
  python diag_errors.py --requests testset/requests_sent_100.tsv \
      --meta testset/meta_sent_100.tsv --dump testset/py_sent_100.tsv \
      --label "python hard100"
  python diag_errors.py ... --cand testset/cand_h100_f.tsv --label "cpp hard100"
"""

import argparse
import io
import os
import sys
from collections import Counter, defaultdict

from pypinyin import Style, lazy_pinyin, pinyin
import zhconv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, load_char2id  # noqa

DATA = r"D:\Workspace\rime\weasel\output\data"
DICT_YAML = os.path.join(DATA, "luna_pinyin.dict.yaml")
ESSAY_TXT = os.path.join(DATA, "essay.txt")
RARE = 0.10


def clean_syl(s):
    s = s.replace("ü", "v")
    return "".join(c for c in s if "a" <= c <= "z")


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


def load_dump(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            out[p[0]] = p[2]
    return out


def load_cand_first(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            first = p[2].split("\x1f")[0].split("\x1e")[0]
            out[p[0]] = first
    return out


def load_dict():
    """char_readings: simp char -> {syl: weight}; words: simp word -> set(codes)."""
    char_w = defaultdict(dict)
    words = set()
    for line in io.open(DICT_YAML, encoding="utf-8", errors="replace"):
        if not line or line[0] in "#-.":
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 2 or not f[0]:
            continue
        text = zhconv.convert(f[0], "zh-hans")
        code = f[1].strip()
        if len(text) == 1:
            syl = clean_syl(code)
            if not syl:
                continue
            w = None
            if len(f) > 2 and f[2].endswith("%"):
                try:
                    w = float(f[2][:-1]) / 100.0
                except ValueError:
                    w = None
            old = char_w[text].get(syl)
            if old is None or (w is not None and w > old):
                char_w[text][syl] = w
        else:
            words.add(text)
    return char_w, words


def load_essay_words():
    words = set()
    for line in io.open(ESSAY_TXT, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if f and len(f[0]) > 1:
            words.add(zhconv.convert(f[0], "zh-hans"))
    return words


def word_coverage(s, words, maxlen=4):
    """Chars covered by greedy longest-match dictionary words (len >= 2)."""
    cov = [False] * len(s)
    i = 0
    while i < len(s):
        matched = 0
        for k in range(min(maxlen, len(s) - i), 1, -1):
            if s[i:i + k] in words:
                matched = k
                break
        if matched:
            for j in range(i, i + matched):
                cov[j] = True
            i += matched
        else:
            i += 1
    return sum(cov)


def allowed_chars(char2id, cache):
    import json
    if os.path.exists(cache):
        with io.open(cache, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("size") == len(char2id):
            return {k: set(v) for k, v in data["map"].items()}
    m = defaultdict(set)
    for ch in char2id:
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
                m[syl].add(ch)
            elif r == "ê":
                m["e"].add(ch)
    return m


def target_syllables(tgt):
    syls = [clean_syl(s) for s in
            lazy_pinyin(tgt, style=Style.NORMAL, errors="ignore")]
    if len(syls) != len(tgt):
        syls = []
        for ch in tgt:
            py = pinyin(ch, style=Style.NORMAL, heteronym=True,
                        errors="ignore")
            syls.append(clean_syl(py[0][0]) if py and py[0] else "")
    return syls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--cand", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    keys, target, meta = load_requests(args.requests, args.meta)
    tgtmap = args.dump and load_dump(args.dump) or load_cand_first(args.cand)
    char2id = load_char2id(VOCAB)
    syl2chars = allowed_chars(
        char2id, os.path.join(HERE, "testset", "syllable2chars.json"))
    char_w, dict_words = load_dict()
    essay_words = load_essay_words()
    words = dict_words | essay_words

    n = len(tgtmap)
    fails = []
    stats = Counter()
    ex = defaultdict(list)
    for rid, out in tgtmap.items():
        tgt = target.get(rid, "")
        if not tgt or out == tgt:
            continue
        syls = target_syllables(tgt)
        reach = (len(syls) == len(tgt) and all(
            ch in syl2chars.get(s, ()) for ch, s in zip(tgt, syls)))
        rare_pos = []
        for i, (a, b) in enumerate(zip(out, tgt)):
            if a == b or i >= len(syls):
                continue
            wo = char_w.get(a, {}).get(syls[i])
            wt = char_w.get(b, {}).get(syls[i])
            if wo is not None and wo < RARE and (wt is None or wt > wo):
                rare_pos.append((i, a, b, wo, wt))
        wc_t = word_coverage(tgt, words) if reach else 0
        wc_o = word_coverage(out, words)
        word_gain = wc_t - wc_o
        cls = []
        if not reach:
            cls.append("coverage")
        if rare_pos:
            cls.append("rare")
        if word_gain >= 2 and reach:
            cls.append("word")
        if not cls:
            cls.append("other")
        stats[tuple(cls)] += 1
        if len(ex[cls[0]]) < args.examples:
            syl_disp = " ".join(syls)
            ex[cls[0]].append(
                (rid, tgt, out, meta.get(rid, ("", "", ""))[2][-20:],
                 syl_disp, rare_pos, wc_t, wc_o))

    print("=== %s ===" % args.label)
    print("cases=%d failed=%d" % (n, n - sum(
        1 for rid, o in tgtmap.items() if o == target.get(rid, ""))))
    primary = Counter()
    for combo, c in stats.items():
        primary[combo[0]] += c
    for k, v in primary.most_common():
        print("  %-9s %3d (%.1f%%)" % (k, v, 100.0 * v / max(1, len(
            [1 for rid, o in tgtmap.items()
             if o != target.get(rid, "")]))))
    multi = {k: v for k, v in stats.items() if len(k) > 1}
    if multi:
        print("  combos:", {"/".join(k): v for k, v in multi.items()})
    for cls, rows in ex.items():
        print("--- examples: %s ---" % cls)
        for rid, tgt, out, ctx, syl, rare_pos, wc_t, wc_o in rows:
            print("  #%s target=%s out=%s wcov=%d/%d" % (rid, tgt, out,
                                                         wc_t, wc_o))
            print("      syl: %s | ctx: ...%s" % (syl, ctx))
            if rare_pos:
                parts = syl.split()
                print("      rare: %s" % [
                    "pos%d %s(syl=%s w=%.3f)<-%s" %
                    (i, a, parts[i], wo, b) for i, a, b, wo, wt in rare_pos])


if __name__ == "__main__":
    main()
