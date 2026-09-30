"""Build pronunciation tables from the rime luna_pinyin data (Step 1).

Outputs (into testset/):
- char_readings.tsv : simplified char -> syllable -> weight
  weights come from the explicit percentages in luna_pinyin.dict.yaml
  (heteronym entries); readings without an explicit percentage are
  assigned the remaining probability mass (or 1.0 when the char has no
  percentages at all).
- word_pron.tsv     : simplified word -> syllable sequence -> source/weight
  sources: dict.yaml multi-char entries (authoritative codes) and
  essay.txt phrases (top-2 reading combinations by weighted char readings).

Usage: python build_readings.py [--stats]
"""

import argparse
import io
import os
import sys
from collections import defaultdict

import zhconv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_reranker import VOCAB, load_char2id  # noqa

DATA = r"D:\Workspace\rime\weasel\output\data"
DICT_YAML = os.path.join(DATA, "luna_pinyin.dict.yaml")
ESSAY_TXT = os.path.join(DATA, "essay.txt")
OUT_CHAR = os.path.join(HERE, "testset", "char_readings.tsv")
OUT_WORD = os.path.join(HERE, "testset", "word_pron.tsv")
TESTSET = os.path.join(HERE, "testset")


def clean_syl(s):
    s = s.replace("ü", "v")
    return "".join(c for c in s if "a" <= c <= "z")


def load_dict():
    char_raw = defaultdict(dict)  # simp char -> {syl: weight or None}
    words = []  # (simp word, syl_seq, weight, source)
    for line in io.open(DICT_YAML, encoding="utf-8", errors="replace"):
        if not line or line[0] in "#-.":
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 2 or not f[0]:
            continue
        simp = zhconv.convert(f[0], "zh-hans")
        syls = [clean_syl(x) for x in f[1].split()]
        syls = [s for s in syls if s]
        if not syls:
            continue
        w = None
        if len(f) > 2 and f[2].endswith("%"):
            try:
                w = float(f[2][:-1]) / 100.0
            except ValueError:
                w = None
        if len(simp) == 1 and len(syls) == 1:
            old = char_raw[simp].get(syls[0])
            if old is None or (w is not None and w > old):
                char_raw[simp][syls[0]] = w
        elif len(simp) > 1 and len(syls) == len(simp):
            wt = None
            if len(f) > 2 and not f[2].endswith("%"):
                try:
                    wt = float(f[2])
                except ValueError:
                    wt = None
            words.append((simp, " ".join(syls), wt, "dict"))
    return char_raw, words


def finalize_char_weights(char_raw):
    char_w = {}
    for ch, readings in char_raw.items():
        expl = {s: w for s, w in readings.items() if w is not None}
        miss = [s for s, w in readings.items() if w is None]
        if not expl:
            for s in miss:
                char_w.setdefault(ch, {})[s] = 1.0
            continue
        total = sum(expl.values())
        if total > 1.0:
            expl = {s: w / total for s, w in expl.items()}
            total = 1.0
        rest = max(0.0, 1.0 - total)
        share = rest / len(miss) if miss else 0.0
        for s, w in expl.items():
            char_w.setdefault(ch, {})[s] = max(1e-3, w)
        for s in miss:
            char_w.setdefault(ch, {})[s] = max(1e-3, share)
    return char_w


def best_reading_seqs(ch, char_w, topn=2):
    """Top reading combinations for a word: per-char top-2 then top-2 products."""
    opts = []
    for c in ch:
        r = char_w.get(c)
        if not r:
            return []
        ranked = sorted(r.items(), key=lambda kv: -kv[1])[:2]
        opts.append(ranked)
    combos = [([], 1.0)]
    for ranked in opts:
        nxt = []
        for seq, p in combos:
            for syl, w in ranked:
                nxt.append((seq + [syl], p * w))
        nxt.sort(key=lambda x: -x[1])
        combos = nxt[:topn]
    if not combos:
        return []
    best = combos[0][1]
    return [c for c in combos if c[1] >= 0.25 * best]


def write_word_variants(src_path, out_dir):
    """Derived tables used by the ablation configs (see run_sweep.py)."""
    dict_path = os.path.join(out_dir, "word_pron_dict.tsv")
    hi_path = os.path.join(out_dir, "word_pron_hi.tsv")
    nd = nh = 0
    with io.open(src_path, encoding="utf-8") as src, \
            io.open(dict_path, "w", encoding="utf-8", newline="\n") as outd, \
            io.open(hi_path, "w", encoding="utf-8", newline="\n") as outh:
        for line in src:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            if f[2] == "dict":
                outd.write(line)
                nd += 1
            else:
                try:
                    w = float(f[3]) if f[3] else 0.0
                except ValueError:
                    w = 0.0
                if w >= 500.0:
                    outh.write(line)
                    nh += 1
    print("variants: dict=%d hi>=500=%d -> %s, %s"
          % (nd, nh, dict_path, hi_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    vocab = set(char2id)

    char_raw, words = load_dict()
    char_w = finalize_char_weights(char_raw)

    for ch, readings in char_raw.items():
        weights = [w for w in readings.values() if w is not None]
        if weights and abs(sum(weights) - 1.0) > 0.05:
            print("WARN sum!=1", ch, readings)

    with io.open(OUT_CHAR, "w", encoding="utf-8", newline="\n") as f:
        f.write("# char\tsyllable\tweight\t(source: pct=explicit, rest=derived)\n")
        for ch in sorted(char_w):
            for syl in sorted(char_w[ch]):
                src = "pct" if char_raw.get(ch, {}).get(syl) is not None \
                    else "rest"
                f.write("%s\t%s\t%.4f\t%s\n"
                        % (ch, syl, char_w[ch][syl], src))

    word_rows = {}
    for simp, seq, wt, src in words:
        if not simp or any(c not in vocab for c in simp):
            continue
        key = (simp, seq)
        if key not in word_rows or (wt or 0) > (word_rows[key][0] or 0):
            word_rows[key] = (wt, src)
    dict_words = {w for (w, _) in word_rows}

    essay = 0
    for line in io.open(ESSAY_TXT, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if not f or len(f[0]) < 2 or len(f[0]) > 4:
            continue
        simp = zhconv.convert(f[0], "zh-hans")
        if any(c not in vocab for c in simp) or simp in dict_words:
            continue
        try:
            wt = float(f[1]) if len(f) > 1 else None
        except ValueError:
            wt = None
        for seq, p in best_reading_seqs(simp, char_w):
            key = (simp, " ".join(seq))
            if key not in word_rows:
                word_rows[key] = (wt, "essay")
                essay += 1

    with io.open(OUT_WORD, "w", encoding="utf-8", newline="\n") as f:
        f.write("# word\tsyllables\tsource\tweight\n")
        for (simp, seq) in sorted(word_rows):
            wt, src = word_rows[(simp, seq)]
            f.write("%s\t%s\t%s\t%s\n"
                    % (simp, seq, src, "" if wt is None else "%.2f" % wt))

    print("chars: %d (with explicit pct: %d)"
          % (len(char_w), sum(1 for c in char_w.values()
                              for w in c.values() if w < 1.0)))
    print("words: %d (dict %d, essay+ %d)"
          % (len(word_rows), len({k for k in word_rows
                                  if word_rows[k][1] == "dict"}), essay))
    print("-> %s\n-> %s" % (OUT_CHAR, OUT_WORD))
    write_word_variants(OUT_WORD, os.path.dirname(OUT_WORD))

    if args.stats:
        cover_check(char_w, word_rows)


def cover_check(char_w, word_rows):
    import json
    from pypinyin import Style, lazy_pinyin
    from diag_errors import clean_syl as cs, load_requests
    syl2chars = defaultdict(set)
    for ch, readings in char_w.items():
        for s in readings:
            syl2chars[s].add(ch)

    words_by_seq = defaultdict(set)
    for (w, seq) in word_rows:
        words_by_seq[seq].add(w)

    for req, meta, label in (
            ("requests_sent_100.tsv", "meta_sent_100.tsv", "hard100"),
            ("requests_words_100.tsv", "meta_words_100.tsv", "words100")):
        _, target, _ = load_requests(os.path.join(TESTSET, req),
                                     os.path.join(TESTSET, meta))
        n = syl_ok = char_ok = word_cov = 0
        for rid, tgt in target.items():
            if not tgt:
                continue
            syls = [cs(s) for s in lazy_pinyin(tgt, style=Style.NORMAL,
                                               errors="ignore")]
            if len(syls) != len(tgt):
                continue
            n += 1
            if all(s in syl2chars for s in syls):
                syl_ok += 1
            if all(c in syl2chars.get(s, ()) for c, s in zip(tgt, syls)):
                char_ok += 1
            cov = [False] * len(tgt)
            for i in range(len(tgt)):
                for k in range(min(4, len(tgt) - i), 0, -1):
                    seq = " ".join(syls[i:i + k])
                    if tgt[i:i + k] in words_by_seq.get(seq, ()):
                        for j in range(i, i + k):
                            cov[j] = True
                        break
            word_cov += sum(cov)
        tot = sum(len(t) for t in target.values())
        print("[%s] n=%d syl_ok=%d char_ok=%d word_char_coverage=%.3f"
              % (label, n, syl_ok, char_ok, word_cov / max(1, tot)))


if __name__ == "__main__":
    main()
