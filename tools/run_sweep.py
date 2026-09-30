"""Run the pronunciation-weight / word-constraint ablation matrix.

Usage: python run_sweep.py hard [--only A,B2.0] [--beam 4]
Appends results to testset/sweep_results.tsv.
"""

import argparse
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MLM = r"D:\Workspace\rime\third-party\lm-train\reranker_v2\mlm_ft_v2"
RESULTS = os.path.join(HERE, "testset", "sweep_results.tsv")

SETS = {
    "hard": ("testset/requests_sent_100.tsv", "testset/meta_sent_100.tsv",
             "testset/candidates_sent_hard_luna_pinyin_simp.tsv"),
    "words": ("testset/requests_words_100.tsv", "testset/meta_words_100.tsv",
              "testset/candidates_sent_words_luna_pinyin_simp.tsv"),
    "hard_hold": ("testset/requests_hard_hold.tsv",
                  "testset/meta_hard_hold.tsv",
                  "testset/candidates_sent_hard_luna_pinyin_simp.tsv"),
    "words_hold": ("testset/requests_words_hold.tsv",
                   "testset/meta_words_hold.tsv",
                   "testset/candidates_sent_words_luna_pinyin_simp.tsv"),
}

CONFIGS = [
    ("A", []),
    ("B0.5", ["--readings", "testset/char_readings.tsv",
              "--lambda-pron", "0.5"]),
    ("B1", ["--readings", "testset/char_readings.tsv",
            "--lambda-pron", "1"]),
    ("B2", ["--readings", "testset/char_readings.tsv",
            "--lambda-pron", "2"]),
    ("B4", ["--readings", "testset/char_readings.tsv",
            "--lambda-pron", "4"]),
    ("R0", ["--allowed", "rime", "--readings", "testset/char_readings.tsv"]),
    ("R2", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
            "--lambda-pron", "2"]),
    ("R4", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
            "--lambda-pron", "4"]),
    ("C0.5", ["--word-table", "testset/word_pron.tsv",
              "--lambda-word", "0.5"]),
    ("C1", ["--word-table", "testset/word_pron.tsv", "--lambda-word", "1"]),
    ("C2", ["--word-table", "testset/word_pron.tsv", "--lambda-word", "2"]),
    ("C4", ["--word-table", "testset/word_pron.tsv", "--lambda-word", "4"]),
    ("C8", ["--word-table", "testset/word_pron.tsv", "--lambda-word", "8"]),
    ("RC1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--word-table", "testset/word_pron.tsv", "--lambda-word", "1"]),
    ("RC2", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--word-table", "testset/word_pron.tsv", "--lambda-word", "2"]),
    ("RC4", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--word-table", "testset/word_pron.tsv", "--lambda-word", "4"]),
    ("P0.001", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--word-table", "testset/word_pron.tsv", "--lambda-word", "2",
                "--min-reading-weight", "0.001"]),
    ("P0.01", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
               "--word-table", "testset/word_pron.tsv", "--lambda-word", "2",
               "--min-reading-weight", "0.01"]),
    ("P0.05", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
               "--word-table", "testset/word_pron.tsv", "--lambda-word", "2",
               "--min-reading-weight", "0.05"]),
    ("RP0.01", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--min-reading-weight", "0.01"]),
    ("D", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
           "--lambda-pron", "2", "--word-table", "testset/word_pron.tsv",
           "--lambda-word", "1"]),
    ("R0B0.25", ["--allowed", "rime", "--readings",
                 "testset/char_readings.tsv", "--lambda-pron", "0.25"]),
    ("R0B0.5", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--lambda-pron", "0.5"]),
    ("RP0.005", ["--allowed", "rime", "--readings",
                 "testset/char_readings.tsv", "--min-reading-weight", "0.005"]),
    ("RP0.02", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--min-reading-weight", "0.02"]),
    ("PW0.5", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
               "--min-reading-weight", "0.01", "--word-table",
               "testset/word_pron.tsv", "--lambda-word", "0.5"]),
    ("PW1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron.tsv", "--lambda-word", "1"]),
    ("PD1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron_dict.tsv", "--lambda-word", "1"]),
    ("PD2", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron_dict.tsv", "--lambda-word", "2"]),
    ("PD4", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron_dict.tsv", "--lambda-word", "4"]),
    ("PH1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron_hi.tsv", "--lambda-word", "1"]),
    ("PH2", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--min-reading-weight", "0.01", "--word-table",
             "testset/word_pron_hi.tsv", "--lambda-word", "2"]),
    ("PB0.5a", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--lambda-pron", "0.5", "--min-reading-weight", "0.005"]),
    ("PB0.5b", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--lambda-pron", "0.5", "--min-reading-weight", "0.01"]),
    ("PB1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--lambda-pron", "1", "--min-reading-weight", "0.005"]),
    ("PBW", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--lambda-pron", "0.5", "--min-reading-weight", "0.005",
             "--word-table", "testset/word_pron.tsv", "--lambda-word", "0.5"]),
    ("PBH", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--lambda-pron", "0.5", "--min-reading-weight", "0.005",
             "--word-table", "testset/word_pron_hi.tsv",
             "--lambda-word", "0.5"]),
    ("PBD", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--lambda-pron", "0.5", "--min-reading-weight", "0.005",
             "--word-table", "testset/word_pron_dict.tsv",
             "--lambda-word", "1"]),
    ("R0B0.75", ["--allowed", "rime", "--readings",
                 "testset/char_readings.tsv", "--lambda-pron", "0.75"]),
    ("R0B1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
              "--lambda-pron", "1"]),
    ("RW0.25", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
                "--lambda-pron", "0.5", "--word-table",
                "testset/word_pron.tsv", "--lambda-word", "0.25"]),
    ("RW0.5", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
               "--lambda-pron", "0.5", "--word-table",
               "testset/word_pron.tsv", "--lambda-word", "0.5"]),
    ("RD1", ["--allowed", "rime", "--readings", "testset/char_readings.tsv",
             "--lambda-pron", "0.5", "--word-table",
             "testset/word_pron_dict.tsv", "--lambda-word", "1"]),
]

LINE = re.compile(r"beam=(\d+)\s+exact_all=(\d+)/(\d+)\s+\(([\d.]+)\)"
                  r"\s+exact_reachable=(\d+)/(\d+)\s+\(([\d.]+)\)"
                  r"\s+char_acc=([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--only", default="")
    ap.add_argument("--beam", default="4")
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args()

    req, meta, cand = SETS[args.set]
    only = {x for x in args.only.split(",") if x}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with io.open(RESULTS, "a", encoding="utf-8", newline="\n") as rf:
        if os.path.getsize(RESULTS) == 0:
            rf.write("set\tconfig\tbeam\texact\tn\tchar_acc\tsecs\n")
        for name, flags in CONFIGS:
            if only and name not in only:
                continue
            cmd = [sys.executable, "lm_decode_sentences.py", "--mlm", MLM,
                   "--requests", req, "--meta", meta, "--candidates", cand,
                   "--beam", args.beam] + flags
            t0 = time.time()
            env = dict(os.environ, PYTHONIOENCODING="utf-8",
                       TRANSFORMERS_VERBOSITY="error")
            p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
            secs = time.time() - t0
            m = None
            for line in (p.stdout or "").splitlines():
                mm = LINE.search(line)
                if mm:
                    m = mm
            if not m:
                print("[%s/%s] FAILED rc=%d\n%s" % (args.set, name, p.returncode,
                                                    (p.stderr or "")[-800:]))
                continue
            exact, n, char_acc = m.group(2), m.group(3), m.group(8)
            print("[%s/%s] beam=%s exact=%s/%s (%.3f) char_acc=%.4f  %.0fs"
                  % (args.set, name, m.group(1), exact, n,
                     int(exact) / int(n), float(char_acc), secs), flush=True)
            rf.write("%s\t%s\t%s\t%s\t%s\t%s\t%.0f\n"
                     % (args.set, name, m.group(1), exact, n, char_acc, secs))
            rf.flush()
            if args.dump_dir:
                dst = os.path.join(args.dump_dir,
                                   "%s_%s.tsv" % (args.set, name))
                cmd2 = cmd + ["--dump", dst]
                subprocess.run(cmd2, cwd=HERE, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env=env)


if __name__ == "__main__":
    main()
