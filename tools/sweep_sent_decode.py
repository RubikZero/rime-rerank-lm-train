"""Sweep beam x topk for constrained sentence decoding: accuracy vs latency.

Times only the decode loop (model load excluded, GPU synchronized), so the
numbers reflect per-sentence generation cost on this machine.
"""

import argparse
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lm_decode_sentences import (CACHE, VOCAB, build_cases,  # noqa
                                 build_syllable_map, decode,
                                 load_char2id, load_ime_raw, load_model,
                                 load_requests)
import torch  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--beams", default="1,4,8,12")
    ap.add_argument("--topks", default="8,12,16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    char2id = load_char2id(VOCAB)
    rev = {cid: ch for ch, cid in char2id.items()}
    syl2ids = build_syllable_map(char2id, CACHE)
    keys, target, meta = load_requests(args.requests, args.meta)
    cases = build_cases(char2id, rev, syl2ids, keys, target, meta, args.ctx)
    ime = load_ime_raw(args.candidates)
    n = len(cases)
    char_tot = sum(len(c["target"]) for c in cases)
    ime_exact = sum(1 for c in cases if ime.get(c["rid"], (-1, ""))[0] == 0)
    ime_char = sum(sum(1 for i, ch in enumerate(c["target"])
                       if i < len(ime.get(c["rid"], (-1, ""))[1])
                       and ime.get(c["rid"], (-1, ""))[1][i] == ch)
                   for c in cases)
    print("cases=%d  IME exact=%.4f char=%.4f" %
          (n, ime_exact / n, ime_char / char_tot), flush=True)

    model, tok, device = load_model(args.mlm)
    cuda = device == "cuda"

    def run(beam, topk, subset):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = decode(subset, model, tok, device, args.ctx, beam=beam,
                      topk=topk)
        if cuda:
            torch.cuda.synchronize()
        return outs, time.perf_counter() - t0

    if args.warmup:
        run(4, 8, cases[:args.warmup])

    beams = [int(x) for x in args.beams.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    rows = []
    for beam in beams:
        for topk in topks:
            outs, dt = run(beam, topk, cases)
            exact = sum(1 for c, o in zip(cases, outs)
                        if o == c["target"])
            ch = sum(sum(1 for a, b in zip(o, c["target"]) if a == b)
                     for c, o in zip(cases, outs))
            rows.append((beam, topk, exact / n, ch / char_tot, dt,
                         dt / n * 1000))
            print("beam=%-2d topk=%-2d exact=%.4f char=%.4f "
                  "time=%6.1fs (%.2f ms/sent)" %
                  (beam, topk, exact / n, ch / char_tot, dt, dt / n * 1000),
                  flush=True)

    if args.out:
        lines = ["| beam | topk | 整句精确匹配 | 字级 | 总耗时(400例) | "
                 "每句延迟 |", "|---:|---:|---:|---:|---:|---:|",
                 "| IME | — | %.4f | %.4f | — | — |"
                 % (ime_exact / n, ime_char / char_tot)]
        for beam, topk, ex, ch, dt, ms in rows:
            lines.append("| %d | %d | %.4f | %.4f | %.1fs | %.1f ms |"
                         % (beam, topk, ex, ch, dt, ms))
        best = max(rows, key=lambda r: r[2])
        lines.append("\n- 最优：beam=%d topk=%d（exact=%.4f，%.1f ms/句）"
                     % (best[0], best[1], best[2], best[5]))
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("written:", args.out, flush=True)


if __name__ == "__main__":
    main()
