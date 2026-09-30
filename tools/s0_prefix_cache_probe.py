"""S0 probe: can a frozen (cacheable) prefix replace today's full bidirectional
forward, and how much would per-step KV caching actually save?

Everything here is offline: it only reads the deployed weights
(third-party/lm-train/reranker_v2/mlm_ft_v2) and the real validation split of
third-party/lm-train/reranker_v2/data.npz.  No plugin, model export or training
script is touched.

Ranking follows train_mlm_reranker.evaluate(): candidate-restricted softmax at
the [MASK] position, so `base_right` must reproduce the recorded val top1=0.8985
-- that is the harness self-check.

Variants
  base_right   [PAD..][CLS] ctx [MASK] [SEP]   full bidirectional   = today's inference
  base_left    [CLS] ctx [MASK] [SEP]          full bidirectional, no leading PAD (alignment effect)
  frozen       prefix = [CLS] ctx cached        prefix bidirectional, [MASK] attends to all of it
  frozen_tailK frozen prefix up to ctx[-K:]     last K ctx tokens recomputed in the chunk
  causal_left  [CLS] ctx [MASK]                 left-to-right encoding (positions from 0)
  shift1       base_right + one leading PAD     same tokens, positions +1 (position sensitivity)

Usage
  python s0_prefix_cache_probe.py parity  [--limit N] [--device cuda|cpu]
  python s0_prefix_cache_probe.py latency [--rows 1,16] [--lengths 32,48,...] [--device cpu]
  python s0_prefix_cache_probe.py export_kv [--out DIR]
  python s0_prefix_cache_probe.py ort_latency [--dir DIR] [--rows 1,16]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"D:\Workspace\rime\third-party\lm-train"
MODEL_DIR = os.path.join(TRAIN, "reranker_v2", "mlm_ft_v2")
DATA = os.path.join(TRAIN, "reranker_v2", "data.npz")
SCRATCH = os.path.join(HERE, "s0-onnx")

CLS, MASK, SEP, PAD = 101, 103, 102, 0


# ---------------------------------------------------------------- mini BERT
class MiniBert(torch.nn.Module):
    """BERT forward with an explicit key mask, so arbitrary attention patterns
    (bidirectional / prefix-LM / causal) can be compared on the same weights.

    The key mask follows HF: only *keys* are masked by `att`, a padded query
    still attends to every real key (that is what HF's 4D mask does, and it is
    why no row ends up fully masked).
    """

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        self.emb = model.bert.embeddings
        self.layers = model.bert.encoder.layer
        self.cls = model.cls
        self.heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _embed(self, ids, positions):
        h = self.emb.word_embeddings(ids)
        h = h + self.emb.position_embeddings(positions)
        tt = getattr(self.emb, "token_type_embeddings", None)
        if tt is not None:
            h = h + tt(torch.zeros_like(ids))
        return self.emb.LayerNorm(h)

    def _split(self, x, b, n):
        # size() instead of the traced example's Python ints: `b, n = x.shape`
        # would bake the export-time sequence length into the graph.
        return x.view(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def forward(self, ids, att, past=None, past_att=None, causal_in_current=False,
                positions=None):
        """ids/att: (B, n) current tokens and their key mask.
        past: list of (K, V) per layer, each (B, heads, past_len, head_dim).
        past_att: (B, past_len) key mask of the cached prefix -- PAD keys inside
        the prefix must stay masked, exactly as in the uncached forward.
        causal_in_current: current token i may only see current keys j <= i.
        Returns (hidden_at_last_position, present_kv)."""
        b, n = ids.shape
        past_len = 0 if past is None else past[0][0].shape[2]
        if positions is None:
            positions = torch.arange(past_len, past_len + n, device=ids.device)
        h = self._embed(ids, positions)

        # key mask over past + current, (B, 1, 1, past_len + n)
        if past is not None:
            pk = (past_att.bool() if past_att is not None else
                  torch.ones(b, past_len, dtype=torch.bool, device=ids.device))
            key_ok = torch.cat([pk, att.bool()], dim=1)[:, None, None, :]
        else:
            key_ok = att.bool()[:, None, None, :]

        present = []
        for layer, (k_past, v_past) in zip(
                self.layers, past if past is not None else
                [(None, None)] * len(self.layers)):
            attn = layer.attention
            q = self._split(attn.self.query(h), b, n) * self.scale
            k = self._split(attn.self.key(h), b, n)
            v = self._split(attn.self.value(h), b, n)
            if k_past is not None:
                k = torch.cat([k_past, k], dim=2)
                v = torch.cat([v_past, v], dim=2)
            present.append((k, v))

            scores = q @ k.transpose(-1, -2)                 # (B, H, n, past+n)
            allowed = key_ok.expand(-1, self.heads, -1, -1).clone()
            if causal_in_current:
                # Apply causality to the current-token block only.  The mask keeps
                # the (B, H, 1, ...) shape masked_fill broadcasts over, instead of
                # expanding the query dim and slicing it in place.
                qi = torch.arange(n, device=ids.device)[:, None]
                kj = torch.arange(n, device=ids.device)[None, :]
                causal_ok = (kj <= qi)[None, None]
                scores = torch.cat(
                    [scores[..., :past_len],
                     scores[..., past_len:].masked_fill(
                         ~causal_ok, torch.finfo(scores.dtype).min)], dim=-1)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores, dim=-1)
            ctx = (probs @ v).transpose(1, 2).flatten(2)     # (B, n, H*D)

            ctx = attn.output.dropout(attn.output.dense(ctx))
            h = attn.output.LayerNorm(h + ctx)
            inter = layer.intermediate.dense(h)
            inter = F.gelu(inter, approximate="none")        # config hidden_act: gelu
            out = layer.output.dropout(layer.output.dense(inter))
            h = layer.output.LayerNorm(h + out)

        return h, present

    def logits_at(self, hidden, index):
        return self.cls(hidden[:, index, :])


def load_model(device):
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR,
                                                 attn_implementation="eager")
    model.eval().to(device)
    return model


# ---------------------------------------------------------------- data / metrics
def load_split(limit, seed=0):
    d = np.load(DATA)
    sel = np.flatnonzero(d["split"] == 1)
    if limit and limit < len(sel):
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(sel, size=limit, replace=False))
    ctx = torch.from_numpy(d["ctx"][sel].astype(np.int64))
    cand = torch.from_numpy(d["cand"][sel].astype(np.int64))
    mask = torch.from_numpy(d["mask"][sel].astype(np.int64))
    target = torch.from_numpy(d["target_pos"][sel].astype(np.int64))
    return ctx, cand, mask, target


def restricted_scores(logits, cand, mask):
    s = logits.gather(1, cand).masked_fill(mask == 0, -1e9)
    return s


def metrics(scores, target):
    order = scores.argsort(dim=1, descending=True)
    rank = (order == target.unsqueeze(1)).nonzero()[:, 1] + 1
    n = scores.shape[0]
    return {
        "n": int(n),
        "top1": float((rank == 1).float().mean()),
        "top3": float((rank <= 3).float().mean()),
        "mrr": float((1.0 / rank.float()).mean()),
    }


def agreement(a, b):
    """top-1 agreement, mean |margin| delta and mean KL between two candidate
    distributions; margins are reported on the candidate-restricted probabilities."""
    pa, pb = F.softmax(a, dim=1), F.softmax(b, dim=1)
    top1 = float((a.argmax(1) == b.argmax(1)).float().mean())
    def m(x):
        v = x.topk(2, dim=1).values
        return v[:, 0] - v[:, 1]
    margin = float((m(pa) - m(pb)).abs().mean())
    kl = float((pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(1).mean())
    return top1, margin, kl


# ---------------------------------------------------------------- variants
def content_bounds(ctx):
    """First column that is not PAD (ctx is left-padded / right-aligned)."""
    real = (ctx != PAD)
    lengths = real.sum(dim=1)
    return lengths


def build_right(ctx):
    b = ctx.shape[0]
    dev = ctx.device
    ids = torch.cat([torch.full((b, 1), CLS, dtype=torch.long, device=dev),
                     ctx,
                     torch.full((b, 1), MASK, dtype=torch.long, device=dev),
                     torch.full((b, 1), SEP, dtype=torch.long, device=dev)], dim=1)
    att = torch.cat([torch.ones(b, 1, dtype=torch.long, device=dev),
                     (ctx != PAD).long(),
                     torch.ones(b, 2, dtype=torch.long, device=dev)], dim=1)
    return ids, att, 1 + ctx.shape[1]          # index of [MASK]


def build_left_bucket(ctx, idx):
    """Exact left-aligned rows for one group of equal content length:
    [CLS] ctx_real [MASK] [SEP], positions from 0 (no padding at all)."""
    b, w = ctx.shape
    dev = ctx.device
    L = int((ctx[idx] != PAD)[0].sum())
    rows = ctx[idx][:, w - L:]
    ids = torch.cat([torch.full((len(idx), 1), CLS, dtype=torch.long, device=dev),
                     rows,
                     torch.full((len(idx), 1), MASK, dtype=torch.long, device=dev),
                     torch.full((len(idx), 1), SEP, dtype=torch.long, device=dev)], dim=1)
    return ids, torch.ones_like(ids), L


def run_variants(model, mb, ctx, cand, mask, target, device, tail_ks=(0, 8, 16, 32, 64)):
    results, score_keep = {}, {}
    b, w = ctx.shape
    target = target.to(device)

    # --- base_right: today's inference (also the anchor for val top1) ---
    ids, att, mask_pos = build_right(ctx)
    ids, att = ids.to(device), att.to(device)
    with torch.inference_mode():
        h, _ = mb(ids, att)
        s_base = restricted_scores(mb.logits_at(h, mask_pos), cand, mask)
    results["base_right"] = metrics(s_base, target)
    score_keep["base_right"] = s_base

    # --- shift1: identical tokens, positions +1 (isolates position sensitivity) ---
    with torch.inference_mode():
        h, _ = mb(ids, att, positions=torch.arange(1, ids.shape[1] + 1, device=device))
        s_shift = restricted_scores(mb.logits_at(h, mask_pos), cand, mask)
    results["shift1"] = metrics(s_shift, target)
    score_keep["shift1"] = s_shift

    # --- frozen / frozen_tailK: prefix K/V from a bidirectional prefix pass,
    #     then the remaining ctx tail + [MASK] [SEP] in one chunk with causal
    #     attention inside the chunk.  Layout and positions are identical to
    #     base_right, so the only difference is what the frozen prefix saw.
    for k in tail_ks:
        cut = ids.shape[1] - 2 - k              # tokens kept in the frozen prefix
        modes = ("causal",) if k == 0 else ("causal", "bidir")
        for mode in modes:
            with torch.inference_mode():
                _, past = mb(ids[:, :cut], att[:, :cut])
                cur, cur_att = ids[:, cut:], att[:, cut:]
                h, _ = mb(cur, cur_att, past=past, past_att=att[:, :cut],
                          causal_in_current=(mode == "causal"))
                s = restricted_scores(mb.logits_at(h, cur.shape[1] - 2), cand, mask)
            if k == 0:
                name = "frozen"
            else:
                name = ("frozen_tail%d" % k) if mode == "causal" else ("frozen_bi%d" % k)
            results[name] = metrics(s, target)
            score_keep[name] = s

    # --- left-aligned family, one exact layout per content length ---
    lengths = (ctx != PAD).sum(dim=1)
    s_left = torch.empty_like(score_keep["base_right"])
    s_causal = torch.empty_like(s_left)
    for L in torch.unique(lengths).tolist():
        idx = (lengths == L).nonzero().flatten()
        if idx.numel() == 0:
            continue
        ids_l, att_l, L = build_left_bucket(ctx, idx)
        ids_l, att_l = ids_l.to(device), att_l.to(device)
        cand_b, mask_b = cand[idx], mask[idx]
        with torch.inference_mode():
            h, _ = mb(ids_l, att_l)
            s_left[idx] = restricted_scores(mb.logits_at(h, 1 + L), cand_b, mask_b)

            # strict left-to-right encoding of [CLS] ctx, then the [MASK] step
            past = None
            for t in range(ids_l.shape[1] - 2):
                _, past = mb(ids_l[:, t:t + 1], torch.ones(len(idx), 1, dtype=torch.long,
                                                           device=device),
                             past=past, causal_in_current=True)
            cur = torch.full((len(idx), 2), MASK, dtype=torch.long, device=device)
            h, _ = mb(cur, torch.ones_like(cur), past=past, causal_in_current=True)
            s_causal[idx] = restricted_scores(mb.logits_at(h, 0), cand_b, mask_b)
    results["base_left"] = metrics(s_left, target)
    score_keep["base_left"] = s_left
    results["causal_left"] = metrics(s_causal, target)
    score_keep["causal_left"] = s_causal

    return results, score_keep


def cmd_parity(args):
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = load_model(device)
    mb = MiniBert(model).to(device).eval()
    ctx, cand, mask, target = load_split(args.limit)
    ctx, cand, mask = ctx.to(device), cand.to(device), mask.to(device)

    # self-check: the mini forward must reproduce the HF model
    ids, att, mask_pos = build_right(ctx[:16])
    with torch.inference_mode():
        h, _ = mb(ids, att)
        mine = mb.logits_at(h, mask_pos)
        theirs = model(input_ids=ids, attention_mask=att).logits[:, -2, :]
    print("self-check max|mini - HF| = %.3e" % (mine - theirs).abs().max().item(),
          flush=True)

    # The explicit attention matrix is (rows, heads, seq, seq); chunk the rows so
    # the full validation split fits in 8 GB.
    chunks = {}
    for start in range(0, ctx.shape[0], args.chunk):
        sl = slice(start, start + args.chunk)
        with torch.inference_mode():
            _, sc = run_variants(model, mb, ctx[sl], cand[sl], mask[sl], target[sl],
                                 device)
        for k, v in sc.items():
            chunks.setdefault(k, []).append(v.float().cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores = {k: torch.cat(v) for k, v in chunks.items()}
    target = target.cpu()
    results = {k: metrics(v, target) for k, v in scores.items()}
    base = scores["base_right"]

    print("\n%-14s %6s %8s %8s %8s | %10s %10s %10s" %
          ("variant", "n", "top1", "top3", "mrr", "top1agree", "d|margin|", "KL"))
    order = ["base_right", "base_left", "shift1", "frozen", "frozen_tail8",
             "frozen_bi8", "frozen_tail16", "frozen_bi16", "frozen_tail32",
             "frozen_bi32", "frozen_tail64", "frozen_bi64", "causal_left"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        if name == "base_right":
            print("%-14s %6d %8.4f %8.4f %8.4f | %10s %10s %10s" %
                  (name, r["n"], r["top1"], r["top3"], r["mrr"], "-", "-", "-"))
            continue
        a, m, kl = agreement(base, scores[name])
        print("%-14s %6d %8.4f %8.4f %8.4f | %10.4f %10.4f %10.4f" %
              (name, r["n"], r["top1"], r["top3"], r["mrr"], a, m, kl))
    print("\nJSON " + json.dumps({"variants": results,
                                  "vs_base_right": {
                                      k: dict(zip(("top1agree", "d_margin", "kl"),
                                                  agreement(base, v)))
                                      for k, v in scores.items() if k != "base_right"}},
                                 ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- latency
class IncrementalBert(torch.nn.Module):
    """Exportable "frozen prefix + recomputed tail" step.

    The attention that `parity` selected: the cached prefix keys/values stay
    frozen (with the prefix's own key mask), while the current chunk -- the last
    K context tokens plus [MASK] [SEP] -- is recomputed with *bidirectional*
    attention among itself.  Nothing here is causal, so the graph stays close to
    the model the weights were trained as.

    inputs : input_ids (B, n), past_attention_mask (B, P), position_ids (n),
             past K/V per layer
    outputs: logits at the [MASK] (second to last chunk position), present K/V

    position_ids is an explicit input on purpose: deriving positions from the
    cache length becomes a traced constant, which silently bakes the export-time
    prefix length into the graph.
    """

    def __init__(self, mb, layers):
        super().__init__()
        self.mb = mb
        self.layers = layers

    def forward(self, input_ids, past_attention_mask, position_ids, *past):
        b, n = input_ids.shape
        att = torch.ones_like(input_ids)
        pk = [(past[2 * i], past[2 * i + 1]) for i in range(self.layers)]
        h, present = self.mb(input_ids, att, past=pk,
                             past_att=past_attention_mask.long(),
                             positions=position_ids)
        # The last two chunk positions are [MASK] [SEP]; returning both keeps the
        # read position independent of the runtime sequence length (a baked index
        # would only work at the export-time length).
        return (self.mb.cls(h.narrow(1, h.size(1) - 2, 2)),
                *[t for kv in present for t in kv])


class PrefixEncoder(torch.nn.Module):
    """Bidirectional pass that returns the cache instead of logits."""

    def __init__(self, mb, layers):
        super().__init__()
        self.mb = mb
        self.layers = layers

    def forward(self, input_ids, attention_mask, position_ids):
        _, present = self.mb(input_ids, attention_mask.long(), positions=position_ids)
        return tuple(t for kv in present for t in kv)


def cmd_latency(args):
    device = torch.device(args.device)
    model = load_model(device)
    mb = MiniBert(model).to(device).eval()
    rows = [int(x) for x in args.rows.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    tail = args.tail

    def timeit(fn, warmup=2, reps=5):
        with torch.inference_mode():
            for _ in range(warmup):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps * 1000.0

    print("device=%s threads=%d rows=%s tail=%d" %
          (device, torch.get_num_threads(), rows, tail))
    print("\n%-8s %-6s %10s %11s %10s %10s %9s" %
          ("ctx_len", "rows", "full ms", "prefix ms", "step ms", "chunk tok", "speedup"))
    for L in lengths:
        for R in rows:
            ids = torch.full((R, L + 2), 500, dtype=torch.long, device=device)
            att = torch.ones_like(ids)
            full = timeit(lambda: mb(ids, att))
            cut = max(L - tail, 1)
            pref_ids = torch.full((R, cut), 500, dtype=torch.long, device=device)
            pref_att = torch.ones_like(pref_ids)
            prefix = timeit(lambda: mb(pref_ids, pref_att))
            with torch.inference_mode():
                _, past = mb(pref_ids, pref_att)
                cur = torch.full((R, min(tail, L - 1) + 2), 500, dtype=torch.long,
                                 device=device)
                cur[:, -2] = MASK
                cur[:, -1] = SEP
                cur_att = torch.ones_like(cur)
                p_att = pref_att
            step = timeit(lambda: mb(cur, cur_att, past=past, past_att=p_att))
            print("%-8d %-6d %10.2f %11.2f %10.2f %10d %8.2fx" %
                  (L, R, full, prefix, step, cur.shape[1], full / step))
    return 0


# ---------------------------------------------------------------- onnx
def _kv_names(layers):
    return (["past_key_values.%d.%s" % (l, kv) for l in range(layers)
             for kv in ("key", "value")],
            ["present.%d.%s" % (l, kv) for l in range(layers)
             for kv in ("key", "value")])


def _export(module, inputs, path, names_in, names_out, dyn):
    # The shipped export_onnx.py goes through the legacy (TorchScript) exporter;
    # the dynamo exporter cannot convert this many dynamic_axes to dynamic_shapes,
    # so keep the same path here.
    torch.onnx.export(module, inputs, path, input_names=names_in,
                      output_names=names_out, dynamic_axes=dyn, opset_version=17,
                      do_constant_folding=True, dynamo=False)
    import onnx
    proto = onnx.load(path)
    onnx.save_model(proto, path, save_as_external_data=False)
    data = path + ".data"
    if os.path.exists(data):
        os.remove(data)
    print("exported %s (%.1f MB)" % (path, os.path.getsize(path) / 1048576))


def cmd_export_kv(args):
    """Export the two graphs the cached scheme needs, plus an I/O contract check."""
    model = load_model(torch.device("cpu"))
    mb = MiniBert(model).to("cpu").eval()
    layers = len(mb.layers)
    heads, head_dim = mb.heads, mb.head_dim
    os.makedirs(args.out, exist_ok=True)
    kv_in, kv_out = _kv_names(layers)

    # 1) prefix encoder: bidirectional pass -> cache (this is what gets frozen)
    enc_in = ["input_ids", "attention_mask", "position_ids"]
    enc = PrefixEncoder(mb, layers)
    ids = torch.full((1, 64), 500, dtype=torch.long)
    att = torch.ones_like(ids)
    enc_pos = torch.arange(64, dtype=torch.long)
    enc_dyn = {"input_ids": {0: "batch", 1: "seq"},
               "attention_mask": {0: "batch", 1: "seq"},
               "position_ids": {0: "seq"}}
    for l in range(layers):
        for kv in ("key", "value"):
            enc_dyn["present.%d.%s" % (l, kv)] = {0: "batch", 2: "total_seq"}
    _export(enc, (ids, att, enc_pos), os.path.join(args.out, "encode_kv.onnx"),
            enc_in, kv_out, enc_dyn)

    # 2) step: chunk (+ frozen prefix) -> logits + next cache
    step = IncrementalBert(mb, layers)
    P, K = 64, args.tail
    step_ids = torch.full((1, K + 2), 500, dtype=torch.long)
    step_ids[:, -2] = MASK
    step_ids[:, -1] = SEP
    past_mask = torch.ones(1, P, dtype=torch.long)
    pos = torch.arange(P, P + K + 2, dtype=torch.long)
    past = []
    for _ in range(layers):
        past += [torch.zeros(1, heads, P, head_dim), torch.zeros(1, heads, P, head_dim)]
    step_in = ["input_ids", "past_attention_mask", "position_ids"] + kv_in
    step_out = ["logits"] + kv_out
    step_dyn = {"input_ids": {0: "batch", 1: "seq"},
                "past_attention_mask": {0: "batch", 1: "past_seq"},
                "position_ids": {0: "seq"}}
    for l in range(layers):
        for kv in ("key", "value"):
            step_dyn["past_key_values.%d.%s" % (l, kv)] = {0: "batch", 2: "past_seq"}
            step_dyn["present.%d.%s" % (l, kv)] = {0: "batch", 2: "total_seq"}
    _export(step, (step_ids, past_mask, pos, *past),
            os.path.join(args.out, "step_kv.onnx"), step_in, step_out, step_dyn)

    # 3) contract check: the exported step must reproduce the torch step exactly.
    #    Run at the export-time chunk length *and* at a different one: a traced
    #    graph can silently bake the example's sequence length in.
    #    NOTE: torch.onnx.export leaves the module in training mode, so the eager
    #    reference must be recomputed after an explicit .eval() -- otherwise this
    #    compares dropout-on against dropout-off and reports garbage.
    import onnxruntime as ort
    mb.eval()
    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    step_sess = ort.InferenceSession(os.path.join(args.out, "step_kv.onnx"), opt,
                                     providers=["CPUExecutionProvider"])
    enc_sess = ort.InferenceSession(os.path.join(args.out, "encode_kv.onnx"), opt,
                                    providers=["CPUExecutionProvider"])
    for test_n in (K + 2, 12):
        with torch.inference_mode():
            ctx = torch.randint(1, 2000, (1, 40))
            ids, att, mpos = build_right(ctx)
            cut = 40 - (test_n - 2)
            pref_ids, pref_att = ids[:, :1 + cut], att[:, :1 + cut]
            cur = ids[:, 1 + cut:1 + cut + test_n]
            cur_att = att[:, 1 + cut:1 + cut + test_n]
            _, past = mb(pref_ids, pref_att)
            h, present = mb(cur, cur_att, past=past, past_att=pref_att)
            torch_logits = mb.logits_at(h, cur.shape[1] - 2).numpy()
            ref_full = mb.logits_at(mb(ids, att)[0], mpos).numpy()
        feed = {"input_ids": cur.numpy(), "past_attention_mask": pref_att.numpy(),
                "position_ids": np.arange(pref_ids.shape[1],
                                          pref_ids.shape[1] + cur.shape[1], dtype=np.int64)}
        for l in range(layers):
            for kv, tensor in zip(("key", "value"), past[l]):
                feed["past_key_values.%d.%s" % (l, kv)] = tensor.numpy()
        out = step_sess.run(None, feed)
        onnx_logits, onnx_kv = out[0][:, 0, :], out[1:]
        kv_diff = max(float(np.abs(onnx_kv[2 * l + j] - t.numpy()).max())
                      for l in range(layers) for j, t in enumerate(present[l]))
        print("export check chunk=%-3d: max|torch_step - onnx_step| = %.3e "
              "(cache %.3e)" % (test_n, float(np.abs(torch_logits - onnx_logits).max()),
                                kv_diff))
    # the encoder graph must also be length-generic
    for plen in (64, 9):
        with torch.inference_mode():
            p_ids = torch.randint(1, 2000, (2, plen))
            p_att = torch.ones_like(p_ids)
            _, past_ref = mb(p_ids, p_att)
        e_out = enc_sess.run(None, {"input_ids": p_ids.numpy(),
                                    "attention_mask": p_att.numpy(),
                                    "position_ids": np.arange(plen, dtype=np.int64)})
        e_diff = max(float(np.abs(e_out[2 * l + j] - past_ref[l][j].numpy()).max())
                     for l in range(layers) for j in range(2))
        print("export check encode  len=%-3d: max|torch - onnx| = %.3e" % (plen, e_diff))
    print("(informational) random-context full vs frozen_tail%d: max|dlogit| = %.3e" %
          (args.tail, float(np.abs(ref_full - torch_logits).max())))
    return 0


def cmd_ort_latency(args):
    """Today's full pass vs the cached scheme, ONNX Runtime CPU, real cache tensors."""
    import onnxruntime as ort
    rows = [int(x) for x in args.rows.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    tail = args.tail
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = args.threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    full = ort.InferenceSession(os.path.join(args.dir, "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(os.path.join(SCRATCH, "encode_kv.onnx"), so,
                               providers=["CPUExecutionProvider"])
    step = ort.InferenceSession(os.path.join(SCRATCH, "step_kv.onnx"), so,
                                providers=["CPUExecutionProvider"])
    heads, head_dim, n_layers = 4, 64, 4
    kv_in, kv_out = _kv_names(n_layers)

    def timeit(fn, warmup=3, reps=15):
        """Median of `reps` runs: the mean is dominated by scheduler noise."""
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    print("onnxruntime %s intra_op_threads=%d rows=%s tail=%d" %
          (ort.__version__, args.threads, rows, tail))
    print("\n%-8s %-5s %9s %10s %9s %8s %9s %9s" %
          ("ctx_len", "rows", "full ms", "prefix ms", "step ms", "chunk", "cache MB", "speedup"))
    for L in lengths:
        for R in rows:
            ids = np.full((R, L + 3), 500, dtype=np.int64)
            att = np.ones_like(ids)
            mpos = np.full((R,), L + 2, dtype=np.int64)
            full_ms = timeit(lambda: full.run(None, {"input_ids": ids,
                                                     "attention_mask": att,
                                                     "mask_index": mpos}))
            cut = max(L - tail, 1)
            p_ids = np.full((R, cut), 500, dtype=np.int64)
            p_att = np.ones_like(p_ids)
            enc_ms = timeit(lambda: enc.run(None, {"input_ids": p_ids,
                                                   "attention_mask": p_att,
                                                   "position_ids": np.arange(cut, dtype=np.int64)}))
            step_ids = np.full((R, min(tail, L) + 2), 500, dtype=np.int64)
            step_ids[:, -2] = MASK
            step_ids[:, -1] = SEP
            feed = {"input_ids": step_ids, "past_attention_mask": p_att,
                    "position_ids": np.arange(cut, cut + step_ids.shape[1],
                                              dtype=np.int64)}
            cache_bytes = 0
            for l in range(n_layers):
                for kv in ("key", "value"):
                    arr = np.zeros((R, heads, cut, head_dim), dtype=np.float32)
                    cache_bytes += arr.nbytes
                    feed["past_key_values.%d.%s" % (l, kv)] = arr
            step_ms = timeit(lambda: step.run(None, feed))
            print("%-8d %-5d %9.2f %10.2f %9.2f %8d %9.2f %8.2fx" %
                  (L, R, full_ms, enc_ms, step_ms, step_ids.shape[1],
                   cache_bytes / 1048576, full_ms / step_ms))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_parity)
    p = sub.add_parser("latency")
    p.add_argument("--rows", default="1,16")
    p.add_argument("--lengths", default="32,48,64,96,128,200,256")
    p.add_argument("--tail", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_latency)
    p = sub.add_parser("export_kv")
    p.add_argument("--out", default=SCRATCH)
    p.add_argument("--tail", type=int, default=32)
    p.set_defaults(func=cmd_export_kv)
    p = sub.add_parser("ort_latency")
    p.add_argument("--dir", default=os.path.join(TRAIN, "onnx", "mlm_ft_v2"))
    p.add_argument("--rows", default="1,16")
    p.add_argument("--lengths", default="32,48,64,96,128,200,256")
    p.add_argument("--tail", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_ort_latency)
    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
