"""Export a fine-tuned MLM to ONNX for the IME plugin.

Two graphs are exported:
  model_full.onnx  : logits for every position (reference/eval)
  model.onnx       : logits at an explicit mask_index only (plugin, low memory)
plus dynamic int8 versions, vocab.txt, lm_config.json and (optional) the
syllable table.

--kv (default) additionally exports the prefix-cache pair used by the plugin's
context cache ("freeze the far part of the window, recompute a short tail"):
  encode_prefix.onnx : [CLS] + context prefix -> key/value per layer
  score_tail.onnx    : tail + candidate + [MASK] [SEP] (+ frozen prefix) -> logits
"""

import argparse
import io
import json
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_VOCAB = os.path.join(HERE, "..", "models", "uer",
                          "chinese_roberta_L-4_H-256", "vocab.txt")


class MaskIndexWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.cls = model.cls

    def forward(self, input_ids, attention_mask, mask_index):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        b, s, h = hidden.shape
        idx = mask_index.view(b, 1, 1).expand(b, 1, h)
        selected = torch.gather(hidden, 1, idx)
        return self.cls(selected)


# --------------------------------------------------------------------------
# Prefix-cache graphs (plugin: "context cache").
#
# The plugin freezes the far part of the context and recomputes only a short tail
# next to the mask.  Freezing is only sound if the tail keeps *bidirectional*
# attention among itself (weasel/doc/lm-kv-cache-s0.md): freezing the far prefix
# and recomputing the last 16-32 tokens reproduces the plain forward to
# 99.6-99.8% top-1 agreement, while a causal tail is 1.4pt worse.
#
# The step graph returns the last two positions instead of indexing the mask
# dynamically: a Python int index would bake the export-time length into the graph.
# --------------------------------------------------------------------------
class FrozenAttentionBert(torch.nn.Module):
    """BERT encoder driven by an explicit key mask.

    Only keys are masked (as in HF): a padded query still attends to every real
    key, and keys of the frozen prefix keep the prefix's own mask.
    """

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        self.emb = model.bert.embeddings
        self.layers = model.bert.encoder.layer
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.hidden_act = getattr(cfg, "hidden_act", "gelu")

    def _split(self, x):
        # size() instead of the traced example's Python ints: `b, n = x.shape`
        # would bake the export-time sequence length into the graph.
        return x.view(x.size(0), x.size(1), self.num_heads,
                      self.head_dim).transpose(1, 2)

    def _embed(self, ids, positions):
        h = self.emb.word_embeddings(ids) + self.emb.position_embeddings(positions)
        tt = getattr(self.emb, "token_type_embeddings", None)
        if tt is not None:
            h = h + tt(torch.zeros_like(ids))
        return self.emb.LayerNorm(h)

    def forward(self, input_ids, key_mask, position_ids, past=None):
        """key_mask covers past + current keys (current only when past is None)."""
        h = self._embed(input_ids, position_ids)
        if past is not None:
            past_len = past[0][0].size(2)
            keys_ok = torch.cat(
                [key_mask[:, :past_len].bool()[:, None, None, :],
                 key_mask[:, past_len:].bool()[:, None, None, :]], dim=3)
        else:
            keys_ok = key_mask.bool()[:, None, None, :]
        present = []
        for idx, layer in enumerate(self.layers):
            attn = layer.attention
            q = self._split(attn.self.query(h)) * self.scale
            k = self._split(attn.self.key(h))
            v = self._split(attn.self.value(h))
            if past is not None:
                k = torch.cat([past[idx][0], k], dim=2)
                v = torch.cat([past[idx][1], v], dim=2)
            present.append((k, v))
            scores = q @ k.transpose(-1, -2)
            scores = scores.masked_fill(~keys_ok, torch.finfo(scores.dtype).min)
            ctx = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).flatten(2)
            ctx = attn.output.dropout(attn.output.dense(ctx))
            h = attn.output.LayerNorm(h + ctx)
            inter = layer.intermediate.dense(h)
            inter = (F.gelu(inter) if self.hidden_act == "gelu"
                     else F.gelu(inter, approximate="tanh"))
            h = layer.output.LayerNorm(
                h + layer.output.dropout(layer.output.dense(inter)))
        return h, present


class PrefixEncoder(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.encoder = FrozenAttentionBert(model)
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids):
        _, present = self.encoder(input_ids, attention_mask.long(), position_ids)
        return tuple(t for kv in present for t in kv)


class TailStep(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.encoder = FrozenAttentionBert(model)
        self.cls = model.cls
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids,
                past_attention_mask, *past):
        # The frozen prefix is shared by every row of the batch, so the plugin
        # feeds it with batch 1; expand it inside the graph instead of making the
        # caller copy it per row.
        rows = input_ids.size(0)
        pk = [(past[2 * i].expand(rows, -1, -1, -1),
               past[2 * i + 1].expand(rows, -1, -1, -1))
              for i in range(self.num_layers)]
        mask = torch.cat([past_attention_mask.long(), attention_mask.long()], dim=1)
        h, _ = self.encoder(input_ids, mask, position_ids, past=pk)
        # last two positions are [MASK] [SEP]; returning both keeps the read
        # position independent of the runtime chunk length
        return self.cls(h.narrow(1, h.size(1) - 2, 2))


def dedupe_initializers(path):
    """Collapse byte-identical initializers and repoint the nodes.

    The legacy tracer writes a parameter once per module path, so the same tensor
    can land in the file twice.  A tied embedding / LM head does *not* collapse
    here: Gather needs [vocab, hidden] and MatMul the transpose, so those two
    copies legitimately differ in layout.
    """
    import onnx
    from onnx import numpy_helper
    proto = onnx.load(path)
    seen, remap, keep = {}, {}, []
    for init in proto.graph.initializer:
        key = (init.data_type, tuple(init.dims),
               numpy_helper.to_array(init).tobytes())
        if key in seen:
            remap[init.name] = seen[key]
        else:
            seen[key] = init.name
            keep.append(init)
    if not remap:
        return 0
    for node in proto.graph.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]
    del proto.graph.initializer[:]
    proto.graph.initializer.extend(keep)
    onnx.save_model(proto, path)
    return len(remap)


def export_prefix_cache(model, out_dir, args):
    """Export the two prefix-cache graphs next to the mask-index model."""
    num_layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // heads
    seq, tail = args.kv_seq, args.kv_tail

    prefix = PrefixEncoder(model, num_layers).eval()
    ids = torch.full((1, seq), 500, dtype=torch.long)
    att = torch.ones_like(ids)
    pos = torch.arange(seq, dtype=torch.long)
    kv_names = ["present.%d.%s" % (l, kv) for l in range(num_layers)
                for kv in ("key", "value")]
    dyn = {"input_ids": {0: "batch", 1: "seq"},
           "attention_mask": {0: "batch", 1: "seq"},
           "position_ids": {0: "seq"}}
    for name in kv_names:
        dyn[name] = {0: "batch", 2: "total_seq"}
    t0 = time.time()
    # dynamo=False: this many dynamic axes cannot be turned into dynamic_shapes
    torch.onnx.export(prefix, (ids, att, pos),
                      os.path.join(out_dir, "encode_prefix.onnx"),
                      input_names=["input_ids", "attention_mask", "position_ids"],
                      output_names=kv_names, dynamic_axes=dyn,
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print("exported prefix encoder in %.1fs" % (time.time() - t0), flush=True)

    step = TailStep(model, num_layers).eval()
    step_ids = torch.full((1, tail + 2), 500, dtype=torch.long)
    step_ids[:, -2] = 103          # [MASK]
    step_ids[:, -1] = 102          # [SEP]
    past_mask = torch.ones(1, seq, dtype=torch.long)
    step_pos = torch.arange(seq, seq + tail + 2, dtype=torch.long)
    past = []
    for _ in range(num_layers):
        past += [torch.zeros(1, heads, seq, head_dim),
                 torch.zeros(1, heads, seq, head_dim)]
    kv_in = ["past.%d.%s" % (l, kv) for l in range(num_layers)
             for kv in ("key", "value")]
    step_dyn = {"input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {0: "seq"},
                "past_attention_mask": {0: "batch", 1: "past_seq"}}
    for name in kv_in:
        step_dyn[name] = {0: "batch", 2: "past_seq"}
    t0 = time.time()
    torch.onnx.export(step, (step_ids, torch.ones_like(step_ids), step_pos,
                             past_mask, *past),
                      os.path.join(out_dir, "score_tail.onnx"),
                      input_names=["input_ids", "attention_mask", "position_ids",
                                   "past_attention_mask"] + kv_in,
                      output_names=["logits"], dynamic_axes=step_dyn,
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print("exported tail scorer in %.1fs" % (time.time() - t0), flush=True)

    for name in ("encode_prefix.onnx", "score_tail.onnx"):
        p = os.path.join(out_dir, name)
        data = p + ".data"
        if os.path.exists(data):
            import onnx
            proto = onnx.load(p)
            onnx.save_model(proto, p, save_as_external_data=False)
            os.remove(data)
        dropped = dedupe_initializers(p)
        print("%-20s %.1f MB%s" % (name, os.path.getsize(p) / 1048576.0,
                                   "  (deduped %d)" % dropped if dropped else ""))

    if args.kv_int8 != "none":
        targets = ["encode_prefix.onnx"] if args.kv_int8 == "encode" else \
                  ["encode_prefix.onnx", "score_tail.onnx"]
        for name in targets:
            src = os.path.join(out_dir, name)
            tmp = src + ".int8.tmp"
            quantize(src, tmp)
            os.replace(tmp, src)
            print("%-20s %.1f MB (int8)" % (name, os.path.getsize(src) / 1048576.0),
                  flush=True)


def export(model, args, out_path, input_names, forward_inputs, dynamic_axes,
           output_names):
    torch.onnx.export(
        model, forward_inputs, out_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=args.opset,
        do_constant_folding=True)
    import onnx
    proto = onnx.load(out_path)
    onnx.save_model(proto, out_path, save_as_external_data=False)
    data_path = out_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)


def quantize(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seq", type=int, default=260)
    ap.add_argument("--kv-only", action="store_true",
                    help="export only the prefix-cache graphs (skips the "
                         "mask-index/full/int8 models, which stay as they are)")
    ap.add_argument("--no-kv", dest="kv", action="store_false",
                    help="skip the prefix-cache graphs (encode_prefix/score_tail)")
    ap.add_argument("--kv-seq", type=int, default=64,
                    help="example prefix length for the prefix-cache graphs")
    ap.add_argument("--kv-tail", type=int, default=16,
                    help="example recomputed-tail length for score_tail.onnx")
    ap.add_argument("--kv-int8", choices=("none", "encode", "both"),
                    default="encode",
                    help="quantize the prefix-cache graphs: encode costs no "
                         "measurable accuracy (99.50%% agreement either way) and "
                         "saves 23 MB; both saves another 40 MB but the tail "
                         "scorer loses about 1.5pt of top-1 agreement")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()

    ids = torch.randint(1, 2000, (1, args.seq), dtype=torch.long)
    att = torch.ones_like(ids)

    if args.kv_only:
        # lm_config.json / model.onnx are already in place and unchanged; this
        # path only adds the prefix-cache graphs next to them.
        export_prefix_cache(model, args.out, args)
        return

    full_path = os.path.join(args.out, "model_full.onnx")
    t0 = time.time()
    export(model, args, full_path,
           ["input_ids", "attention_mask"], (ids, att),
           {"input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"}},
           ["logits"])
    print("exported full fp32 in %.1fs" % (time.time() - t0), flush=True)
    quantize(full_path, os.path.join(args.out, "model_full.int8.onnx"))

    mask_index = torch.tensor([args.seq - 3], dtype=torch.long)
    wrapper = MaskIndexWrapper(model).eval()
    mask_path = os.path.join(args.out, "model.onnx")
    t0 = time.time()
    export(wrapper, args, mask_path,
           ["input_ids", "attention_mask", "mask_index"],
           (ids, att, mask_index),
           {"input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "mask_index": {0: "batch"},
            "logits": {0: "batch", 1: "mask"}},
           ["logits"])
    print("exported mask-index fp32 in %.1fs" % (time.time() - t0), flush=True)
    quantize(mask_path, os.path.join(args.out, "model.int8.onnx"))

    if args.kv:
        export_prefix_cache(model, args.out, args)

    if os.path.exists(BASE_VOCAB):
        shutil.copy(BASE_VOCAB, os.path.join(args.out, "vocab.txt"))

    cfg = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "max_position": model.config.max_position_embeddings,
        "mask_token_id": tok.mask_token_id,
        "cls_token_id": tok.cls_token_id,
        "sep_token_id": tok.sep_token_id,
        "pad_token_id": tok.pad_token_id,
        "unk_token_id": tok.unk_token_id,
        "opset": args.opset,
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with io.open(os.path.join(args.out, "lm_config.json"), "w",
                 encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for name in ("model.onnx", "model.int8.onnx", "model_full.onnx",
                 "model_full.int8.onnx", "encode_prefix.onnx", "score_tail.onnx"):
        p = os.path.join(args.out, name)
        if os.path.exists(p):
            print("%-20s %.1f MB" % (name, os.path.getsize(p) / 1048576.0),
                  flush=True)
    print(json.dumps(cfg, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
