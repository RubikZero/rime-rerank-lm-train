"""Print the I/O contract of the exported ONNX models, and check at runtime that
the mask-index graph really returns logits for a single position.

Usage: python check_export_io.py <model.onnx> [more.onnx ...]
"""
import hashlib
import os
import sys

import numpy as np
import onnxruntime as ort


def describe(path):
    print("=====", path)
    if not os.path.exists(path):
        print("  missing")
        return
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print("  size=%d bytes  sha256=%s" % (os.path.getsize(path), digest[:24]))
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    for i in sess.get_inputs():
        print("  in : %-16s %-12s %s" % (i.name, i.type, i.shape))
    for o in sess.get_outputs():
        print("  out: %-16s %-12s %s" % (o.name, o.type, o.shape))

    # Run one batch with an explicit mask position and report the real shapes.
    seq = 64
    feed = {
        "input_ids": np.ones((1, seq), dtype=np.int64),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
    }
    if "mask_index" in inputs:
        feed["mask_index"] = np.array([seq - 3], dtype=np.int64)
    out = sess.run(None, feed)[0]
    print("  runtime: %s -> %s (dtype %s)" % (seq, out.shape, out.dtype))
    if "mask_index" in inputs:
        print("  single-position graph: output has %d position(s)" % (out.shape[1],))
    else:
        print("  all-position graph: every position scored (not used online)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        describe(arg)
