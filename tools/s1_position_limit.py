"""What the shipped graphs do at the position limit (512) and just past it.

The plugin's clamps are meant to keep every row below the model's 512 trained
positions, so this only answers "what would happen if that were bypassed": is it
a hard error, a silent clamp, or garbage?
"""
import numpy as np
import onnxruntime as ort

MODEL = r"C:\Users\user\AppData\Roaming\Rime\lm-mlm\model.onnx"
ENCODE = r"C:\Users\user\AppData\Roaming\Rime\lm-mlm\encode_prefix.onnx"

so = ort.SessionOptions()
so.intra_op_num_threads = 2
sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
print("model inputs:", [(i.name, i.shape) for i in sess.get_inputs()])

for length in (256, 512, 513, 520):
    ids = np.ones((1, length), dtype=np.int64) * 5      # any in-vocab id
    att = np.ones((1, length), dtype=np.int64)
    mask_index = np.array([length - 1], dtype=np.int64)  # gather at the last slot
    try:
        out = sess.run(None, {"input_ids": ids, "attention_mask": att,
                              "mask_index": mask_index})
        logits = out[0]
        print(f"  len={length:4d}: ok, logits{logits.shape}, "
              f"finite={np.isfinite(logits).all()}")
    except Exception as exc:                             # noqa: BLE001
        print(f"  len={length:4d}: FAILED -> {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}")

enc = ort.InferenceSession(ENCODE, so, providers=["CPUExecutionProvider"])
print("\nencode_prefix inputs:", [(i.name, i.shape) for i in enc.get_inputs()])
for length in (480, 512, 513):
    ids = np.ones((1, length), dtype=np.int64) * 5
    att = np.ones((1, length), dtype=np.int64)
    pos = np.arange(length, dtype=np.int64)
    try:
        out = enc.run(None, {"input_ids": ids, "attention_mask": att,
                             "position_ids": pos})
        print(f"  len={length:4d}: ok, {len(out)} tensors, "
              f"first{out[0].shape}")
    except Exception as exc:                             # noqa: BLE001
        print(f"  len={length:4d}: FAILED -> {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}")
