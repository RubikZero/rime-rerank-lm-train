import time

import torch
from transformers import AutoModelForMaskedLM

BASE = r"D:\Workspace\rime\third-party\models\uer\chinese_roberta_L-4_H-256"
ID = torch.randint(0, 21128, (64, 72))
MASK = torch.ones_like(ID)
LABELS = torch.full_like(ID, -100)
LABELS[:, 60:64] = 123


def run(device, steps):
    model = AutoModelForMaskedLM.from_pretrained(BASE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ids, mask, labels = ID.to(device), MASK.to(device), LABELS.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    extra = ""
    if device == "cuda":
        extra = " peak_gpu_mem=%.1fMB" % (
            torch.cuda.max_memory_allocated() / 2 ** 20)
    return "%-5s steps=%d %.0f tok/s device=%s%s" % (
        device, steps, steps * 64 * 72 / dt,
        next(model.parameters()).device, extra)


print("torch", torch.__version__, "cuda", torch.version.cuda,
      "avail", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
print(run("cuda", 100), flush=True)
print(run("cpu", 20), flush=True)
