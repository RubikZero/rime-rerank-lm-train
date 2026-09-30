"""Point both lm_ranker.yaml files at the larger context window.

The plugin now accepts up to 480 (the model was trained with 512 positions and a
decode row is [CLS] + window + one token per syllable + [MASK] [SEP]), and 384 is
the new shipped default.
"""
import io

TARGETS = [
    r"D:\Workspace\rime\third-party\lm-test\runtime\user\lm_ranker.yaml",
    r"C:\Users\user\AppData\Roaming\Rime\lm_ranker.yaml",
]

PAIRS = [
    ("max_context: 256           # 上下文窗口上限（8~256，自适应窗口不会超过它）",
     "max_context: 384           # 上下文窗口上限（32~480，自适应窗口不会超过它；\n"
     "                           # 模型位置数为 512，解码行 = 1 + 窗口 + 音节数(≤16) + 2，故上限 480）"),
    ("（都填 256 即旧行为）", "（都填 384 即固定窗口）"),
]

for path in TARGETS:
    text = io.open(path, encoding="utf-8").read()
    changed = False
    for old, new in PAIRS:
        if old in text:
            text = text.replace(old, new, 1)
            changed = True
        else:
            print("  MISS in %s: %r" % (path, old[:32]))
    if changed:
        io.open(path, "w", encoding="utf-8", newline="").write(text)
    print("  %s -> %s" % ("patched" if changed else "unchanged", path))
