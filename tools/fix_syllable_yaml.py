"""Add sentence.max_syllables next to min_syllables in both lm_ranker.yaml files.

The user's own config gets 16 (they hit the silent 12 syllable cut-off); the
isolated test runtime keeps the shipped default of 12 so tests exercise what a
fresh install does.
"""
import io
import re

TARGETS = [
    (r"D:\Workspace\rime\third-party\lm-test\runtime\user\lm_ranker.yaml", "12"),
    (r"C:\Users\user\AppData\Roaming\Rime\lm_ranker.yaml", "16"),
]

COMMENT = ("    max_syllables: {value}       "
           "# 触发整句生成的最大音节数（min_syllables~16）：\n"
           "                             # 超过它就不再生成整句（解码行 = 1 + 窗口 + 音节数 + 2，\n"
           "                             # 模型位置数为 512，所以 16 是硬上限）\n")

for path, value in TARGETS:
    text = io.open(path, encoding="utf-8").read()
    if re.search(r"(?m)^\s*max_syllables:", text):
        text = re.sub(r"(?m)^\s*max_syllables:.*\n(\s*#.*\n)*", "", text)
        print("  replaced existing key in %s" % path)
    match = re.search(r"(?m)^(\s*min_syllables:[^\n]*\n)", text)
    if not match:
        print("  MISS min_syllables in %s" % path)
        continue
    insert = match.group(1) + COMMENT.format(value=value)
    text = text[:match.start()] + insert + text[match.end():]
    io.open(path, "w", encoding="utf-8", newline="").write(text)
    print("  patched %s (max_syllables=%s)" % (path, value))
