"""Analyze whether wubi98 dict weights and luna_pinyin/essay frequencies can
be used as priors/features for reranker training, and quantify how much
headroom context-based reordering has.

Outputs a markdown report to priors_report.md.
"""

import io
import os
from collections import Counter, defaultdict

import zhconv

HERE = os.path.dirname(os.path.abspath(__file__))
WUBI = r"C:\Users\user\AppData\Roaming\Rime\wubi98.dict.yaml"
LUNA = r"D:\Workspace\rime\weasel\output\data\luna_pinyin.dict.yaml"
ESSAY = r"D:\Workspace\rime\weasel\output\data\essay.txt"
OUT = os.path.join(HERE, "priors_report.md")


def is_han(ch):
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
        (0xF900 <= o <= 0xFAFF)


def load_essay(path):
    freq = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    v = float(parts[1])
                except ValueError:
                    continue
                k = zhconv.convert(parts[0], "zh-cn")
                freq[k] = freq.get(k, 0.0) + v
    return freq


def load_dict(path):
    """Return list of (text, code, weight)."""
    entries = []
    started = False
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not started:
                if line.strip() == "...":
                    started = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0].strip():
                continue
            weight = 0.0
            if len(parts) > 2:
                try:
                    weight = float(parts[2])
                except ValueError:
                    weight = 0.0
            entries.append((parts[0].strip(), parts[1].strip(), weight))
    return entries


def bucket(ratio):
    if ratio >= 0.5:
        return ">=0.5"
    if ratio >= 0.2:
        return "0.20-0.50"
    if ratio >= 0.05:
        return "0.05-0.20"
    return "<0.05"


def analyze_codes(code2items, min_size=2):
    """code2items: code -> list of (text, weight). Returns stats dict."""
    amb = {c: sorted(v, key=lambda x: -x[1]) for c, v in code2items.items()
           if len(v) >= min_size}
    size_dist = Counter(len(v) for v in amb.values())
    ratio_dist = Counter()
    for items in amb.values():
        w1 = items[0][1]
        w2 = items[1][1]
        ratio = (w2 / w1) if w1 > 0 else 0.0
        ratio_dist[bucket(ratio)] += 1
    return amb, size_dist, ratio_dist


def main():
    lines = []
    freq = load_essay(ESSAY)
    single_freq = {w: v for w, v in freq.items()
                   if len(w) == 1 and is_han(w)}
    total_single = sum(single_freq.values())
    lines.append("# 先验可用性统计（wubi98 dict / luna_pinyin + essay）\n")
    lines.append("- essay 条目：**%d**，单字条目：**%d**，单字频率总和：%.0f\n"
                 % (len(freq), len(single_freq), total_single))

    # ---------- wubi98 ----------
    wubi = load_dict(WUBI)
    w_code_all = defaultdict(list)
    w_code_single = defaultdict(list)
    w_char_codes = defaultdict(list)
    for text, code, weight in wubi:
        simp = zhconv.convert(text, "zh-cn")
        w_code_all[code].append((text, weight))
        if len(simp) == 1 and is_han(simp):
            w_code_single[code].append((simp, weight))
            w_char_codes[simp].append((code, weight))

    amb_all, size_all, ratio_all = analyze_codes(w_code_all)
    amb_single, size_single, ratio_single = analyze_codes(w_code_single)
    lines.append("\n## 1. 五笔98词库（wubi98.dict.yaml，%d 条）\n" % len(wubi))
    lines.append("### 1.1 编码歧义规模\n")
    lines.append("| 口径 | 有候选的码 | ≥2 候选的码 | 最大候选数 |\n|---|---|---|---|\n")
    lines.append("| 全部词条 | %d | %d | %d |\n" %
                 (len(w_code_all), len(amb_all), max(size_all)))
    lines.append("| 仅单字 | %d | %d | %d |\n" %
                 (len(w_code_single), len(amb_single),
                  max(size_single) if size_single else 0))
    lines.append("\n单字歧义码候选数分布（候选数: 码数）：%s\n" %
                 ", ".join("%d:%d" % (k, v) for k, v in
                           sorted(size_single.items())[:12]))
    lines.append("\n### 1.2 先验不确定性（同一码内 top2/top1 权重比）\n")
    lines.append("| 比值 | 单字歧义码数 | 占比 |\n|---|---|---|\n")
    n_amb_single = sum(ratio_single.values())
    for k in (">=0.5", "0.20-0.50", "0.05-0.20", "<0.05"):
        c = ratio_single.get(k, 0)
        lines.append("| %s | %d | %.1f%% |\n" %
                     (k, c, 100.0 * c / max(1, n_amb_single)))

    # usage-weighted simulation: essay single-char frequencies as text proxy.
    # weight is split evenly across a character's codes to avoid double count.
    stat = Counter()
    flip_examples = []
    for ch, f in single_freq.items():
        codes = w_char_codes.get(ch)
        if not codes:
            continue
        share = f / len(codes)
        for code, weight in codes:
            items = amb_single.get(code)
            if not items:
                continue
            top1_w = items[0][1]
            rank = next((i for i, (t, _) in enumerate(items) if t == ch),
                        None)
            if rank is None:
                continue
            r = (weight / top1_w) if top1_w > 0 else 0.0
            stat["occ"] += share
            if rank == 0:
                stat["rank1"] += share
            else:
                stat["not_rank1"] += share
                stat["flip_" + bucket(r)] += share
            if rank > 0:
                flip_examples.append((share, ch, code, r))
    occ = max(1.0, stat["occ"])
    lines.append("\n### 1.3 用文本代理（essay 单字频率）加权的可翻盘空间\n")
    lines.append("- 覆盖到歧义码的单字出现权重占比：%.1f%%（其余为字符无码或码唯一）\n"
                 % (100.0 * stat["occ"] / total_single))
    lines.append("- 其中先验已排第 1：**%.1f%%**；先验未排第 1（理论上需要上下文翻盘）：**%.1f%%**\n"
                 % (100.0 * stat["rank1"] / occ,
                    100.0 * stat["not_rank1"] / occ))
    lines.append("- 未排第 1 的样本里，候选权重仍具竞争力（比值分桶）：\n")
    for k in (">=0.5", "0.20-0.50", "0.05-0.20", "<0.05"):
        lines.append("  - %s：%.1f%%\n" %
                     (k, 100.0 * stat.get("flip_" + k, 0) /
                      max(1.0, stat["not_rank1"])))
    flip_examples.sort(reverse=True)
    lines.append("- 高频示例（字/码/比值）：%s\n" %
                 "; ".join("%s/%s/%.2f" % (c, code, r)
                           for _, c, code, r in flip_examples[:10]))

    # ---------- luna_pinyin ----------
    luna = load_dict(LUNA)
    syl2chars = defaultdict(list)
    char2syl = defaultdict(set)
    for text, code, weight in luna:
        simp = zhconv.convert(text, "zh-cn")
        if len(simp) == 1 and is_han(simp):
            syl2chars[code].append((simp, freq.get(simp, 0.0)))
            char2syl[simp].add(code)
    amb_syl = {c: sorted(v, key=lambda x: -x[1])
               for c, v in syl2chars.items() if len(v) >= 2}
    ratio_syl = Counter()
    for items in amb_syl.values():
        w1 = items[0][1]
        w2 = items[1][1]
        ratio_syl[bucket(w2 / w1 if w1 > 0 else 0.0)] += 1
    lines.append("\n## 2. 拼音单字（luna_pinyin 词表 + essay 词频）\n")
    lines.append("- 单字音节码：%d，其中 ≥2 候选：**%d**（%.1f%%）\n" %
                 (len(syl2chars), len(amb_syl),
                  100.0 * len(amb_syl) / max(1, len(syl2chars))))
    sizes = sorted((len(v) for v in amb_syl.values()), reverse=True)
    lines.append("- 候选数：中位 %d，最大 %d，≥10 的音节占 %.1f%%\n" %
                 (sizes[len(sizes) // 2] if sizes else 0,
                  sizes[0] if sizes else 0,
                  100.0 * sum(1 for s in sizes if s >= 10) /
                  max(1, len(sizes))))
    lines.append("\n### 2.1 先验不确定性（音节内 top2/top1 频率比）\n")
    lines.append("| 比值 | 音节数 | 占比 |\n|---|---|---|\n")
    n = sum(ratio_syl.values())
    for k in (">=0.5", "0.20-0.50", "0.05-0.20", "<0.05"):
        c = ratio_syl.get(k, 0)
        lines.append("| %s | %d | %.1f%% |\n" % (k, c, 100.0 * c / max(1, n)))

    # usage-weighted pinyin single-char headroom
    stat = Counter()
    examples = []
    for ch, f in single_freq.items():
        codes = list(char2syl.get(ch, ()))
        if not codes:
            continue
        share = f / len(codes)
        for code in codes:
            items = amb_syl.get(code)
            if not items:
                continue
            top1_w = items[0][1]
            rank = next((i for i, (t, _) in enumerate(items) if t == ch),
                        None)
            if rank is None:
                continue
            r = (freq.get(ch, 0.0) / top1_w) if top1_w > 0 else 0.0
            stat["occ"] += share
            if rank == 0:
                stat["rank1"] += share
            else:
                stat["not_rank1"] += share
                stat["flip_" + bucket(r)] += share
            if rank > 0:
                examples.append((share, ch, code, r))
    occ = max(1.0, stat["occ"])
    lines.append("\n### 2.2 用文本代理加权的可翻盘空间（拼音单字）\n")
    lines.append("- 覆盖到歧义音节的单字出现权重：%.1f%%\n" %
                 (100.0 * stat["occ"] / total_single))
    lines.append("- 先验已排第 1：**%.1f%%**；未排第 1：**%.1f%%**\n" %
                 (100.0 * stat["rank1"] / occ,
                  100.0 * stat["not_rank1"] / occ))
    for k in (">=0.5", "0.20-0.50", "0.05-0.20", "<0.05"):
        lines.append("  - %s：%.1f%%\n" %
                     (k, 100.0 * stat.get("flip_" + k, 0) /
                      max(1.0, stat["not_rank1"])))
    examples.sort(reverse=True)
    lines.append("- 高频示例（字/音节/比值）：%s\n" %
                 "; ".join("%s/%s/%.2f" % (c, code, r)
                           for _, c, code, r in examples[:10]))

    with io.open(OUT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("written:", OUT, flush=True)


if __name__ == "__main__":
    main()
