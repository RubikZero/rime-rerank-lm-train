# Compares the visible candidate menus of two IPC capture files block by block.
#
# Each response from WeaselServer starts with "action=" and carries the serialized
# candidate queue in "ctx.cand=".  The number of responses differs between runs
# (the client polls with a Shift probe until the LM answer arrives), so the files
# are split into blocks and compared as multisets of menus rather than as text.
import sys
from collections import Counter


def blocks(path):
    out, cur = [], []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("action=") and cur:
                out.append(cur)
                cur = []
            cur.append(line)
    if cur:
        out.append(cur)
    return out


def menus(path):
    result = []
    for block in blocks(path):
        for line in block:
            if line.startswith("ctx.cand="):
                result.append(line)
    return result


def final(path):
    for block in reversed(blocks(path)):
        for line in block:
            if line.startswith("ctx.cand="):
                return line
    return ""


def marks(path):
    return sum(line.count("[LM]") for line in menus(path))


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_menus, b_menus = menus(a_path), menus(b_path)
    a_final, b_final = final(a_path), final(b_path)
    print(f"  menus: {len(a_menus)} vs {len(b_menus)}")
    print(f"  distinct menus: {len(set(a_menus))} vs {len(set(b_menus))}")
    print(f"  [LM] marks in menus: {marks(a_path)} vs {marks(b_path)}")
    print(f"  final menu identical: {a_final == b_final}")
    if set(a_menus) != set(b_menus):
        only_a = Counter(a_menus) - Counter(b_menus)
        only_b = Counter(b_menus) - Counter(a_menus)
        for label, diff in (("int8-only", only_a), ("fp32-only", only_b)):
            for line, count in list(diff.items())[:4]:
                print(f"  {label} (x{count}): {line[:150]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
