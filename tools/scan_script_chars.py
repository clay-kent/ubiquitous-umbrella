"""Markdown / Python に混入した「別言語の文字」を洗い出す。

手編集でアラビア文字・キリル文字・ハングル・ポルトガル語などが
日本語文に紛れたことがあるため、**日本語・英語で使わない文字**だけを
検出する（①②③・§・±・⚠・×・÷ などの記号は正常として無視）。

使い方: uv run python tools/scan_script_chars.py
"""

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = (".venv", ".git", "__pycache__", ".mypy_cache")

# 「明らかに別言語」のスクリプト・記号の範囲だけを疑う。
SUSPICIOUS_RANGES = [
    (0x0400, 0x04FF, "キリル文字"),
    (0x0500, 0x052F, "キリル文字補助"),
    (0x0530, 0x058F, "アルメニア文字"),
    (0x0590, 0x05FF, "ヘブライ文字"),
    (0x0600, 0x06FF, "アラビア文字"),
    (0x0700, 0x074F, "シリア文字"),
    (0x0370, 0x03FF, "ギリシャ文字"),
    (0x0E00, 0x0E7F, "タイ文字"),
    (0xAC00, 0xD7AF, "ハングル"),
    (0x1100, 0x11FF, "ハングル字母"),
    (0x0100, 0x017F, "ラテン拡張A（ポルトガル語等）"),
    (0xFFFD, 0xFFFD, "置換文字(U+FFFD)"),
]

# 簡体字（中国語）専用の文字。日本語のかな・漢字も CJK 統合漢字なので
# 範囲では区別できないため、混入しやすい簡体字を名指しで疑う。
# 日本語でも使う字（会・后・里・面・争・体・余・虫 など）は**入れない**。
SIMPLIFIED_ONLY = (
    "杂实诚关见说这们时过为从无长门问间单车东乐书带应该语读听认记试验场"
    "边达运动员师课题级组结构简顶线变图标检识议观词讲谢证让请谁调谈归还"
    "进远连迟适选铺链铁错钱锁钟镜闭闻队阳险隐难骑响际复处务历叶岁业个丰"
    "严丧举义乌鸦乔习乡买亏亚产亩亲亿仅仓仪价众优伟传伤伦伪伞侦侧侨丛丝"
    "临丽览户仅侨"
)


def classify(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, name in SUSPICIOUS_RANGES:
        if lo <= cp <= hi:
            return name
    if ch in SIMPLIFIED_ONLY:
        return "簡体字（中国語）"
    return None


def main() -> int:
    bad = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(s in path.parts for s in SKIP):
            continue
        # このスクリプト自身は簡体字の一覧を持っているので除外する。
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in (".py", ".md", ".json", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for col, ch in enumerate(line, 1):
                name = classify(ch)
                if name:
                    uname = unicodedata.name(ch, "?")
                    print(
                        f"{path.relative_to(ROOT)}:{lineno}:{col} "
                        f"U+{ord(ch):04X} {name} {uname!r} "
                        f"in: {line.strip()[:60]}"
                    )
                    bad += 1
    print(f"scan done, suspicious={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
