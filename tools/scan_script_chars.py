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


def classify(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, name in SUSPICIOUS_RANGES:
        if lo <= cp <= hi:
            return name
    return None


def main() -> int:
    bad = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(s in path.parts for s in SKIP):
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
