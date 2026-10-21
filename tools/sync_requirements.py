# -*- coding: utf-8 -*-
"""pyproject.toml から requirements.txt を生成する。

pyproject.toml と uv.lock が正であり、requirements.txt は pip 向けの
派生ファイル。手で編集すると必ずずれるので、pre-commit で自動同期する。

    python tools/sync_requirements.py          # 書き換える
    python tools/sync_requirements.py --check  # ずれていれば exit 1
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

HEADER = """\
# 自動生成 — 手編集しないこと。
# 正となるのは pyproject.toml / uv.lock で、ここは pip install -r 用の派生。
# tools/sync_requirements.py が生成する（pre-commit で自動同期）。
"""


def parse_requirement(spec: str) -> tuple[str, str]:
    """'numpy>=2.5.0' -> ('numpy', '>=2.5.0')"""
    m = re.match(r"^([A-Za-z0-9._-]+)(.*)$", spec.strip())
    if not m:
        raise ValueError(f"解釈できない依存指定: {spec!r}")
    return m.group(1), m.group(2).strip()


def render(pyproject_text: str, *, include_dev: bool) -> str:
    data = tomllib.loads(pyproject_text)
    project = data["project"]
    sections: list[tuple[str, list[str]]] = [
        ("runtime", list(project.get("dependencies", []))),
    ]
    if include_dev:
        sections.append(("dev", list(data.get("dependency-groups", {}).get("dev", []))))

    lines = [HEADER, "# --- runtime ---"]
    lines += [f"{parse_requirement(s)[0]}{parse_requirement(s)[1]}" for s in sections[0][1]]
    if include_dev:
        lines += ["", "# --- dev ---"]
        lines += [f"{parse_requirement(s)[0]}{parse_requirement(s)[1]}" for s in sections[1][1]]
    return "\n".join(lines) + "\n"


def main() -> int:
    include_dev = "--runtime-only" not in sys.argv
    check = "--check" in sys.argv
    wanted = render(PYPROJECT.read_text(encoding="utf-8"), include_dev=include_dev)
    current = REQUIREMENTS.read_text(encoding="utf-8") if REQUIREMENTS.exists() else ""

    if check:
        if current != wanted:
            print("requirements.txt が pyproject.toml とずれている。", file=sys.stderr)
            print("  python tools/sync_requirements.py", file=sys.stderr)
            return 1
        print("requirements.txt は最新")
        return 0

    if current == wanted:
        print("requirements.txt は既に最新")
        return 0

    REQUIREMENTS.write_text(wanted, encoding="utf-8")
    print("requirements.txt を書き換えた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())