#!/usr/bin/env python
"""pre-commit フック: requirements.txt が pyproject.toml とずれていたら更新する。

`.git/hooks/pre-commit` に置くと、commit 前に自動で走る
（`.pre-commit-config.yaml` の導入はしない。小規模プロジェクトなので）。

インストール:

    python tools/install_hook.py

これは git のローカル設定だけなのでリポジトリには入らない。
CI でずれを検出したいときは `--check`（ずれていれば exit 1）を使う。
"""

import subprocess
import sys
from pathlib import Path

HOOK = """#!/bin/sh
# tools/install_hook.py が生成した。requirements.txt を pyproject.toml に同期する。
# ずれていればその場で書き換えてから commit を進める（ずれていなければ何もしない）。
#
# 注意: git はフックを sh で実行する。Windows パスでも壊れないよう
# 前方スラッシュに直し、空白を含むため必ずクォートする。
set -e
exec "{python}" "{script}"
"""


def sh_path(p: Path) -> str:
    """sh のクォート内で安全に書けるよう前方スラッシュへ変換する。"""
    return p.resolve().as_posix()


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    git_dir = Path(subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir

    hooks = git_dir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    path = hooks / "pre-commit"

    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "sync_requirements" in text:
            print("既に同期フックがあるので上書きはしない: {0}".format(path))
            print("再生成するには .git/hooks/pre-commit を消してから実行してください")
            return 0
        print("注意: 既存の pre-commit フックを壊さないよう触りません。", file=sys.stderr)
        print("      中身を確認して、必要なら手で追記してください: {0}".format(path))
        return 1

    script = repo / "tools" / "sync_requirements.py"
    path.write_text(
        HOOK.format(python=sh_path(Path(sys.executable)), script=sh_path(script)),
        encoding="utf-8",
    )
    print("インストールした: {0}".format(path))
    print("コミット時に {0} が自動同期されます".format(script.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())