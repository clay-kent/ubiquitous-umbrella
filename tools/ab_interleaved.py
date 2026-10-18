"""同一セッション内の A/B 比較（絶対値は他セッションと比較できない）。

この環境では外部プロセス（PxiTpService, Zed など）が CPU を使い続けるため、
絶対値が大きく変わる（実測: prism7 が 2.8s → 13.8s に変動）。
**同一セッション内で交互に走らせて比べる** ことで、ロードの影響を相殺する。

⚠️ ただしこのスクリプトは同じコードを2回測るだけなので、
git の状態を切り替えながら使うには不十分。実際の前後比較には
``git stash`` と :mod:`bench_layout_only` / :mod:`bench_enum_only` を使う。
"""

import statistics
import subprocess
import sys

BENCH = "tools/timing_parts.py"


def run(solid: str) -> tuple[float, float]:
    """(total秒, enumerate秒) を返す。"""
    out = subprocess.run(
        [sys.executable, BENCH, solid],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    total = enum = 0.0
    for line in out.stdout.splitlines():
        if "total" in line:
            total = float(line.split(":")[1].strip().rstrip("s"))
        if "enumerate" in line:
            enum = float(line.split(":")[1].strip().rstrip("s"))
    return total, enum


def main():
    solid = sys.argv[1] if len(sys.argv) > 1 else "prism7"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with_change, without_change = [], []
    for _ in range(rounds):
        with_change.append(run(solid))
        without_change.append(run(solid))
    a = statistics.median(x[0] for x in with_change)
    b = statistics.median(x[0] for x in without_change)
    ae = statistics.median(x[1] for x in with_change)
    be = statistics.median(x[1] for x in without_change)
    print(f"{solid}: interleaved A/B, n={rounds} each")
    print(f"  WITH change    : total={a:7.2f}s  enumerate={ae:6.3f}s")
    print(f"  WITHOUT change : total={b:7.2f}s  enumerate={be:6.3f}s")
    if be > 0:
        print(f"  enumerate ratio (with/without) = {ae / be:.2f}")


if __name__ == "__main__":
    main()