"""ノイズを排除するための反復計測（中央値とIQRを出す）。

この環境の計測は ±20% ばらつく（同一コードで 37s〜46s 実測）ので、
1回の計測で「改善した」と結論してはいけない。最低3回測って中央値で比べる。
"""

import statistics
import subprocess
import sys


def run(n: int, solid: str) -> list[float]:
    vals = []
    for _ in range(n):
        out = subprocess.run(
            [sys.executable, "tools/timing_parts.py", solid],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in out.stdout.splitlines():
            if "total" in line:
                vals.append(float(line.split(":")[1].strip().rstrip("s")))
    return vals


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    solid = sys.argv[2] if len(sys.argv) > 2 else "stairs"
    v = run(n, solid)
    v.sort()
    print(f"{solid}: n={len(v)} min={v[0]:.2f}s median={statistics.median(v):.2f}s max={v[-1]:.2f}s")
    print(f"  spread = {100*(v[-1]-v[0])/statistics.median(v):.0f}% of median")