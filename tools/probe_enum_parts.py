"""列挙の内訳: `_is_connected` の呼び出し回数と時間、1 本あたりの再帰量。

enumerate_spanning_trees は 1 本 7us 級。うち `_is_connected`（rest から
(a,b) のリストを作り直して BFS）がどれだけ占めるかを見る。
`_is_connected` はモジュールグローバルとして解決されるので差し替えられる。

    uv run python tools/probe_enum_parts.py stairs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fifth
from bench_ladder import SOLIDS

STATS = {"calls": 0, "time": 0.0, "edges": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)

    real = fifth._is_connected_edges

    def wrapped(e, n):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        r = real(e, n)
        STATS["time"] += time.perf_counter() - t0
        STATS["calls"] += 1
        STATS["edges"] += len(e)
        return r

    fifth._is_connected_edges = wrapped
    t0 = time.perf_counter()
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    total = time.perf_counter() - t0
    fifth._is_connected_edges = real

    n = len(trees)
    print(f"{name}: trees={n} total={total:.3f}s ({total / n * 1e6:.2f} us/tree)")
    print(f"  _is_connected calls={STATS['calls']} ({STATS['calls'] / n:.2f}/tree) "
          f"time={STATS['time']:.3f}s ({STATS['time'] / total * 100:.1f}%)")
    print(f"  平均入力辺数={STATS['edges'] / max(STATS['calls'], 1):.1f}")
    print(f"  残り（再帰本体・出力構築）={total - STATS['time']:.3f}s "
          f"({(total - STATS['time']) / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
