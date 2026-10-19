"""列挙のコストのうち「木を Python のリスト+タプルで作る」分を切り分ける。

enumerate_spanning_trees は 1 本あたり (F-1) 個の tuple を持つ list を yield する。
stairs は 150,528 本（= 135 万個の tuple）なので、オブジェクト生成そのものが
効いている可能性がある。同じ値を合成で作り、生成だけの時間を測る。

    uv run python tools/probe_enum_objects.py stairs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)

    t0 = time.perf_counter()
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    t_enum = time.perf_counter() - t0
    n = len(trees)
    n_edges = len(trees[0])
    print(f"{name}: trees={n} edges/tree={n_edges}")
    print(f"  列挙（生成込み）      {t_enum:7.3f}s   ({t_enum / n * 1e6:6.2f} us/tree)")

    # 合成: 同じ個数を、同じ形で作り直す（値は適当）。
    t0 = time.perf_counter()
    out = []
    for i in range(n):
        out.append([(i & 7, (i + 1) & 7) for _ in range(n_edges)])
    t_build = time.perf_counter() - t0
    print(f"  合成の list+tuple 生成  {t_build:7.3f}s   ({t_build / n * 1e6:6.2f} us/tree)")
    del out

    # 合成: 1 本ずつ tuple を作らず、既存の tuple を共有する場合。
    shared = [(0, 1)] * n_edges
    t0 = time.perf_counter()
    out2 = []
    for _i in range(n):
        out2.append(list(shared))
    t_share = time.perf_counter() - t0
    print(f"  合成の list のみ生成    {t_share:7.3f}s   ({t_share / n * 1e6:6.2f} us/tree)")
    del out2, shared


if __name__ == "__main__":
    main()
