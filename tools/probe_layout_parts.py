"""展開（`_layout_batch`）の内訳: 何本を展開し、`_tree_bfs` がどれだけ占めるか。

`_tree_bfs` はモジュールグローバルとして呼ばれるので、差し替えて合計時間を
測れる（ラッパの固定費は 1 呼び出し 1us 弱で、本数ぶんしか乗らない）。

    uv run python tools/probe_layout_parts.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS

ST = {"trees": 0, "bfs_time": 0.0, "bfs_calls": 0, "layout_time": 0.0,
      "batches": 0, "batch_rows": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)

    real_bfs = fifth._tree_bfs
    real_batch = fifth._layout_batch

    def bfs(n_faces, tree_edges):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        r = real_bfs(n_faces, tree_edges)
        ST["bfs_time"] += time.perf_counter() - t0
        ST["bfs_calls"] += 1
        return r

    def batch(n_faces, trees, precomputed):  # type: ignore[no-untyped-def]
        ST["batches"] += 1
        ST["batch_rows"] += len(trees)
        t0 = time.perf_counter()
        r = real_batch(n_faces, trees, precomputed)
        ST["layout_time"] += time.perf_counter() - t0
        return r

    fifth._tree_bfs = bfs
    fifth._layout_batch = batch
    n = 0
    t0 = time.perf_counter()
    for _t, _p in fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals):
        n += 1
    total = time.perf_counter() - t0
    fifth._tree_bfs = real_bfs
    fifth._layout_batch = real_batch

    print(f"{name}: nets={n} pipeline={total:.3f}s")
    print(f"  _layout_batch calls={ST['batches']} rows={ST['batch_rows']} "
          f"time={ST['layout_time']:.3f}s ({ST['layout_time'] / total * 100:.1f}%)")
    print(f"  _tree_bfs calls={ST['bfs_calls']} time={ST['bfs_time']:.3f}s "
          f"({ST['bfs_time'] / total * 100:.1f}% of pipeline, "
          f"{ST['bfs_time'] / max(ST['batch_rows'], 1) * 1e6:.2f} us/tree)")
    print(f"  残り（ベクトル展開 + GEOS）={ST['layout_time'] - ST['bfs_time']:.3f}s")


if __name__ == "__main__":
    main()
