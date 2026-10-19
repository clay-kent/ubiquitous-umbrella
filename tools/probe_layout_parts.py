"""`layout_tree` の内部を部位別に測る（CPU 時間、全一意ネット合計）。

- BFS（木の探索）
- rot/off の伝播（3x3 行列積が辺数ぶん）
- 軸 eizum + 平行移動定数
- 投影（成分ごとの積和）
- Polygon 生成

    uv run python tools/probe_layout_parts.py stairs
"""

import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

_PARTS = ("bfs", "propagate", "axes", "project", "polygons")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    prep = fifth.prepare_layout(v, faces, normals)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    # 一意な木だけ集める（シグネチャで重複を落とす）
    seen = set()
    unique = []
    rest = iter(trees)
    while True:
        window = list(islice(rest, fifth._LADDER_WINDOW))
        if not window:
            break
        for tree, (a, b) in zip(window, fifth._batch_canonical_keys(ctx, mirror_ctx, window)):
            k = min(a, b)
            if k in seen:
                continue
            seen.add(k)
            unique.append(tree)

    fold_of = prep["fold_of"]
    n_faces = len(faces)
    origin = prep["origin"]
    u_x = prep["u_x"]
    u_y = prep["u_y"]
    idx_rows = prep["flat_indices"]
    P = prep["flat_pts"]

    clocks = dict.fromkeys(_PARTS, 0.0)
    t_all = time.process_time()
    for tree in unique:
        t0 = time.process_time()
        _parents, pairs = fifth._tree_bfs(n_faces, tree)
        t1 = time.process_time()
        clocks["bfs"] += t1 - t0

        rot = np.empty((n_faces, 3, 3))
        off = np.empty((n_faces, 3))
        rot[0] = np.eye(3)
        off[0] = 0.0
        for u, vv in pairs:
            r, w = fold_of[(u, vv)]
            rot[vv] = rot[u] @ r
            off[vv] = rot[u] @ w + off[u]
        t2 = time.process_time()
        clocks["propagate"] += t2 - t1

        axes_x = np.einsum("b,fba->fa", u_x, rot)
        axes_y = np.einsum("b,fba->fa", u_y, rot)
        shift = off - origin
        cs_x = (shift * u_x).sum(axis=1)
        cs_y = (shift * u_y).sum(axis=1)
        t3 = time.process_time()
        clocks["axes"] += t3 - t2

        AX = axes_x[idx_rows]
        AY = axes_y[idx_rows]
        xs = P[:, 0] * AX[:, 0] + P[:, 1] * AX[:, 1] + P[:, 2] * AX[:, 2] + cs_x[idx_rows]
        ys = P[:, 0] * AY[:, 0] + P[:, 1] * AY[:, 1] + P[:, 2] * AY[:, 2] + cs_y[idx_rows]
        flat = np.column_stack([xs, ys])
        t4 = time.process_time()
        clocks["project"] += t4 - t3

        shapely.polygons(shapely.linearrings(flat, indices=idx_rows))
        t5 = time.process_time()
        clocks["polygons"] += t5 - t4

    total = time.process_time() - t_all
    print(f"{name}: unique={len(unique)} layouts total_cpu={total:.3f}s"
          f"  ({1e6 * total / len(unique):.1f} us/net)")
    for k in _PARTS:
        print(f"  {k:10s} {clocks[k]:8.4f}s {100 * clocks[k] / total:5.1f}%")


if __name__ == "__main__":
    main()
