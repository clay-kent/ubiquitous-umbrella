"""`_layout_batch` の内部位相ごとの時間（本番コードの写しに計時点を入れたもの）。

`_layout_batch` は「木ごとの親子決定 → 回転伝播 → 投影 → GEOS」で構成される。
どの位相が効いているかを知るために、本体をそのまま写して区間を計る。
写しが本番と一致することは冒頭で確認する（一致しなければ結果を信用しない）。

    uv run python tools/probe_layout_phases.py stairs
"""

import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

PH = (
    "par_child",
    "propagate",
    "axes",
    "project",
    "pack",
    "linearrings",
    "polygons",
    "bounds",
    "slice",
)
ST = {k: 0.0 for k in PH}
ST["rows"] = 0
ST["batches"] = 0


def _tick() -> float:
    return time.perf_counter()


def traced(
    n_faces: int,
    trees: list[list[tuple[int, int]]],
    precomputed: fifth.Layout,
) -> tuple[list[Any], np.ndarray]:
    n_batch = len(trees)
    if n_batch == 0:
        return [], np.empty((0, n_faces, 4), dtype=np.float64)
    n_edges = n_faces - 1

    t0 = _tick()
    par = np.empty((n_batch, n_edges), dtype=np.int64)
    child = np.empty((n_batch, n_edges), dtype=np.int64)
    for b, tree in enumerate(trees):
        _parents, pairs = fifth._tree_bfs(n_faces, tree)
        par[b] = [u for u, _v in pairs]
        child[b] = [v for _u, v in pairs]
    t1 = _tick()

    rot_tab = precomputed["fold_r_tab"]
    off_tab = precomputed["fold_w_tab"]
    rr = rot_tab[par, child]
    ww = off_tab[par, child]

    rot = np.empty((n_batch, n_faces, 3, 3))
    off = np.empty((n_batch, n_faces, 3))
    rot[:, 0] = fifth._EYE3
    off[:, 0] = 0.0
    rows = np.arange(n_batch)
    for k in range(n_edges):
        u = par[:, k]
        v = child[:, k]
        ru = rot[rows, u]
        rot[rows, v] = np.matmul(ru, rr[:, k])
        off[rows, v] = np.matmul(ru, ww[:, k, :, None])[:, :, 0] + off[rows, u]
    t2 = _tick()

    u_x = precomputed["u_x"]
    u_y = precomputed["u_y"]
    origin = precomputed["origin"]
    axes_x = np.matmul(u_x, rot)
    axes_y = np.matmul(u_y, rot)
    shift = off - origin
    cs_x = (shift * u_x).sum(axis=2)
    cs_y = (shift * u_y).sum(axis=2)
    t3 = _tick()

    idx_rows = precomputed["flat_indices"]
    P = precomputed["flat_pts"]
    AX = axes_x[:, idx_rows]
    AY = axes_y[:, idx_rows]
    xs = (
        P[None, :, 0] * AX[:, :, 0]
        + P[None, :, 1] * AX[:, :, 1]
        + P[None, :, 2] * AX[:, :, 2]
        + cs_x[:, idx_rows]
    )
    ys = (
        P[None, :, 0] * AY[:, :, 0]
        + P[None, :, 1] * AY[:, :, 1]
        + P[None, :, 2] * AY[:, :, 2]
        + cs_y[:, idx_rows]
    )
    t4 = _tick()

    flat = np.column_stack([xs.ravel(), ys.ravel()])
    ring_idx = (np.arange(n_batch)[:, None] * n_faces + idx_rows[None, :]).ravel()
    t5 = _tick()

    rings = shapely.linearrings(flat, indices=ring_idx)
    t6 = _tick()
    built = cast(list[fifth.Polygon], shapely.polygons(rings))
    t7 = _tick()
    bnds = np.asarray(shapely.bounds(built), dtype=np.float64).reshape(n_batch, n_faces, 4)
    t8 = _tick()
    polys = [built[b * n_faces : (b + 1) * n_faces] for b in range(n_batch)]
    t9 = _tick()

    ST["par_child"] += t1 - t0
    ST["propagate"] += t2 - t1
    ST["axes"] += t3 - t2
    ST["project"] += t4 - t3
    ST["pack"] += t5 - t4
    ST["linearrings"] += t6 - t5
    ST["polygons"] += t7 - t6
    ST["bounds"] += t8 - t7
    ST["slice"] += t9 - t8
    ST["rows"] += n_batch
    ST["batches"] += 1
    return polys, bnds


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    n_window = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    prep = fifth.prepare_layout(v, faces, normals)

    trees = []
    for tree in fifth.enumerate_spanning_trees(edges, len(faces)):
        trees.append(tree)
        if len(trees) >= n_window * fifth._LADDER_WINDOW:
            break

    # 写しが本番と一致することの確認（最初の窓だけ）。
    w0 = trees[: fifth._LADDER_WINDOW]
    ref = fifth._layout_batch(len(faces), w0, prep)
    ref_polys, ref_b = ref[0], ref[1]
    got, got_b = traced(len(faces), w0, prep)[:2]
    same_b = np.array_equal(ref_b, got_b)
    same_p = all(
        np.array_equal(shapely.to_wkb(a), shapely.to_wkb(b))
        for a, b in zip(ref_polys, got)
    )
    if not (same_b and same_p):
        raise SystemExit(f"写しが本番と不一致 (bounds={same_b}, polys={same_p})")

    for k in PH:
        ST[k] = 0.0
    ST["rows"] = 0
    ST["batches"] = 0

    t0 = _tick()
    for i in range(0, len(trees), fifth._LADDER_WINDOW):
        traced(len(faces), trees[i : i + fifth._LADDER_WINDOW], prep)
    total = _tick() - t0

    print(f"{name}: rows={ST['rows']} batches={ST['batches']} total={total:.3f}s "
          f"({total / ST['rows'] * 1e6:.2f} us/net)")
    for k in PH:
        print(f"  {k:12s} {ST[k]:7.3f}s ({ST[k] / total * 100:5.1f}%, "
              f"{ST[k] / ST['rows'] * 1e6:6.2f} us/net)")


if __name__ == "__main__":
    main()
