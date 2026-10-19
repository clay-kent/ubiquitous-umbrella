"""外接矩形を通った候補ペアの内訳を測る（intersection を減らせるか）。

`is_valid_layout` の費用はほぼ `intersection(...).area` で、実測 stairs では
37,704 ネットに対して 51,605 回（1.37/net）。うち

- 本当に重なっている（＝不正ネットの最初のペア）
- 接しているだけ（正しいネット。折り線以外の辺/頂点接触）
- 実は離れている（bbox だけ重なった）

のどれがどれだけあるかで「安い前判定を挟む価値」が決まる。

    uv run python tools/probe_valid_parts.py stairs
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

    seen = set()
    layouts = []
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
            layouts.append(
                (
                    fifth.layout_tree(v, faces, normals, tree, prep),
                    {tuple(sorted(e)) for e in tree},
                )
            )

    n = len(layouts)
    n_cand = 0
    n_disjoint = 0
    n_touch = 0
    n_real = 0
    t_inter = 0.0
    t_pred = 0.0
    t_all = time.process_time()
    for polys, folds in layouts:
        bnds = shapely.bounds(np.asarray(polys, dtype=object)).tolist()
        np_ = len(polys)
        for i in range(np_):
            ax0, ay0, ax1, ay1 = bnds[i]
            for j in range(i + 1, np_):
                if (i, j) in folds:
                    continue
                bx0, by0, bx1, by1 = bnds[j]
                if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                    continue
                dx = max(ax1, bx1) - min(ax0, bx0)
                dy = max(ay1, by1) - min(ay0, by0)
                thresh = fifth.OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox * oy <= thresh:
                    continue
                n_cand += 1
                t0 = time.process_time()
                inter = polys[i].intersects(polys[j])
                t_pred += time.process_time() - t0
                if not inter:
                    n_disjoint += 1
                    continue
                t0 = time.process_time()
                area = polys[i].intersection(polys[j]).area
                t_inter += time.process_time() - t0
                if area > thresh:
                    n_real += 1
                else:
                    n_touch += 1
    total = time.process_time() - t_all
    print(f"{name}: layouts={n} candidates={n_cand}")
    print(f"  disjoint (intersects=False) {n_disjoint:6d}"
          f"  {100 * n_disjoint / n_cand:5.1f}%")
    print(f"  touching (area<=thresh)     {n_touch:6d}"
          f"  {100 * n_touch / n_cand:5.1f}%")
    print(f"  real overlap (area>thresh)  {n_real:6d}"
          f"  {100 * n_real / n_cand:5.1f}%")
    print(f"  total_cpu={total:.3f}s  intersects={t_pred:.4f}s"
          f"  intersection={t_inter:.4f}s")


if __name__ == "__main__":
    main()
