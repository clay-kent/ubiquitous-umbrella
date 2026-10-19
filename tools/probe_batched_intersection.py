"""交差計算を「ネットごとに逐次・早期終了」で回す版と「窓内で一括」の版を比べる。

`_valid_batch` の候補ペアは stairs で 124,039、そのうち実際に交差計算まで
進むのは 51,605（58% は早期終了で省かれている）。一方で 1 回の
``polys[i].intersection(polys[j])`` は Python/GEOS の固定費が乗る。

判定は「候補ペアのどれか 1 つでも重なれば不正」なので、**順序は結果に効かない**
（早期終了は速さのためだけ）。つまり候補を全部まとめて交差計算してから
ネットごとに any を取っても判定はビット単位で同じ。

ここでは同じ窓データに対して両方を交互に走らせ、時間と判定の一致を数える。

    uv run python tools/probe_batched_intersection.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

ACC = {"loop": 0.0, "batch": 0.0}
CNT = {"candidates": 0, "nets": 0, "mismatch": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    pairs, pair_idx = fifth.pair_table(len(faces))
    pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
    pair_j = np.array([j for _, j in pairs], dtype=np.intp)

    def compare(polys_list, bnds, trees_arr):  # type: ignore[no-untyped-def]
        n_batch = len(polys_list)
        n_faces = bnds.shape[1]
        if n_batch == 0 or n_faces < 2:
            return [True] * n_batch
        # 候補ペアを先に共通で作る（絞り込みの時間は両方に同じだけ乗る）。
        bi = bnds[:, pair_i, :]
        bj = bnds[:, pair_j, :]
        ax0, ay0, ax1, ay1 = bi[:, :, 0], bi[:, :, 1], bi[:, :, 2], bi[:, :, 3]
        bx0, by0, bx1, by1 = bj[:, :, 0], bj[:, :, 1], bj[:, :, 2], bj[:, :, 3]
        near = (ax1 >= bx0) & (bx1 >= ax0) & (ay1 >= by0) & (by1 >= ay0)
        dx = np.maximum(ax1, bx1) - np.minimum(ax0, bx0)
        dy = np.maximum(ay1, by1) - np.minimum(ay0, by0)
        thresh = fifth.OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
        ox = np.minimum(ax1, bx1) - np.maximum(ax0, bx0)
        oy = np.minimum(ay1, by1) - np.maximum(ay0, by0)
        cand = near & (ox * oy > thresh)
        cand[np.arange(n_batch, dtype=np.intp)[:, None],
             pair_idx[trees_arr[:, :, 0], trees_arr[:, :, 1]]] = False
        rows, cols = np.nonzero(cand)
        counts = np.bincount(rows, minlength=n_batch)
        col_list = cols.tolist()
        thr_list = thresh[rows, cols].tolist()
        CNT["candidates"] += int(rows.size)
        CNT["nets"] += n_batch

        # (a) ネットごとに逐次・早期終了
        t = time.perf_counter()
        out_a = [True] * n_batch
        pos = 0
        for k in range(n_batch):
            n_cand = int(counts[k])
            if n_cand == 0:
                continue
            polys = polys_list[k]
            end = pos + n_cand
            for tt in range(pos, end):
                i, j = pairs[col_list[tt]]
                if polys[i].intersection(polys[j]).area > thr_list[tt]:
                    out_a[k] = False
                    break
            pos = end
        ACC["loop"] += time.perf_counter() - t

        # (b) 窓内で一括（早期終了なし）
        t = time.perf_counter()
        flat = np.concatenate([np.asarray(p, dtype=object) for p in polys_list])
        gi = rows * n_faces + pair_i[cols]
        gj = rows * n_faces + pair_j[cols]
        areas = shapely.area(shapely.intersection(flat[gi], flat[gj]))
        hit = areas > thresh[rows, cols]
        bad = np.zeros(n_batch, dtype=bool)
        bad[rows[hit]] = True
        out_b = np.logical_not(bad).tolist()
        ACC["batch"] += time.perf_counter() - t

        if out_a != out_b:
            CNT["mismatch"] += 1
        return out_a

    fifth._valid_batch = compare
    n = 0
    t0 = time.perf_counter()
    for _t, _p in fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals):
        n += 1
    total = time.perf_counter() - t0

    print(f"{name}: nets={n} pipeline={total:.3f}s")
    print(f"  candidates={CNT['candidates']} nets_in_batch={CNT['nets']} mismatch={CNT['mismatch']}")
    print(f"  (a) per-net loop  {ACC['loop']:7.3f}s")
    print(f"  (b) batched       {ACC['batch']:7.3f}s")
    print(f"  ratio a/b = {ACC['loop'] / ACC['batch']:.3f}x")


if __name__ == "__main__":
    main()
