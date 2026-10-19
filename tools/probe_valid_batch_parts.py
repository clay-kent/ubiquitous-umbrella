"""`_valid_batch` の内訳（外接矩形の絞り込み / 候補の取り出し / 交差計算）を測る。

`_valid_batch` の中身をここに写して各部分の累計時間と交差計算の回数を数える。
パイプラインを 1 回走らせるだけなので、分割の比が読めればよい。

    uv run python tools/probe_valid_batch_parts.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import fifth
from bench_ladder import SOLIDS

ACC = {"prefilter": 0.0, "extract": 0.0, "loop_scan": 0.0, "intersect": 0.0}
N = {"candidates": 0, "intersections": 0, "nets": 0, "cand_nets": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    pairs, pair_idx = fifth.pair_table(len(faces))
    pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
    pair_j = np.array([j for _, j in pairs], dtype=np.intp)

    def timed(polys_list, bnds, trees_arr):  # type: ignore[no-untyped-def]
        n_batch = len(polys_list)
        n_faces = bnds.shape[1]
        if n_batch == 0 or n_faces < 2:
            return [True] * n_batch
        N["nets"] += n_batch
        t = time.perf_counter()
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
        ACC["prefilter"] += time.perf_counter() - t

        t = time.perf_counter()
        rows, cols = np.nonzero(cand)
        out = [True] * n_batch
        counts = np.bincount(rows, minlength=n_batch)
        col_list = cols.tolist()
        thr_list = thresh[rows, cols].tolist()
        ACC["extract"] += time.perf_counter() - t
        N["candidates"] += int(rows.size)
        N["cand_nets"] += int(np.count_nonzero(counts))

        t = time.perf_counter()
        pos = 0
        for k in range(n_batch):
            n_cand = int(counts[k])
            if n_cand == 0:
                continue
            polys = polys_list[k]
            end = pos + n_cand
            for tt in range(pos, end):
                i, j = pairs[col_list[tt]]
                N["intersections"] += 1
                if polys[i].intersection(polys[j]).area > thr_list[tt]:
                    out[k] = False
                    break
            pos = end
        ACC["loop_scan"] += time.perf_counter() - t
        return out

    fifth._valid_batch = timed
    n = 0
    t0 = time.perf_counter()
    for _t, _p in fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals):
        n += 1
    total = time.perf_counter() - t0

    print(f"{name}: nets={n} pipeline={total:.3f}s")
    print(f"  nets in batch   : {N['nets']}  (候補あり {N['cand_nets']})")
    print(f"  candidates      : {N['candidates']}  intersections: {N['intersections']}")
    for k, s in ACC.items():
        print(f"  {k:12s} {s:7.3f}s  ({s / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()
