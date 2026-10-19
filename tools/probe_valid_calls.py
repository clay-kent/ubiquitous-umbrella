"""`_valid_batch` が実際に交差計算へ回す面対の中身を調べる。

「1 回 50us 前後かかる」のが GEOS の素の仕事なのか、接するだけの
縮退した配置（結果が点・線・空）のせいなのかを見る。

    uv run python tools/probe_valid_calls.py stairs
"""

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

WINDOWS: list[tuple] = []


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    orig = fifth._valid_batch

    def wrapper(polys: object, bnds: object, trees_arr: object) -> object:
        WINDOWS.append((polys, bnds, trees_arr))
        return orig(polys, bnds, trees_arr)

    fifth._valid_batch = wrapper
    n_nets = sum(1 for _ in fifth.iter_unique_nets(v, faces, iter(trees), normals))
    fifth._valid_batch = orig

    kinds: Counter[str] = Counter()
    times: dict[str, list[float]] = {}
    n_calls = 0
    n_verts_a: Counter[int] = Counter()
    overlaps = 0
    for polys_list, bnds, trees_arr in WINDOWS:
        if bnds.shape[1] < 2:
            continue
        n_faces = bnds.shape[1]
        pairs, pair_idx = fifth.pair_table(n_faces)
        pi = np.array([i for i, _ in pairs], dtype=np.intp)
        pj = np.array([j for _, j in pairs], dtype=np.intp)
        bi, bj = bnds[:, pi, :], bnds[:, pj, :]
        ax0, ay0, ax1, ay1 = bi[:, :, 0], bi[:, :, 1], bi[:, :, 2], bi[:, :, 3]
        bx0, by0, bx1, by1 = bj[:, :, 0], bj[:, :, 1], bj[:, :, 2], bj[:, :, 3]
        near = (ax1 >= bx0) & (bx1 >= ax0) & (ay1 >= by0) & (by1 >= ay0)
        dx = np.maximum(ax1, bx1) - np.minimum(ax0, bx0)
        dy = np.maximum(ay1, by1) - np.minimum(ay0, by0)
        thresh = fifth.OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
        ox = np.minimum(ax1, bx1) - np.maximum(ax0, bx0)
        oy = np.minimum(ay1, by1) - np.maximum(ay0, by0)
        cand = near & (ox * oy > thresh)
        cand[np.arange(len(polys_list), dtype=np.intp)[:, None],
             pair_idx[trees_arr[:, :, 0], trees_arr[:, :, 1]]] = False
        rows, cols = np.nonzero(cand)
        counts = np.bincount(rows, minlength=len(polys_list)) if rows.size else np.zeros(0, dtype=int)
        thr = thresh[rows, cols].tolist() if rows.size else []
        pos = 0
        for k in range(len(polys_list)):
            n_cand = int(counts[k]) if counts.size else 0
            if n_cand == 0:
                continue
            polys = polys_list[k]
            for t in range(pos, pos + n_cand):
                i, j = pairs[cols[t]]
                a, b = polys[i], polys[j]
                t0 = time.perf_counter()
                inter = a.intersection(b)
                dt = time.perf_counter() - t0
                area = inter.area
                n_calls += 1
                key = inter.geom_type if not inter.is_empty else "EMPTY"
                kinds[key] += 1
                n_verts_a[len(a.exterior.coords)] += 1
                times.setdefault(key, []).append(dt)
                if area > thr[t]:
                    overlaps += 1
                    break
            pos += n_cand

    print(f"{name}: nets={n_nets} calls={n_calls} overlaps={overlaps}")
    print(f"  結果の種類: {dict(kinds)}")
    print(f"  面Aの頂点数: {dict(sorted(n_verts_a.items()))}")
    for key, ts in sorted(times.items(), key=lambda kv: -len(kv[1])):
        ts.sort()
        print(f"  {key:18s} n={len(ts):6d} 中央値 {ts[len(ts) // 2] * 1e6:6.1f}us "
              f"最小 {ts[0] * 1e6:6.1f}us")


if __name__ == "__main__":
    main()
