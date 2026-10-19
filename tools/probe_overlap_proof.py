"""「面積 > 閾値」を GEOS の intersection 無しで**証明**できるかを見積もる。

案: 面ごとに「内部に完全に入る円」（最大内接円）を前計算しておく（剛体変換で
半径は不変）。候補ペア (A, B) について、A の円の中心 p が B の内部にあり、
B の境界までの距離も十分なら、A ∩ B は半径 min(r_A, dist(p, ∂B)) の円板を含むので

    area(A ∩ B) >= pi * min(r_A, dist(p, ∂B))^2 > thresh

が**証明**できる（GEOS の intersection を呼ばずに不正と判定できる）。逆向きも試す。

ここでは本番と同じ候補列を記録し、実際に「面積 > 閾値」になったペアに対して
この条件が成り立つ割合を測る（成り立てばその回数ぶん GEOS 呼び出しが消える）。

    uv run python tools/probe_overlap_proof.py stairs
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
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
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

    pairs_hit = 0
    proven = 0
    no_circle = 0
    ratios: list[float] = []
    n_probe = 0
    t0 = time.perf_counter()
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
                area = a.intersection(b).area
                if area > thr[t]:
                    pairs_hit += 1
                    ratios.append(area / min(a.area, b.area))
                    ok = False
                    for p_, q_ in ((a, b), (b, a)):
                        mic = shapely.maximum_inscribed_circle(p_, 1e-9)
                        c = shapely.Point(mic.coords[0])
                        r = mic.length
                        n_probe += 1
                        if r <= 0:
                            no_circle += 1
                            continue
                        d = shapely.distance(c, q_.exterior)
                        if q_.contains(c) and min(r, d) ** 2 * np.pi > thr[t]:
                            ok = True
                            break
                    proven += ok
                    if limit and pairs_hit >= limit:
                        break
                if area > thr[t]:
                    break
            pos += n_cand
            if limit and pairs_hit >= limit:
                break
        if limit and pairs_hit >= limit:
            break

    dt = time.perf_counter() - t0
    ratios.sort()
    print(f"{name}: nets={n_nets} 面積超過ペア={pairs_hit} 証明できた={proven} "
          f"({proven / max(pairs_hit, 1) * 100:.1f}%) 内接円なし={no_circle}  ({dt:.1f}s)")
    if ratios:
        print(f"  交差面積/小さい面の面積: 最小 {ratios[0]:.3g} "
              f"中央 {ratios[len(ratios) // 2]:.3g} 最大 {ratios[-1]:.3g}")


if __name__ == "__main__":
    main()
