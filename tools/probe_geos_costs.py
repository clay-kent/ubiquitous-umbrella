"""実際に交差計算へ回る面対で、GEOS 呼び出しの種類別コストを測る。

この環境では GEOS のスカラ呼び出しが極端に高い（4角形の自己交差ですら 47us、
`shapely.Polygon` の構築が 91us）。`_valid_batch` は 51,605 回
`intersection().area` を呼ぶので、**より安い呼び出しで同じ判定が得られるか**が
今いちばん効く。結果の種類（Polygon / Point / 線 / 空）ごとに分けて、
代替呼び出しのコストと「面積 > 閾値」との整合を調べる。

    uv run python tools/probe_geos_costs.py stairs
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

WINDOWS: list[tuple] = []


def collect(name: str) -> list[tuple[object, object, float, str, float]]:
    """本番と同じ順で候補ペアを取り出し、(A, B, 閾値, 結果種別, 面積) を貯める。"""
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
    sum(1 for _ in fifth.iter_unique_nets(v, faces, iter(trees), normals))
    fifth._valid_batch = orig

    out = []
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
                inter = a.intersection(b)
                area = inter.area
                out.append((a, b, float(thr[t]), inter.geom_type if not inter.is_empty else "EMPTY", area))
                if area > thr[t]:
                    break
            pos += n_cand
    return out


def timed(fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    per_kind = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    data = collect(name)
    print(f"{name}: pairs={len(data)}")

    by_kind: dict[str, list] = defaultdict(list)
    for rec in data:
        by_kind[rec[3]].append(rec)
    print("  種類:", {k: len(v) for k, v in by_kind.items()})

    agree: Counter[str] = Counter()
    for kind, recs in sorted(by_kind.items()):
        sample = recs[:per_kind]
        n = len(sample)
        costs = {
            "intersection": timed(lambda: [a.intersection(b) for a, b, *_ in sample], 1),
            "intersection.area": timed(lambda: [a.intersection(b).area for a, b, *_ in sample], 1),
            "area": timed(lambda: [a.area for a, b, *_ in sample], 1),
            "intersects": timed(lambda: [a.intersects(b) for a, b, *_ in sample], 1),
            "overlaps": timed(lambda: [a.overlaps(b) for a, b, *_ in sample], 1),
            "touches": timed(lambda: [a.touches(b) for a, b, *_ in sample], 1),
            "distance": timed(lambda: [a.distance(b) for a, b, *_ in sample], 1),
            "relate_pattern": timed(
                lambda: [a.relate_pattern(b, "T********") for a, b, *_ in sample], 1
            ),
        }
        # 「面積 > 閾値」と predicate の一致（*この種類の中では*）
        for label, fn in (("overlaps", lambda a, b: a.overlaps(b)),
                          ("intersects", lambda a, b: a.intersects(b)),
                          ("interior_pt", None)):
            if fn is None:
                continue
            for a, b, thr, _k, area in sample:
                got = bool(fn(a, b))
                want = area > thr
                agree[f"{label}_{kind}"] += (got == want)
        print(f"  {kind:18s} n={n:5d}  面積>閾値={sum(1 for r in sample if r[4] > r[2])}/{n}")
        for label, us in costs.items():
            print(f"     {label:20s} {us:7.2f} us")
        for label in ("overlaps", "intersects"):
            ok = agree[f"{label}_{kind}"]
            print(f"     {label:20s} 一致 {ok}/{n}")


if __name__ == "__main__":
    main()
