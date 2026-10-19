"""面ごとの「内部に確実に入る円」で、交差面積>閾値を GEOS 無しで証明できる割合を測る。

この環境の GEOS は極端に遅い（`intersection` が 65〜95us、4角形の自己交差ですら
47us）。`_valid_batch` は 51,605 回それを呼ぶので、**GEOS を呼ばずに「重なって
いる」と証明できる**なら大きな削減になる。

証明の形（健全 = 証明できたら本当に面積 > 閾値）:
面 F について「内部に完全に入る円」（中心 p_F, 半径 r_F）を前計算しておく。
剛体変換で配置しても半径は不変。候補ペア (A, B) について A の円の中心 p が
B の内部にあり、B の境界までの距離 d も分かれば

    A ∩ B ⊇ 中心 p 半径 min(r_A, d) の円板  ⟹  area >= pi * min(r_A, d)^2

なので、pi * min(r_A, d)^2 > thresh なら面積 > 閾値が証明できる。逆向きも試す。

ここでは本番と同じ手順で候補列を記録し、「実際に面積 > 閾値になったペア」に対して
探針の本数 k ごとに証明できる割合を出す（そのぶん GEOS 呼び出しが消える）。

    uv run python tools/probe_probe_rates.py stairs
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

WINDOWS: list[tuple] = []


def face_poly_2d(vertices: np.ndarray, face: list[int]) -> np.ndarray:
    """面の3D頂点を面内の2D座標にする（正規直交基底は面から作る）。"""
    pts3 = np.array([vertices[i] for i in face], dtype=np.float64)
    origin = pts3[0]
    e1 = pts3[1] - origin
    e1 = e1 / np.linalg.norm(e1)
    n = np.cross(pts3[1] - pts3[0], pts3[2] - pts3[0])
    n = n / np.linalg.norm(n)
    e2 = np.cross(n, e1)
    d = pts3 - origin
    return np.column_stack([d @ e1, d @ e2])


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """偶数奇数則。p は (n,2)、poly は (m,2)。"""
    x = p[:, 0:1]
    y = p[:, 1:2]
    x0 = poly[:, 0][None, :]
    y0 = poly[:, 1][None, :]
    x1 = np.roll(poly[:, 0], -1)[None, :]
    y1 = np.roll(poly[:, 1], -1)[None, :]
    cond = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (cond & (x < xc)).sum(axis=1) % 2 == 1


def _seg_dist(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """点 p(k,2) から polygon の各辺（線分）への距離の最小値 (k,)。"""
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a  # (m,2)
    ee = (e * e).sum(axis=1)
    out = np.full(len(p), np.inf)
    for m in range(len(a)):
        pe = p - a[m]
        t = (pe @ e[m]) / ee[m] if ee[m] > 0 else np.zeros(len(p))
        t = np.clip(t, 0.0, 1.0)
        closest = a[m] + t[:, None] * e[m]
        d = np.linalg.norm(p - closest, axis=1)
        out = np.minimum(out, d)
    return out


def make_probes(
    vertices: np.ndarray, face: list[int], k: int, grid: int = 28
) -> tuple[np.ndarray, np.ndarray]:
    """面の探針 (a, b) 係数（v0, v1, v2 基準）と半径 r を返す。"""
    pts = face_poly_2d(vertices, face)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    GX, GY = np.meshgrid(xs, ys)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    inside = _point_in_poly(P, pts)
    dist = _seg_dist(P, pts)
    ok = inside & (dist > 0)
    cand = P[ok]
    dcand = dist[ok]
    if len(cand) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    order = np.argsort(-dcand)
    sep = float(np.linalg.norm(hi - lo)) / (3.0 * k)
    chosen: list[int] = []
    for idx in order.tolist():
        if len(chosen) >= k:
            break
        p = cand[idx]
        if all(float(np.linalg.norm(p - cand[j])) >= sep for j in chosen):
            chosen.append(idx)
    sel = cand[chosen]
    rad = dcand[chosen]
    v0, v1, v2 = pts[0], pts[1], pts[2]
    M = np.column_stack([v1 - v0, v2 - v0])
    ab = np.linalg.solve(M, (sel - v0).T).T
    return ab, rad


def place(
    ab: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """(k,2) の係数を、配置済みポリゴンの先頭3頂点に適用して (k,2) を返す。"""
    if len(ab) == 0:
        return np.zeros((0, 2))
    v0, v1, v2 = coords[0], coords[1], coords[2]
    return v0 + ab[:, 0:1] * (v1 - v0) + ab[:, 1:2] * (v2 - v0)


def inside_margin(p: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(多角形の内部か = 偶数奇数則, 境界までの距離)。p は (k,2)。

    ⚠️「全辺で同じ側」という判定は**凸多角形でしか十分でない**（階段形のような
    非凸面では内部の点でも落ちる）。距離 >= rho > 0 が別途言えていれば境界から
    離れているので、偶数奇数則の判定は縮退せずそのまま使える。
    """
    return _point_in_poly(p, poly), _seg_dist(p, poly)


def collect(name: str) -> list[tuple]:
    """本番と同じ順で候補ペアを回し、面積 > 閾値だったペアだけ記録する。"""
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

    hits: list[tuple] = []
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
        cand[
            np.arange(len(polys_list), dtype=np.intp)[:, None],
            pair_idx[trees_arr[:, :, 0], trees_arr[:, :, 1]],
        ] = False
        rows, cols = np.nonzero(cand)
        counts = (
            np.bincount(rows, minlength=len(polys_list))
            if rows.size
            else np.zeros(0, dtype=int)
        )
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
                    hits.append(
                        (
                            i,
                            j,
                            np.array(a.exterior.coords),
                            np.array(b.exterior.coords),
                            float(thr[t]),
                        )
                    )
                    break
            pos += n_cand
    return n_nets, hits


def place_probes(ab: np.ndarray, coords: np.ndarray) -> np.ndarray | None:
    """探針の (a, b) 係数を、配置済みポリゴンの頂点から実際の座標に写す。

    ``shapely.polygons`` を通すと ring の向きが反転している場合があるので、
    (v0,v1,v2) と (v0,v_{m-1},v_{m-2}) の両方を試し、**自分の多角形の内部に
    全探針が入るほう**を採る（探針は面の内部の点なので、どちらかは必ず成り立つ）。
    """
    if len(ab) == 0:
        return None
    for c1, c2 in ((coords[1], coords[2]), (coords[-2], coords[-3])):
        p = coords[0] + ab[:, 0:1] * (c1 - coords[0]) + ab[:, 1:2] * (c2 - coords[0])
        if bool(_point_in_poly(p, coords).all()):
            return p
    return None


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    ks = [1, 2, 3, 4, 6, 8]
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    probe_sets = {fi: make_probes(v, face, max(ks)) for fi, face in enumerate(faces)}
    print(f"{name}: 面ごとの探針 {[len(p[0]) for p in probe_sets.values()]}")

    t0 = time.perf_counter()
    n_nets, hits = collect(name)
    print(f"  nets={n_nets} 面積超過ペア={len(hits)} （収集 {time.perf_counter() - t0:.0f}s）")

    proven = {k: 0 for k in ks}
    placed_ok = 0
    by_pair: dict[tuple[int, int], int] = {}
    for fi, fj, ca, cb, thr in hits:
        by_pair[(fi, fj)] = by_pair.get((fi, fj), 0) + 1
        ok_any = {k: False for k in ks}
        any_placed = False
        for ab, rad, coords_a, coords_b in (
            (probe_sets[fi][0], probe_sets[fi][1], ca, cb),
            (probe_sets[fj][0], probe_sets[fj][1], cb, ca),
        ):
            p = place_probes(ab, coords_a)
            if p is None:
                continue
            any_placed = True
            inside, dist = inside_margin(p, coords_b)
            area = np.pi * np.minimum(rad, dist) ** 2
            good = inside & (area > thr)
            for k in ks:
                if good[:k].any():
                    ok_any[k] = True
        placed_ok += any_placed
        for k in ks:
            proven[k] += ok_any[k]
    print("  面対の内訳:", dict(sorted(by_pair.items(), key=lambda kv: -kv[1])))
    print(f"  探針が配置できたペア: {placed_ok}/{len(hits)}")
    print("  探針数ごとの証明率:")
    for k in ks:
        print(f"    k={k}: {proven[k]}/{len(hits)} = {proven[k] / max(len(hits), 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
