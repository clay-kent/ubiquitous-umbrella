"""`_valid_batch` の内訳と書き方違いの比較（現行入力の記録 → 交互計測）。

`_valid_batch` は今やパイプライン最大のフェーズ（GEOS の intersection 呼び出し
51,605 回 + 外接矩形の numpy 前置フィルタ）。ここでは本番の入力をそのまま記録し、
同じ入力に対して複数の書き方を **窓ごとに順序を入れ替えて** 走らせ、
判定が一致することと所要時間の比を出す。

    uv run python tools/probe_valid_parts.py stairs 5
"""

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

WINDOWS: list[tuple[Any, Any, Any]] = []


def record() -> None:
    """本番の `_valid_batch` 呼び出しの引数を記録する。"""
    orig = fifth._valid_batch

    def wrapper(polys: Any, bnds: Any, trees_arr: Any) -> Any:
        WINDOWS.append((polys, bnds, trees_arr))
        return orig(polys, bnds, trees_arr)

    fifth._valid_batch = wrapper


def cand_of(polys_list: Any, bnds: Any, trees_arr: Any) -> tuple[list, int]:
    """現行と同じ手順で「交差計算に進む面対」と閾値を取り出す（計測用の写し）。"""
    n_batch = len(polys_list)
    n_faces = bnds.shape[1]
    if n_faces < 2:
        return [], 0
    pairs, pair_idx = fifth.pair_table(n_faces)
    pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
    pair_j = np.array([j for _, j in pairs], dtype=np.intp)
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
    cand[np.arange(n_batch, dtype=np.intp)[:, None], pair_idx[trees_arr[:, :, 0], trees_arr[:, :, 1]]] = False
    rows, cols = np.nonzero(cand)
    work = []
    if rows.size:
        counts = np.bincount(rows, minlength=n_batch)
        thr = thresh[rows, cols].tolist()
        pos = 0
        for k in range(n_batch):
            n_cand = int(counts[k])
            if n_cand == 0:
                continue
            work.append((k, cols[pos : pos + n_cand].tolist(), thr[pos : pos + n_cand]))
            pos += n_cand
    return work, len(pairs)


def v_orig(polys_list: Any, work: list, pairs: Any) -> list:
    out = [True] * len(polys_list)
    for k, cols, thr in work:
        polys = polys_list[k]
        for t, col in enumerate(cols):
            i, j = pairs[col]
            if polys[i].intersection(polys[j]).area > thr[t]:
                out[k] = False
                break
    return out


def v_split(polys_list: Any, work: list, pairs: Any) -> list:
    """i, j を先に展開してタプル添字を消す。"""
    out = [True] * len(polys_list)
    for k, cols, thr in work:
        polys = polys_list[k]
        for t, col in enumerate(cols):
            i = tris_i[col]
            j = tris_j[col]
            if polys[i].intersection(polys[j]).area > thr[t]:
                out[k] = False
                break
    return out


def v_list(polys_list: Any, work: list, pairs: Any) -> list:
    """ネットごとに list へ落として添字を速くする。"""
    out = [True] * len(polys_list)
    for k, cols, thr in work:
        polys = polys_list[k].tolist()
        for t, col in enumerate(cols):
            i = tris_i[col]
            j = tris_j[col]
            if polys[i].intersection(polys[j]).area > thr[t]:
                out[k] = False
                break
    return out


def v_modfun(polys_list: Any, work: list, pairs: Any) -> list:
    """shapely.intersection / shapely.area のモジュール関数版。"""
    out = [True] * len(polys_list)
    for k, cols, thr in work:
        polys = polys_list[k]
        for t, col in enumerate(cols):
            i = tris_i[col]
            j = tris_j[col]
            if shapely.area(shapely.intersection(polys[i], polys[j])) > thr[t]:
                out[k] = False
                break
    return out


def v_nogeos(polys_list: Any, work: list, pairs: Any) -> list:
    """交差計算を外したときの残り（Python + 添字）だけを見るための対照。"""
    out = [True] * len(polys_list)
    for k, cols, thr in work:
        polys = polys_list[k]
        for t, col in enumerate(cols):
            i = tris_i[col]
            j = tris_j[col]
            if polys[i].area > thr[t]:
                continue
    return out


def v_batchwin(polys_list: Any, work: list, pairs: Any) -> list:
    """窓の候補ペアを全部まとめて1回のバッチ交差で処理する。

    判定は「候補のどれか1つでも閾値を超えれば不正」なので順序に依らない。
    早期終了は「計算する交差の個数」を減らす効果しかない（51,605 / 124,039）。
    """
    n = len(polys_list)
    if not work:
        return [True] * n
    flat = np.concatenate(polys_list)
    f_per = len(polys_list[0])
    ks = []
    ii = []
    jj = []
    tt = []
    for k, cols, thr in work:
        for t, col in enumerate(cols):
            ks.append(k)
            ii.append(tris_i[col])
            jj.append(tris_j[col])
            tt.append(thr[t])
    kk = np.asarray(ks, dtype=np.intp)
    A = flat[kk * f_per + np.asarray(ii, dtype=np.intp)]
    B = flat[kk * f_per + np.asarray(jj, dtype=np.intp)]
    areas = shapely.area(shapely.intersection(A, B))
    over = areas > np.asarray(tt, dtype=np.float64)
    out = [True] * n
    for idx, k in enumerate(kk.tolist()):
        if over[idx]:
            out[k] = False
    return out


VARIANTS = {
    "orig": v_orig,
    "split": v_split,
    "list": v_list,
    "modfun": v_modfun,
    "nogeos": v_nogeos,
    "batchwin": v_batchwin,
}

tris_i: list[int] = []
tris_j: list[int] = []


def main() -> None:
    global tris_i, tris_j
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    import shapely  # noqa: PLC0415  (v_modfun 用)

    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    record()
    n = sum(1 for _ in fifth.iter_unique_nets(v, faces, iter(trees), normals))

    # 入力の記録から (ネット, 候補列, 閾値) を作り直す。
    prepped = []
    for polys_list, bnds, trees_arr in WINDOWS:
        if bnds.shape[1] < 2:
            continue
        work, _np_ = cand_of(polys_list, bnds, trees_arr)
        if work:
            prepped.append((polys_list, work))
    n_calls = sum(len(cols) for _p, work in prepped for _k, cols, _t in work)
    print(f"{name}: nets={n} windows={len(prepped)} candidate pairs={n_calls}")

    pairs_np = fifth.pair_table(WINDOWS[0][1].shape[1])[0]
    tris_i = [i for i, _ in pairs_np]
    tris_j = [j for _, j in pairs_np]

    # 判定の一致確認。
    ref = [
        v_orig(polys_list, work, pairs_np) for polys_list, work in prepped
    ]
    for label, fn in VARIANTS.items():
        if label == "nogeos":
            continue  # 交差計算を外した対照なので判定は変わる
        for w, (polys_list, work) in enumerate(prepped):
            got = fn(polys_list, work, pairs_np)
            if got != ref[w]:
                raise SystemExit(f"{label}: 判定が不一致 (window {w})")

    totals = {k: [] for k in VARIANTS}
    for r in range(rounds):
        order = sorted(VARIANTS, key=lambda k: (list(VARIANTS).index(k) + r) % len(VARIANTS))
        acc = {k: 0.0 for k in VARIANTS}
        for w, (polys_list, work) in enumerate(prepped):
            for label in order:
                t0 = time.perf_counter()
                VARIANTS[label](polys_list, work, pairs_np)
                acc[label] += time.perf_counter() - t0
        for k in VARIANTS:
            totals[k].append(acc[k])
        print("    round " + " ".join(f"{k}={acc[k]:7.3f}" for k in VARIANTS))
    base = sorted(totals["orig"])
    print("  ratio to orig (median of per-round ratios):")
    for k in VARIANTS:
        if k == "orig":
            continue
        rs = sorted(totals["orig"][i] / totals[k][i] for i in range(rounds))
        print(f"    {k:8s} {rs[len(rs) // 2]:.3f}x")
    print(f"  orig min={base[0]:.3f}s  (per call {base[0] / n_calls * 1e6:.1f} us)")


if __name__ == "__main__":
    main()
