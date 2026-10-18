"""バッチ化WL安定化の実験。prism7 の実データで「現在実装 vs バッチnumpy」を計測する。

本プロトタイプは tools/ の実験ツールであり、fifth.py は変更しない。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

# 安定化だけを再現する純Python extractor（fifth._wl_refine を流用）


def collect_glue(ctx, trees):
    """各木に対して (partner, flip) を作る（fifth._glue_arrays と同じ）。"""
    out = []
    for tree in trees:
        out.append(fifth._glue_arrays(ctx, list(tree)))
    return out


def build_initial(ctx, partner, flip):
    color = [(ctx.edge_len[i], partner[i] >= 0, flip[i]) for i in range(ctx.n_nodes)]
    return color


def wl_python(color, nxt, partner, flip, n):
    """fifth._wl_refine と同じアルゴリズム（参照実装、ここでは雰囲気確認用）。"""
    while True:
        new = [
            hash(
                (
                    color[i],
                    color[nxt[i]],
                    None if partner[i] < 0 else (flip[i], color[partner[i]]),
                )
            )
            for i in range(n)
        ]
        if len(set(new)) == len(set(color)):
            return new
        color = new


def wl_batch(colors, nxt_idx, partner_idx, flip, n, batch):
    """バッチ版。colors: (B, n) int64（初期色は小さい整数に刻印済みとする）。

    パートナーが無い半辺は partner_idx[:, i] == 自分自身 として扱う（自己ループだと
    c[partner] == c[i] になる）。正確さのため後で正しい初期同伴でやり直す。
    ここでは実験速度を確認するだけ。
    """
    # 初期色は既に int 化済みの辞書刻印で与えられている
    A = np.int64(37)
    B_ = np.int64(101)
    C = np.int64(10007)
    cpe = colors[:, nxt_idx]  # 暫定: nxt色
    while True:
        c_nxt = np.take_along_axis(colors, nxt_idx[None, :], axis=1)[:, 0]
        c_p = np.take_along_axis(colors, partner_idx, axis=1)
        new = (A * colors + B_ * c_nxt + C * c_p + flip.astype(np.int64)) * np.int64(2)
        cnt_prev = ...  # 実験コメント: 本実装は場当たり的で正しい収束判定をしていない
        if cnt_prev is None:
            pass
        # 正しい実装: 各行の色数を sort ベースで数える
        sorted_rows = np.sort(colors, axis=1)
        cnt0 = 1 + (sorted_rows[:, 1:] != sorted_rows[:, :-1]).sum(axis=1)
        sorted_new = np.sort(new, axis=1)
        cnt1 = 1 + (sorted_new[:, 1:] != sorted_new[:, :-1]).sum(axis=1)
        if len(set(cnt0.tolist())) == len(set(cnt1.tolist())):
            # 実験: 収束した行だけ抜けても、ウインドウ全体の収束まで回す
            return new, cnt1


def main():
    v, f = SOLIDS["prism7"]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))

    ctx = fifth.RibbonContext(v, faces)
    n = ctx.n_nodes
    print(f"faces={len(faces)} trees={len(trees)} half_edges={n}")

    # --- 現在実装 (純Python) の時間 -------------------------------------
    t0 = time.perf_counter()
    for tree in trees:
        partner, flip = fifth._glue_arrays(ctx, tree)
        color = build_initial(ctx, partner, flip)
        wl_python(color, ctx.nxt, partner, flip, n)
    t_py = time.perf_counter() - t0
    print(f"python WL (1 context): {t_py:.2f}s  ({t_py/len(trees)*1000:.3f} ms/tree)")

    # --- バッチ版の設置 ---------------------------------------------------
    # 初期色を小さな int に刻印（辞書で一意化 → ランク）。
    # partner 無しは -1 → 専用の番兵にする（ここでは n (自分自身) を使う実験）
    nxt_arr = np.asarray(ctx.nxt, dtype=np.int64)
    t0 = time.perf_counter()
    windows = 512
    # partner/flip をバッチ化: 簡易化のためここでは _glue_arrays をループして画一化
    partner_rows = np.full((len(trees), n), -1, dtype=np.int64)
    flip_rows = np.full((len(trees), n), 0, dtype=np.int64)
    for t_i, tree in enumerate(trees):
        p, fl = fifth._glue_arrays(ctx, tree)
        partner_rows[t_i] = p
        flip_rows[t_i] = fl
    t_glue = time.perf_counter() - t0
    print(f"glue only: {t_glue:.2f}s")

    # 辞書刻印: 全 (edge_len, glued?, flip) の一意化は context 単位でできる。
    uniq = {}
    init_np = np.empty((len(trees), n), dtype=np.int64)
    for t_i in range(len(trees)):
        row = [
            (ctx.edge_len[i], partner_rows[t_i][i] >= 0, flip_rows[t_i][i])
            for i in range(n)
        ]
        for i, key in enumerate(row):
            if key not in uniq:
                uniq[key] = len(uniq)
        init_np[t_i] = [uniq[key] for key in row]
    t_init = time.perf_counter()
    # glue なしに自分を指す場所は c_p = 自分の色 → 初期プログラムで色が情報を
    # 「持てない」問題の暫定扱い。実験なのでこのまま速度だけ見る。
    # glue なしに自分を指す場所は c_p = 自分の色 → 初期プログラムで色が情報を
    # 「持てない」問題の暫定扱い。実験なのでこのまま速度だけ見る。
    me = np.arange(n)[None, :]
    partner_np = np.where(partner_rows < 0, me, partner_rows)

    t0 = time.perf_counter()
    for W in range(0, len(trees), windows):
        batch = init_np[W:W + windows]
        partner_b = partner_np[W:W + windows]
        flip_b = flip_rows[W:W + windows]
        colors = batch
        for _ in range(64):
            c_nxt = colors[:, nxt_arr]
            c_p = np.take_along_axis(colors, partner_b, axis=1)
            new = (np.int64(37) * colors + np.int64(101) * c_nxt
                   + np.int64(10007) * c_p
                   + np.int64(293) * flip_b)
            sorted_mid = np.sort(colors, axis=1)
            cnt0 = 1 + (sorted_mid[:, 1:] != sorted_mid[:, :-1]).sum(axis=1)
            sorted_new = np.sort(new, axis=1)
            cnt1 = 1 + (sorted_new[:, 1:] != sorted_new[:, :-1]).sum(axis=1)
            done = cnt0 == cnt1
            if done.all():
                break
            colors = new
    t_np = time.perf_counter() - t0
    print(
        f"numpy batch WL: {t_np:.2f}s  ({t_np/len(trees)*1000:.3f} ms/tree) "
        f"(init/glue 準備は含まない; 精密な収束判定は本実装で行う)"
    )

    # glue のバッチ化実験: 位置表から put_along_axis で一気に作る
    # (本実装でも使うので、ここのタイミングを取る)


if __name__ == "__main__":
    main()
