"""候補ペアを走査する**順序**で交差計算の回数がどれだけ変わるかを測る。

`_valid_batch` の判定は「候補ペアのどれかが重なれば不正」なので、走査順は
結果に効かない（早期終了は速さのためだけ）。ならば「重なりそうなペア」から
先に試すほうが、不正なネットで早く打ち切れて交差計算が減るはず。

ここでは同じ窓で 2 通りの順序を交互に走らせ、交差計算の回数と判定を比べる:

  (a) 現行     : 面対の添字昇順（i 昇順 → j 昇順）
  (b) 再並べ替え: 外接矩形の重なり面積 ox*oy の降順

    uv run python tools/probe_candidate_order.py stairs
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

ACC = {"cur": 0.0, "sorted": 0.0}
CNT = {"cand": 0, "intersect_cur": 0, "intersect_sorted": 0, "mismatch": 0, "nets": 0}


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
        # 並べ替え用の鍵: 外接矩形の重なり面積（大きいほど重なりそう）
        overlap = (ox * oy)[rows, cols].tolist()
        CNT["cand"] += int(rows.size)
        CNT["nets"] += n_batch

        def walk(order, out, acc_key):  # type: ignore[no-untyped-def]
            pos = 0
            for k in range(n_batch):
                n_cand = int(counts[k])
                if n_cand == 0:
                    continue
                polys = polys_list[k]
                end = pos + n_cand
                for tt in order[pos:end]:
                    i, j = pairs[col_list[tt]]
                    CNT[acc_key] += 1
                    if polys[i].intersection(polys[j]).area > thr_list[tt]:
                        out[k] = False
                        break
                pos = end

        t = time.perf_counter()
        out_cur = [True] * n_batch
        walk(list(range(len(col_list))), out_cur, "intersect_cur")
        ACC["cur"] += time.perf_counter() - t

        # ネットごとに overlap 降順へ並べ替える（ネット境界は崩さない）。
        order = list(range(len(col_list)))
        pos = 0
        for k in range(n_batch):
            n_cand = int(counts[k])
            if n_cand:
                seg = order[pos:pos + n_cand]
                seg.sort(key=lambda t_: overlap[t_], reverse=True)
                order[pos:pos + n_cand] = seg
            pos += n_cand
        t = time.perf_counter()
        out_sorted = [True] * n_batch
        walk(order, out_sorted, "intersect_sorted")
        ACC["sorted"] += time.perf_counter() - t

        if out_cur != out_sorted:
            CNT["mismatch"] += 1
        return out_cur

    fifth._valid_batch = timed
    n = 0
    t0 = time.perf_counter()
    for _t, _p in fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals):
        n += 1
    total = time.perf_counter() - t0

    print(f"{name}: nets={n} pipeline={total:.3f}s")
    print(f"  candidates={CNT['cand']} mismatch={CNT['mismatch']}")
    print(f"  (a) 添字昇順 : intersect={CNT['intersect_cur']:7d}  {ACC['cur']:7.3f}s")
    print(f"  (b) 重なり順 : intersect={CNT['intersect_sorted']:7d}  {ACC['sorted']:7.3f}s")
    print(f"  比 a/b = {ACC['cur'] / ACC['sorted']:.3f}x")


if __name__ == "__main__":
    main()
