"""`_batch_stabilize` の 1 ラウンドを部位別に CPU 時間で測る。

「更新 / 色数カウント / 確定記録 / 行の間引きコピー / その他」に分ける。

⚠️ Windows の ``time.process_time`` は粒度が約 15.6ms なので、1 ラウンド
（数 ms）を単発で測ると 0 になる。**全窓・両文脈を合計**してから割合を見る。

    uv run python tools/probe_stabilize_parts.py stairs
"""

import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

_PARTS = ("update", "counts", "record", "prune", "other")


def run_context(ctx, window, clocks):
    edges_arr = np.asarray(window, dtype=np.int64)
    positions = ctx.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
    partner, flip = fifth._glue_batch(ctx, positions)

    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    partner_or_self = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    partner_coef = glued * fifth._WL_C
    tag = glued * fifth._WL_D + flip.astype(np.uint64) * fifth._WL_E
    t0 = time.process_time()
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * fifth._WL_E0 + tag
    counts = fifth._distinct_counts(h)
    clocks["other"] += time.process_time() - t0

    nxt = ctx.np_nxt
    n_rows = len(h)
    row_ids = np.arange(n_rows, dtype=np.int64)
    out_colors = np.empty((n_rows, n_nodes), dtype=np.uint64)
    out_counts = np.empty(n_rows, dtype=np.int64)

    while True:
        t0 = time.process_time()
        nh = h * fifth._WL_A
        t = h[:, nxt]
        t *= fifth._WL_B
        nh += t
        t2 = np.take_along_axis(h, partner_or_self, axis=1)
        t2 *= partner_coef
        nh += t2
        nh += tag
        t1 = time.process_time()
        clocks["update"] += t1 - t0

        new_counts = fifth._distinct_counts(nh)
        t2c = time.process_time()
        clocks["counts"] += t2c - t1

        done = new_counts == counts
        out_colors[row_ids[done]] = nh[done]
        out_counts[row_ids[done]] = new_counts[done]
        t3 = time.process_time()
        clocks["record"] += t3 - t2c

        if done.all():
            break

        keep = ~done
        h = nh[keep]
        partner_or_self = partner_or_self[keep]
        partner_coef = partner_coef[keep]
        tag = tag[keep]
        counts = new_counts[keep]
        row_ids = row_ids[keep]
        clocks["prune"] += time.process_time() - t3


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    clocks = dict.fromkeys(_PARTS, 0.0)
    t_start = time.process_time()
    rest = iter(trees)
    while True:
        window = list(islice(rest, fifth._LADDER_WINDOW))
        if not window:
            break
        run_context(ctx, window, clocks)
        run_context(mirror_ctx, window, clocks)
    total = time.process_time() - t_start

    print(f"{name}: trees={len(trees)} total_cpu={total:.3f}s")
    accounted = sum(clocks.values())
    for k in _PARTS:
        print(f"  {k:8s} {clocks[k]:8.4f}s  {100 * clocks[k] / total:5.1f}%")
    print(f"  (glue_batch など上記以外) {total - accounted:.4f}s"
          f"  {100 * (total - accounted) / total:5.1f}%")


if __name__ == "__main__":
    main()
