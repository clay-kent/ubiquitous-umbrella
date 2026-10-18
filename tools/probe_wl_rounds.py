"""`_batch_stabilize` のラウンドごとの費用と残存行数を測る。

WL は「全行が安定するまで」回すので、後半のラウンドは数行しか残らない。
ラウンドを上限 K で打ち切って残りを個別化に回す案の費用対効果を判断する。

    uv run python tools/probe_wl_rounds.py stairs
"""

import sys
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

_WINDOW = fifth._LADDER_WINDOW


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    trees_all = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    print(f"{name}: {len(trees_all)} trees, window={_WINDOW}")

    round_time: dict[int, float] = defaultdict(float)
    round_rows: dict[int, int] = defaultdict(int)
    round_calls: dict[int, int] = defaultdict(int)

    rest = iter(trees_all)
    for ctx_i, context in enumerate((ctx, mirror_ctx)):
        rest = iter(trees_all)
        while True:
            window = list(islice(rest, _WINDOW))
            if not window:
                break
            edges_arr = np.asarray(window, dtype=np.int64)
            positions = context.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
            partner, flip = fifth._glue_batch(context, positions)
            _time_stabilize(context, partner, flip, round_time, round_rows, round_calls)

    print(f"{'round':>5} {'calls':>7} {'rows':>10} {'time_s':>9} {'ms/call':>8}")
    for r in sorted(round_time):
        print(
            f"{r:>5} {round_calls[r]:>7} {round_rows[r]:>10} "
            f"{round_time[r]:>9.3f} {1000 * round_time[r] / round_calls[r]:>8.3f}"
        )


def _time_stabilize(ctx, partner, flip, round_time, round_rows, round_calls) -> None:
    """`_batch_stabilize` をラウンドごとに計測しながら回す（同じ計算）。"""
    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    partner_or_self = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    partner_coef = glued * fifth._WL_C
    tag = glued * fifth._WL_D + flip.astype(np.uint64) * fifth._WL_E
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * fifth._WL_E0 + tag
    counts = fifth._distinct_counts(h)
    nxt = ctx.np_nxt
    n_rows = len(h)
    row_ids = np.arange(n_rows, dtype=np.int64)
    out_counts = np.empty(n_rows, dtype=np.int64)
    out_counts[row_ids] = counts
    rnd = 0
    while True:
        rnd += 1
        n_before = len(h)
        t0 = time.perf_counter()
        nh = (
            h * fifth._WL_A
            + h[:, nxt] * fifth._WL_B
            + np.take_along_axis(h, partner_or_self, axis=1) * partner_coef
            + tag
        )
        new_counts = fifth._distinct_counts(nh)
        done = new_counts == counts
        if done.all():
            dt = time.perf_counter() - t0
            round_time[rnd] += dt
            round_rows[rnd] += n_before
            round_calls[rnd] += 1
            break
        keep = ~done
        h = nh[keep]
        partner_or_self = partner_or_self[keep]
        partner_coef = partner_coef[keep]
        tag = tag[keep]
        counts = new_counts[keep]
        row_ids = row_ids[keep]
        dt = time.perf_counter() - t0
        round_time[rnd] += dt
        round_rows[rnd] += n_before
        round_calls[rnd] += 1


if __name__ == "__main__":
    main()
