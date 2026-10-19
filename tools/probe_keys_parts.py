"""`_batch_canonical_keys` の部位別コスト（CPU 時間、全窓合計）。

`_batch_stabilize` の外側（位置引き・接着・個別化・キー tuple 化）が
署名フェーズの約 1/4 を占めるので、その内訳を見る。

    uv run python tools/probe_keys_parts.py stairs
"""

import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

_PARTS = ("asarray", "positions", "glue", "stabilize", "individualize", "keysort", "tuples")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    clocks = dict.fromkeys(_PARTS, 0.0)
    n_indiv = 0
    n_rows = 0
    t_all = time.process_time()

    rest = iter(trees)
    while True:
        window = list(islice(rest, fifth._LADDER_WINDOW))
        if not window:
            break
        n = len(window)
        t0 = time.process_time()
        edges_arr = np.asarray(window, dtype=np.int64)
        t1 = time.process_time()
        clocks["asarray"] += t1 - t0
        for context in (ctx, mirror_ctx):
            positions = context.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
            t2 = time.process_time()
            clocks["positions"] += t2 - t1
            partner, flip = fifth._glue_batch(context, positions)
            t3 = time.process_time()
            clocks["glue"] += t3 - t2
            colors, counts = fifth._batch_stabilize(context, partner, flip)
            t4 = time.process_time()
            clocks["stabilize"] += t4 - t3
            n_nodes = context.n_nodes
            nxt_list = context.nxt
            for idx in np.flatnonzero(counts != n_nodes).tolist():
                n_indiv += 1
                fifth._individualize_refine(
                    nxt_list,
                    partner[idx].tolist(),
                    flip[idx].tolist(),
                    colors[idx].tolist(),
                    [0],
                )
            t5 = time.process_time()
            clocks["individualize"] += t5 - t4
            rows = np.flatnonzero(counts == n_nodes)
            sorted_colors = np.sort(colors, axis=1).tolist()
            t6 = time.process_time()
            clocks["keysort"] += t6 - t5
            for idx in rows.tolist():
                tuple(sorted_colors[idx])
            t7 = time.process_time()
            clocks["tuples"] += t7 - t6
            n_rows += len(rows)
            t1 = t7

    total = time.process_time() - t_all
    print(f"{name}: total_cpu={total:.3f}s rows={n_rows} individualized={n_indiv}")
    for k in _PARTS:
        print(f"  {k:14s} {clocks[k]:8.4f}s  {100 * clocks[k] / total:5.1f}%")


if __name__ == "__main__":
    main()
