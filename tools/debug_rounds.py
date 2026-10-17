"""コミット済みコードで、_batch_stabilize が実際に何ラウンド回って
各行が何色まで到達するかを記録する（行ごとの凍結ラウンドも）。"""

import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

A, B, C, D, E, E0 = (
    fifth._WL_A, fifth._WL_B, fifth._WL_C, fifth._WL_D, fifth._WL_E, fifth._WL_E0
)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "prism7"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    n = ctx.n_nodes

    rest = iter(trees)
    final_counts = Counter()
    freeze_round = Counter()
    total_rounds = 0
    windows = 0
    while True:
        w = list(itertools.islice(rest, fifth._LADDER_WINDOW))
        if not w:
            break
        windows += 1
        pos = np.asarray(
            [[ctx.adj_pos_of[e] for e in t] for t in w], dtype=np.int64
        )
        partner, flip = fifth._glue_batch(ctx, pos)

        glued = (partner >= 0).astype(np.uint64)
        por = np.where(partner >= 0, partner, np.arange(n)[None, :])
        pc = glued * C
        tag = glued * D + flip.astype(np.uint64) * E
        uniq = np.unique(ctx.np_edge_len)
        eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
        h = eid[None, :] * E0 + tag
        counts = fifth._distinct_counts(h)
        nxt = ctx.np_nxt
        unstable = np.ones(len(h), dtype=bool)
        rnd = 0
        while unstable.any():
            rnd += 1
            nh = (
                h * A
                + h[:, nxt] * B
                + np.take_along_axis(h, por, axis=1) * pc
                + tag
            )
            nc = fifth._distinct_counts(nh)
            done = unstable & (nc == counts)
            h[done] = nh[done]
            unstable &= ~done
            h[unstable] = nh[unstable]
            counts = np.where(done, counts, nc)
            for r in np.flatnonzero(done).tolist():
                freeze_round[(rnd, int(counts[r]))] += 1
        total_rounds = max(total_rounds, rnd)
        for c in counts.tolist():
            final_counts[c] += 1

    print(f"{name}: windows={windows} n_nodes={n} max_rounds_in_window={total_rounds}")
    print(f"  final color counts: {sorted(final_counts.items())}")
    print(f"  fully refined rows: {final_counts.get(n, 0)} / {len(trees)}")
    print("  (round, colors at freeze) -> rows:")
    for (r, c), k in sorted(freeze_round.items())[:12]:
        print(f"    round {r:2d}: {c:3d} colors -> {k}")


if __name__ == "__main__":
    main()