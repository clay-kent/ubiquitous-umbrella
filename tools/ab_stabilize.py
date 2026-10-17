"""_batch_stabilize の圧縮あり/なしを A/B 計測する。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

A, B, C, D, E = (
    fifth._WL_A,
    fifth._WL_B,
    fifth._WL_C,
    fifth._WL_D,
    fifth._WL_E,
)
E0 = fifth._WL_E0


def stabilize_nocompact(ctx, partner, flip):
    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    por = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    pc = glued * C
    tag = glued * D + flip.astype(np.uint64) * E
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * E0 + tag
    counts = fifth._distinct_counts(h)
    nxt = ctx.np_nxt
    unstable = np.ones(len(h), dtype=bool)
    nround = 0
    while unstable.any():
        nround += 1
        nh = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        nc = fifth._distinct_counts(nh)
        done = unstable & (nc == counts)
        h[done] = nh[done]
        unstable &= ~done
        h[unstable] = nh[unstable]
        counts = np.where(done, counts, nc)
    return h, counts, nround


def stabilize_compact(ctx, partner, flip):
    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    por = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    pc = glued * C
    tag = glued * D + flip.astype(np.uint64) * E
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * E0 + tag
    counts = fifth._distinct_counts(h)
    nxt = ctx.np_nxt
    nround = 0
    while True:
        nround += 1
        nh = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        nc = fifth._distinct_counts(nh)
        done = nc == counts
        h[done] = nh[done]
        keep = ~done
        if keep.all():
            return h, counts, nround
        h = h[keep]
        por = por[keep]
        pc = pc[keep]
        tag = tag[keep]
        counts = nc[keep]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "prism7"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    print(f"{name}: trees={len(trees)} window={window}")

    import itertools

    for label, fn in (("nocompact", stabilize_nocompact), ("compact", stabilize_compact)):
        t0 = time.perf_counter()
        rounds_max = 0
        rest = iter(trees)
        while True:
            w = list(itertools.islice(rest, window))
            if not w:
                break
            pos = np.asarray(
                [[ctx.adj_pos_of[e] for e in t] for t in w], dtype=np.int64
            )
            partner, flip = fifth._glue_batch(ctx, pos)
            _h, _c, nr = fn(ctx, partner, flip)
            rounds_max = max(rounds_max, nr)
        el = time.perf_counter() - t0
        print(f"  {label:10s}: {el:7.3f}s  (max rounds={rounds_max})")


if __name__ == "__main__":
    main()