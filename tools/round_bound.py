"""同型クラス内の凍結ラウンドが一致するかを複数の立体で確認する。

真に不変な停止条件を探すため、
  (1) 行ごとの凍結ラウンドのばらつき
  (2) 分割が完全に安定する（色数が n_nodes に到達する）ラウンドの上限
を測る。
"""

import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

A, B, C, D, E, E0 = (
    fifth._WL_A, fifth._WL_B, fifth._WL_C, fifth._WL_D, fifth._WL_E, fifth._WL_E0
)


def trace(ctx, tree, max_round=60):
    n = ctx.n_nodes
    nxt = ctx.np_nxt
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    pos = np.asarray([[ctx.adj_pos_of[e] for e in tree]], dtype=np.int64)
    p, fl = fifth._glue_batch(ctx, pos)
    g = (p >= 0).astype(np.uint64)
    por = np.where(p >= 0, p, np.arange(n)[None, :])
    pc = g * C
    tag = g * D + fl.astype(np.uint64) * E
    h = (eid * E0)[None, :] + tag
    counts = fifth._distinct_counts(h)
    freeze_round = None
    full_round = None
    for rnd in range(1, max_round + 1):
        h = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        nc = fifth._distinct_counts(h)
        if freeze_round is None and bool((nc == counts)[0]):
            freeze_round = rnd
        if full_round is None and bool((nc == n)[0]):
            full_round = rnd
        counts = nc
        if freeze_round is not None and full_round is not None:
            break
    return freeze_round, full_round, int(counts[0])


def main():
    names = sys.argv[1:] or ["box", "prism4", "prism5", "lshape"]
    for name in names:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        ctx = fifth.RibbonContext(v, faces)
        mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
        edges = fifth.build_adjacency_graph(faces)
        limit = 4000
        trees = list(itertools.islice(fifth.enumerate_spanning_trees(edges, len(faces)), limit))

        classes = defaultdict(list)
        for i, t in enumerate(trees):
            s = fifth._net_signature(ctx, list(t))
            ms = fifth._net_signature(mctx, list(t))
            classes[min(s, ms)].append(i)

        inconsistent = 0
        max_freeze = 0
        max_full = 0
        never_full = 0
        for _key, idxs in classes.items():
            fr = set()
            for i in idxs:
                fz, full, _c = trace(ctx, trees[i])
                fr.add(fz)
                if fz:
                    max_freeze = max(max_freeze, fz)
                if full is None:
                    never_full += 1
                else:
                    max_full = max(max_full, full)
            if len(fr) > 1:
                inconsistent += 1
        print(
            f"{name:8s} trees={len(trees):6d} classes={len(classes):5d} "
            f"inconsistent_freeze={inconsistent:4d}/{len(classes):4d} "
            f"max_freeze_round={max_freeze:3d} max_full_round={max_full:3d} "
            f"never_fully_refined={never_full}"
        )


if __name__ == "__main__":
    main()