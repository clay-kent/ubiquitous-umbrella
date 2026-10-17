"""凍結ラウンドの差が本当に不変量を壊すかを最小例で確認する。

仮説: 同じ分割を持つ 2 つの同型な木が、異なるラウンドで凍結すると
キーが別々になる（= 過分割）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
import make_nets

v, f = make_nets.box(1.0, 1.0, 1.0)
faces = fifth.orient_faces_outward(v, f)
ctx = fifth.RibbonContext(v, faces)
mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), 6))

A, B, C, D, E, E0 = (
    fifth._WL_A, fifth._WL_B, fifth._WL_C, fifth._WL_D, fifth._WL_E, fifth._WL_E0
)
n = ctx.n_nodes
nxt = ctx.np_nxt
uniq = np.unique(ctx.np_edge_len)
eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)


def trace(tree, stop_when_full=True):
    """各ラウンドの色数と、凍結時のキーがどう変わるかを返す。"""
    pos = np.asarray([[ctx.adj_pos_of[e] for e in tree]], dtype=np.int64)
    p, fl = fifth._glue_batch(ctx, pos)
    g = (p >= 0).astype(np.uint64)
    por = np.where(p >= 0, p, np.arange(n)[None, :])
    pc = g * C
    tag = g * D + fl.astype(np.uint64) * E
    h = (eid * E0)[None, :] + tag
    seq = []
    for rnd in range(1, 30):
        nh = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        seq.append((rnd, fifth._distinct_counts(nh)[0], tuple(sorted(nh[0].tolist()))))
        h = nh
    return seq


# 分割が同じ（同型）な木を2本選んで、凍結ラウンドに差があるか見る
from collections import defaultdict

classes = defaultdict(list)
for i, t in enumerate(trees):
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    classes[min(s, ms)].append(i)

freeze_round_hist = defaultdict(set)
for i, t in enumerate(trees):
    seq = trace(t)
    # 分割がMax Colors に初めて到達するラウンド = 凍結の候補
    full = [rnd for rnd, cnt, _ in seq if cnt == n]
    if full:
        freeze_round_hist[min(full)].add(i)

print("cube: first-round-fully-refined distribution:")
for rnd in sorted(freeze_round_hist):
    print(f"  round {rnd}: {len(freeze_round_hist[rnd])} trees")

# 同じ同型クラスの中で凍結ラウンドがばらつく組を探す
varied = 0
for key, idxs in classes.items():
    rounds = set()
    for i in idxs:
        seq = trace(trees[i])
        full = [rnd for rnd, cnt, _ in seq if cnt == n]
        if full:
            rounds.add(min(full))
    if len(rounds) > 1:
        varied += 1
        if varied <= 3:
            print(f"class with varying freeze rounds: rounds={sorted(rounds)} size={len(idxs)}")
print(f"classes with varying freeze rounds: {varied}/{len(classes)}")