"""同型な木を「同じ窓」と「別々の窓」で処理して batch ラベルの一致を比べる。"""

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

from collections import defaultdict

classes = defaultdict(list)
for i, t in enumerate(trees):
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    classes[min(s, ms)].append(i)
big = max(classes.values(), key=len)
i1, i2 = big[0], big[1]
print(f"isomorphic pair: #{i1} {trees[i1]}  /  #{i2} {trees[i2]}")


def run(tlist):
    pos = np.asarray([[ctx.adj_pos_of[e] for e in t] for t in tlist], dtype=np.int64)
    p, fl = fifth._glue_batch(ctx, pos)
    return fifth._batch_stabilize(ctx, p, fl)


# 1) 同じ窓
h2, c2 = run([trees[i1], trees[i2]])
print("same window: counts =", c2.tolist(), " labels_equal =",
      bool((h2[0] == h2[1]).all()))

# 2) 別の窓（1行ずつ）
h1a, c1a = run([trees[i1]])
h1b, c1b = run([trees[i2]])
print("separate windows: counts =", c1a[0], c1b[0], " labels_equal =",
      bool((h1a[0] == h1b[0]).all()))

# 3) 窓の途中（ダミー行を前後に置く）
h3, c3 = run([trees[0], trees[i1], trees[2], trees[i2]])
print("with neighbours: counts =", c3.tolist(), " row1==row3:",
      bool((h3[1] == h3[3]).all()))

# ラベルの値を見比你する
print("same-window label sample :", sorted(int(x) for x in h2[0])[:3])
print("separate   label sample :", sorted(int(x) for x in h1a[0])[:3])

# 4) 収束ラウンドの記録（どこで凍結したか）
A, B, C, D, E, E0 = (
    fifth._WL_A, fifth._WL_B, fifth._WL_C, fifth._WL_D, fifth._WL_E, fifth._WL_E0
)
n = ctx.n_nodes
nxt = ctx.np_nxt
uniq = np.unique(ctx.np_edge_len)
eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
for label, t in (("A", trees[i1]), ("B", trees[i2])):
    pos = np.asarray([[ctx.adj_pos_of[e] for e in t]], dtype=np.int64)
    p, fl = fifth._glue_batch(ctx, pos)
    g = (p >= 0).astype(np.uint64)
    por = np.where(p >= 0, p, np.arange(n)[None, :])
    pc = g * C
    tag = g * D + fl.astype(np.uint64) * E
    h = (eid * E0)[None, :] + tag
    counts = fifth._distinct_counts(h)
    for rnd in range(1, 12):
        nh = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        nc = fifth._distinct_counts(nh)
        stop = bool((nc == counts)[0])
        h, counts = nh, nc
        print(f"  tree{label} round {rnd}: colors={nc[0]} stop={stop}")
        if stop:
            break
    print(f"  tree{label} final label digest:", sorted(int(x) for x in h[0])[:2])