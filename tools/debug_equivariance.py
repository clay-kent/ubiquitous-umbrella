"""同型な2つの木で batch 力学のラベルがどこで割れるかを1ラウンドずつ追う。"""

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

# 参照版で同じクラスになる木を2本選ぶ
from collections import defaultdict

classes = defaultdict(list)
for i, t in enumerate(trees):
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    classes[min(s, ms)].append(i)
big = max(classes.values(), key=len)
i1, i2 = big[0], big[1]
print(f"same reference class: tree#{i1} and tree#{i2}")
print("  ", trees[i1])
print("  ", trees[i2])

A, B, C, D, E, E0 = (
    fifth._WL_A, fifth._WL_B, fifth._WL_C, fifth._WL_D, fifth._WL_E, fifth._WL_E0
)
n = ctx.n_nodes
nxt = ctx.np_nxt
uniq = np.unique(ctx.np_edge_len)
eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)

rows = []
for idx in (i1, i2):
    t = trees[idx]
    pos = np.asarray([[ctx.adj_pos_of[e] for e in t]], dtype=np.int64)
    p, fl = fifth._glue_batch(ctx, pos)
    rows.append((p[0], fl[0]))

h_list = []
counts_list = []
for p, fl in rows:
    glued = (p >= 0).astype(np.uint64)[None, :]
    por = np.where(p >= 0, p, np.arange(n)[None, :])
    pc = glued * C
    tag = glued * D + fl.astype(np.uint64)[None, :] * E
    h = (eid * E0)[None, :] + tag
    h_list.append([p.copy() for p in por])
    counts_list.append((h, por, pc, tag))

for rnd in range(25):
    outs = []
    for j, (h, por, pc, tag) in enumerate(counts_list):
        nh = h * A + h[:, nxt] * B + np.take_along_axis(h, por, axis=1) * pc + tag
        outs.append(nh)
    c0 = fifth._distinct_counts(outs[0])[0]
    c1 = fifth._distinct_counts(outs[1])[0]
    s0 = np.sort(outs[0])
    s1 = np.sort(outs[1])
    same = bool((s0 == s1).all())
    print(f"round {rnd+1}: colors={c0}/{c1} sorted_equal={same}")
    if not same:
        break
    for j in range(2):
        counts_list[j] = (outs[j],) + counts_list[j][1:]