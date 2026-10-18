"""batch 力学と参照力学をラウンドごとに突き合わせる（cube）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth


def cube():
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


def partition(labels):
    m = {}
    for i, c in enumerate(labels):
        m.setdefault(c, []).append(i)
    return tuple(sorted(tuple(sorted(v)) for v in m.values()))


v, f = cube()
faces = fifth.orient_faces_outward(v, f)
ctx = fifth.RibbonContext(v, faces)
trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))
pos = [[ctx.adj_pos_of[(u, w)] for u, w in t] for t in trees]

# batch の分割と ref が一致しない木を1つ見つける
for t_i, tree in enumerate(trees):
    bp, bf = fifth._glue_batch(ctx, np.asarray([pos[t_i]], dtype=np.int64))
    c, cnt = fifth._batch_stabilize(ctx, bp, bf)
    b_lab = tuple(c[0].tolist())
    rp, rf = fifth._glue_arrays(ctx, list(tree))
    init = [(ctx.edge_len[i], rp[i] >= 0, rf[i]) for i in range(ctx.n_nodes)]
    ref = fifth._wl_refine(ctx.nxt, rp, rf, init)
    if partition(b_lab) != partition(ref):
        print(f"inconsistent tree {t_i}: {tree}")
        print("  ref  partition:", partition(ref))
        print("  batch partition:", partition(b_lab))
        print("  ref  labels:", ref)
        print("  batch labels:", list(b_lab))
        break

# ラウンドごとの comparing
print()
print("--- round-by-round trace ---")
n = ctx.n_nodes
tree = trees[0]
rp, rf = fifth._glue_arrays(ctx, list(tree))
init = [(ctx.edge_len[i], rp[i] >= 0, rf[i]) for i in range(n)]

# ref round 1..
ref_colors = list(init)
ref_parts = []
for r in range(30):
    new = [
        hash(
            (
                ref_colors[i],
                ref_colors[ctx.nxt[i]],
                None if rp[i] < 0 else (rf[i], ref_colors[rp[i]]),
            )
        )
        for i in range(n)
    ]
    if len(set(new)) == len(set(ref_colors)):
        ref_parts.append((r + 1, partition(new), "STOP"))
        break
    ref_colors = new
    ref_parts.append((r + 1, partition(new), ""))
print("ref rounds:")
for r, p, note in ref_parts:
    print(f"  round {r}: ncolors={len(p)} {note}")

# batch round-by-round (手動)
bp, bf = fifth._glue_batch(ctx, np.asarray([pos[0]], dtype=np.int64))
glued = bp >= 0
por = np.where(glued, bp, np.arange(n)[None, :])
span = 2 * (n + 1)
uniq = np.unique(ctx.np_edge_len)
eid = np.searchsorted(uniq, ctx.np_edge_len)
code = glued.astype(np.int64) * 2 + bf.astype(np.int64) + (eid * 4 + 1)[None, :]
colors, counts = fifth._rank_relabel(code)
print("batch rounds:")
print(f"  init: {partition(colors[0].tolist())}")
for r in range(30):
    c_nxt = colors[:, ctx.np_nxt]
    c_partner = np.take_along_axis(colors, por, axis=1)
    companion = np.where(glued, (c_partner + 1) * 2 + bf.astype(np.int64), 0)
    keys = (colors * span + c_nxt) * span + companion
    new_colors, new_counts = fifth._rank_relabel(keys)
    stop = new_counts[0] == counts[0]
    print(f"  round {r+1}: ncolors={new_counts[0]} {'STOP' if stop else ''} "
          f"{partition(new_colors[0].tolist())}")
    colors = np.where(stop, colors, new_colors)
    if stop:
        break