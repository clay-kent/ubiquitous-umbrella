"""鏡像コンテキストと positions の対応を精査する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
import make_nets

v, f = make_nets.box(1.0, 1.0, 1.0)
faces = fifth.orient_faces_outward(v, f)
mirror_faces = [list(reversed(x)) for x in faces]
ctx = fifth.RibbonContext(v, faces)
mctx = fifth.RibbonContext(v, mirror_faces)

print("ctx.n_nodes =", ctx.n_nodes, " mctx.n_nodes =", mctx.n_nodes)
print("ctx.np_edge_len == mctx.np_edge_len:",
      np.array_equal(ctx.np_edge_len, mctx.np_edge_len))
print("ctx.adj_pos_of keys == mctx.adj_pos_of keys:",
      set(ctx.adj_pos_of) == set(mctx.adj_pos_of))
print("ctx.adj_nu == mctx.adj_nu:", np.array_equal(ctx.adj_nu, mctx.adj_nu))
print("ctx.adj_fl == mctx.adj_fl:", np.array_equal(ctx.adj_fl, mctx.adj_fl))

# 木ごと版の鏡像シグネチャと、batch 版が mirror_ctx を使うことの差を確認
trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))

# 木ごと版: ctx と mctx で同じ木を処理したときの接着表が一致するか（一致就好）
same_ctx_glue = 0
for t in trees[:20]:
    pc, fc = fifth._glue_arrays(ctx, t)
    pm, fm = fifth._glue_arrays(mctx, t)
    if np.array_equal(np.array(pc), pm) and np.array_equal(np.array(fc), fm):
        same_ctx_glue += 1
print(f"first20: ctx-glue == mctx-glue for {same_ctx_glue}/20 (立方体では多めにてもよい)")

# キー比較: 木ごと版の min(sig, mirror) と batch 版を1木ずつ突き合わせ
from collections import defaultdict

ref_groups = defaultdict(list)
for t in trees:
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    ref_groups[min(s, ms)].append(t)
print("ref key groups:", len(ref_groups))

# batch 版: 窓1で1木ずつ
bad = 0
for t in trees:
    pos = np.asarray([[ctx.adj_pos_of[tuple(sorted(e))] for e in t]], dtype=np.int64)
    keys = fifth._batch_canonical_keys(ctx, mctx, [t])
    gk = keys[0]
    # 同じ-refキーの木の中で batch キーが揃うか
    pass

# クラスごとに batch キーが揃うか
from collections import Counter

bkeys = []
for i in range(0, len(trees), 256):
    win = trees[i:i + 256]
    for t, k in zip(win, fifth._batch_canonical_keys(ctx, mctx, win)):
        bkeys.append(k)

by_ref = defaultdict(set)
for t, (a, b), g in zip(trees, bkeys, []):
    pass

for i, t in enumerate(trees):
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    by_ref[min(s, ms)].add(bkeys[i])

n_bad = sum(1 for v in by_ref.values() if len(v) > 1)
print(f"ref classes with >1 batch key: {n_bad}/{len(by_ref)}")

# 逆: batch キーが同じなのに ref キーが違う組
by_batch = defaultdict(set)
for i, t in enumerate(trees):
    s = fifth._net_signature(ctx, list(t))
    ms = fifth._net_signature(mctx, list(t))
    by_batch[bkeys[i]].add(min(s, ms))
n_over = sum(1 for v in by_batch.values() if len(v) > 1)
print(f"batch keys with >1 ref key (過結合): {n_over}/{len(by_batch)}")