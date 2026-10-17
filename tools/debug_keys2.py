"""batch の色ラベルが同型木間で一致するかを1木ずつ検査する。"""

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


v, f = cube()
faces = fifth.orient_faces_outward(v, f)
ctx = fifth.RibbonContext(v, faces)
trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))

# 参照: 木ごとのWLラベル(hash版)
ref_sig = [fifth._net_signature(ctx, list(t)) for t in trees]
ref_classes = {}
for i, s in enumerate(ref_sig):
    ref_classes.setdefault(s, []).append(i)
print(f"ref classes: {len(ref_classes)}")

# batch: 1行ずつ(窓1)
pos_all = [
    [ctx.adj_pos_of[(u, w)] for u, w in t] for t in trees
]
rows_labels = []
for p in pos_all:
    bp, bf = fifth._glue_batch(ctx, np.asarray([p], dtype=np.int64))
    c, cnt = fifth._batch_stabilize(ctx, bp, bf)
    rows_labels.append(tuple(c[0].tolist()))
print(f"batch labels distinct (window=1): {len(set(rows_labels))}")

# 参照クラスの中で batch ラベルが割れているクラスを探す
bad = 0
for s, idxs in ref_classes.items():
    labs = {rows_labels[i] for i in idxs}
    if len(labs) != 1:
        bad += 1
        if bad <= 3:
            print(f"inconsistent class size={len(idxs)}: batch label variants={len(labs)}")
print(f"inconsistent ref classes: {bad}/{len(ref_classes)}")

# 安定化後の**パーティション**（どの半辺同士が同色か）が一致しているか
def partition(labels):
    m = {}
    for i, c in enumerate(labels):
        m.setdefault(c, []).append(i)
    return frozenset(frozenset(v) for v in m.values())


def ref_partition_from_sig(sig):
    return partition(sig)


bad_p = 0
for s, idxs in ref_classes.items():
    parts = {partition(rows_labels[i]) for i in idxs}
    if len(parts) != 1:
        bad_p += 1
print(f"inconsistent partitions: {bad_p}/{len(ref_classes)}")

# 1行だけ手動で Dynamics を追って見る（最初の木）
tree = trees[0]
rp, rf = fifth._glue_arrays(ctx, list(tree))
init = [(ctx.edge_len[i], rp[i] >= 0, rf[i]) for i in range(ctx.n_nodes)]
ref_lab = fifth._wl_refine(ctx.nxt, rp, rf, init)
bp, bf = fifth._glue_batch(ctx, np.asarray([pos_all[0]], dtype=np.int64))
print("glue match:", bp[0].tolist() == rp, bf[0].tolist() == rf)
b_lab = rows_labels[0]
print("ref partition:", ref_partition_from_sig(ref_lab))
print("batch partition:", partition(b_lab))