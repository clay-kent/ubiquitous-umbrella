"""cube で batch と参照実装のキーを比較するデバッグ用スクリプト。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth


def cube(w=1.0, h=1.0, d=1.0):
    v = np.array(
        [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
         [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]], dtype=float
    )
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


v, f = cube()
faces = fifth.orient_faces_outward(v, f)
ctx = fifth.RibbonContext(v, faces)
trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))
print(f"trees={len(trees)}  half_edges={ctx.n_nodes}")

pairs = fifth._batch_canonical_keys(ctx, ctx, trees)
keys = {min(a, b) for a, b in pairs}
refs = [fifth._net_signature(ctx, list(t)) for t in trees]
ref_keys = {min(s, m) for s, m in
            ((fifth._net_signature(ctx, list(t)), fifth._net_signature(ctx, list(t))) for t in trees)}
print(f"batch keys: {len(keys)} (期待 11, 参照 {len(ref_keys)})")

# 同型だったはずの木が違うキーを持つ例を探す
from collections import Counter
cnt = Counter(min(a, b) for a, b in pairs)
repeated = [(k, c) for k, c in cnt.items() if c > 21]
print("頻出キー（☞ 384/11 = 平均 34.9）:", len(repeated))

# 最初の木2つ mini trace: partner/flip が参照と一致するか
for t_i in [0, 1]:
    tree = trees[t_i]
    positions = np.asarray([[ctx.adj_pos_of[(u, v)] for u, v in tree]], dtype=np.int64)
    bp, bf = fifth._glue_batch(ctx, positions)
    rp, rf = fifth._glue_arrays(ctx, list(tree))
    print(f"tree{t_i} partner一致: {bp[0].tolist() == rp}  flip一致: {bf[0].tolist() == rf}")

# 安定化の一致
for t_i in [0, 1]:
    tree = trees[t_i]
    positions = np.asarray([[ctx.adj_pos_of[(u, v)] for u, v in tree]], dtype=np.int64)
    bp, bf = fifth._glue_batch(ctx, positions)
    bc, bcnt = fifth._batch_stabilize(ctx, bp, bf)
    rp, rf = fifth._glue_arrays(ctx, list(tree))
    init = [(ctx.edge_len[i], rp[i] >= 0, rf[i]) for i in range(ctx.n_nodes)]
    ref = fifth._wl_refine(ctx.nxt, rp, rf, init)
    print(
        f"tree{t_i}: batch色数={bcnt[0]}  ref色数={len(set(ref))}  "
        f"batchパーティション=refパーティション: "
        f"{fifth.PARTITION_CMP_PASS if False else 'n/a'}"
    )
