"""ctx と mctx のキー値集合の重なりを調べる（理論: 同じ20値になるはず）。"""

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

print("np_nxt equal:", np.array_equal(ctx.np_nxt, mctx.np_nxt))
print("adj_nu equal :", np.array_equal(ctx.adj_nu, mctx.adj_nu))
print("adj_fl equal :", np.array_equal(ctx.adj_fl, mctx.adj_fl))
print("adj_nv equal :", np.array_equal(ctx.adj_nv, mctx.adj_nv))

trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), 6))
pos = np.asarray(
    [[ctx.adj_pos_of[tuple(sorted(e))] for e in t] for t in trees], dtype=np.int64
)

pc, fc = fifth._glue_batch(ctx, pos)
pm, fm = fifth._glue_batch(mctx, pos)
hc, cc = fifth._batch_stabilize(ctx, pc, fc)
hm, mc = fifth._batch_stabilize(mctx, pm, fm)

kc = {tuple(sorted(hc[i].tolist())) for i in range(len(trees)) if cc[i] == 24}
km = {tuple(sorted(hm[i].tolist())) for i in range(len(trees)) if mc[i] == 24}
print("ctx distinct (plain rows):", len(kc))
print("mctx distinct (plain rows):", len(km))
print("overlap:", len(kc & km))

# 参照版
rk_c = {fifth._net_signature(ctx, list(t)) for t in trees}
rk_m = {fifth._net_signature(mctx, list(t)) for t in trees}
print("ref ctx distinct:", len(rk_c), " ref mctx distinct:", len(rk_m),
      " overlap:", len(rk_c & rk_m))

# 同一木で ctx と mctx の-glue が一致するもの（自己鏡像の証拠）
same = 0
for i in range(len(trees)):
    if np.array_equal(pc[i], pm[i]) and np.array_equal(fc[i], fm[i]):
        same += 1
print("trees with identical glue in both ctx:", same)