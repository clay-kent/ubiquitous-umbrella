"""バッチ WL の色ハッシュが衝突して「見かけの安定」を起こす頻度を測る。

バッチ版の色は ``A*h[i] + B*h[nxt] + C*h[partner] + ...`` という uint64 の
線形結合。線形なので早いラウンドでは値の線形従属が残り、実際のクラス数より
色数が少なく出ることがある。そうなると「分割が起きたのに色数は変わらない」
＝**見かけの安定**が起き、安定判定が早すぎる。

ここでは各ラウンドで

- ハッシュ経由の色数（``_distinct_counts(nh)``）
- タプル ``(h[i], h[nxt], h[partner], glued, flip)`` をそのまま数えた正確なクラス数

を比べ、「ハッシュのほうが少ない行」を数える。0 でなければ衝突あり。

    uv run python tools/probe_wl_collisions.py lshape
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "lshape"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    arr = np.asarray(trees, dtype=np.int64)
    positions = ctx.adj_pos_matrix[arr[:, :, 0], arr[:, :, 1]]
    partner, flip = fifth._glue_batch(ctx, positions)

    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    partner_or_self = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    partner_coef = glued * fifth._WL_C
    tag = glued * fifth._WL_D + flip.astype(np.uint64) * fifth._WL_E
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    nxt = ctx.np_nxt

    h = eid[None, :] * fifth._WL_E0 + tag
    for rnd in range(1, 15):
        hn = h[:, nxt]
        hp = np.take_along_axis(h, partner_or_self, axis=1)
        nh = h * fifth._WL_A
        nh += hn * fifth._WL_B
        nh += hp * partner_coef
        nh += tag

        hashed = fifth._distinct_counts(nh)
        hl = h.tolist()
        nl = hn.tolist()
        pl = hp.tolist()
        gl = glued.tolist()
        fl = flip.tolist()
        exact = np.fromiter(
            (
                len({(hl[b][j], nl[b][j], pl[b][j], gl[b][j], fl[b][j]) for j in range(n_nodes)})
                for b in range(len(h))
            ),
            dtype=np.int64,
            count=len(h),
        )
        collide = int((exact != hashed).sum())
        print(f"round {rnd:2d}: rows={len(h):6d} collisions={collide:4d} max_gap={int((exact - hashed).max())}")
        h = nh
    print(f"{name}: n_nodes={n_nodes} trees={len(trees)}")


if __name__ == "__main__":
    main()
