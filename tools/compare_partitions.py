"""batch 版と木ごと版の「同型クラス分割」が一致するかを直接測る。

比較する対象:
  A) 木ごと版の ctx キー  fifth._net_signature(ctx, tree)
  B) batch 版の ctx キー   _batch_stabilize の結果から作るキー

これが一致するなら ctx の同型判定は正しく、鏡像は別問題。
不一致なら、どちらの力学が分割 overloaded しているかが分かる。
"""

import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def main():
    names = sys.argv[1:] or ["box", "frustum4"]
    for name in names:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        ctx = fifth.RibbonContext(v, faces)
        edges = fifth.build_adjacency_graph(faces)
        trees = list(
            itertools.islice(fifth.enumerate_spanning_trees(edges, len(faces)), 4000)
        )
        n = ctx.n_nodes

        pos = np.asarray(
            [[ctx.adj_pos_of[e] for e in t] for t in trees], dtype=np.int64
        )
        partner, flip = fifth._glue_batch(ctx, pos)
        colors, counts = fifth._batch_stabilize(ctx, partner, flip)

        # A) 木ごと版の ctx キー
        ref_groups = defaultdict(list)
        for i, t in enumerate(trees):
            ref_groups[fifth._net_signature(ctx, list(t))].append(i)

        # B) batch 版の ctx キー（プレーン経路 = 全色別になった行）
        plain = {
            i: tuple(sorted(colors[i].tolist()))
            for i in range(len(trees))
            if counts[i] == n
        }
        batch_groups = defaultdict(list)
        for i, k in plain.items():
            batch_groups[k].append(i)

        # A のクラスが B で何クラスに割れるか
        split = [len({plain[i] for i in idxs if i in plain}) for idxs in ref_groups.values()]
        # B のクラスが A で何クラスに割れるか（過剰結合の検出）
        merged = [len({j for j, idxs in ref_groups.items() if any(i in idxs for i in rows)})
                  for rows in batch_groups.values()]

        plain_total = len(plain)
        print(f"{name}: trees={len(trees)} ref_classes={len(ref_groups)} "
              f"plain_rows={plain_total} batch_classes={len(batch_groups)}")
        print(f"   ref クラスが batch で割れる個数の分布: "
              f"{sorted(set(split))} (1なら一致)")
        over = [m for m in merged if m > 1]
        print(f"   batch クラスが ref でマージされる（過剰結合）: {len(over)}件")
        if over:
            print(f"     例: {sorted(merged, reverse=True)[:5]}")


if __name__ == "__main__":
    main()