"""重なり判定コストの上限構造を見る。

交差計算が呼ばれる回数は「bbox が重なるペア数」で決まる。
これが面数に対してどう増えるかを測ると、どこに天井があるか見える。
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS


def main():
    for name in ["prism7", "lshape", "stairs"]:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        edges = fifth.build_adjacency_graph(faces)

        # 一意な展開図のサンプルを取得（重複キー除去はスキップして近似）
        nets = []
        rest = fifth.enumerate_spanning_trees(edges, len(faces))
        seen = set()
        ctx = fifth.RibbonContext(v, faces)
        mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
        for chunk in iter(lambda: list(itertools.islice(rest, 2048)), []):
            for tree, (a, b) in zip(chunk, fifth._batch_canonical_keys(ctx, mctx, chunk)):
                k = min(a, b)
                if k in seen:
                    continue
                seen.add(k)
                polys = fifth.layout_tree(v, faces, normals, tree, prep)
                nets.append(polys)
            if len(nets) >= 400:
                break

        n_pairs_total = 0
        n_bbox_hit = 0
        n_valid = 0
        for polys in nets:
            n = len(polys)
            ii, jj = np.triu_indices(n, k=1)
            n_pairs_total += len(ii)
            arr = shapely.bounds(np.asarray(polys, dtype=object))
            hit = ~(
                (arr[ii, 2] < arr[jj, 0])
                | (arr[jj, 2] < arr[ii, 0])
                | (arr[ii, 3] < arr[jj, 1])
                | (arr[jj, 3] < arr[ii, 1])
            )
            n_bbox_hit += int(hit.sum())
            if fifth.is_valid_layout(polys):
                n_valid += 1
        print(
            f"{name:8s} faces={len(faces):2d} sampled_nets={len(nets):4d} "
            f"pairs/net={n_pairs_total / len(nets):5.1f} "
            f"bbox_hit/net={n_bbox_hit / len(nets):5.1f} "
            f"({n_bbox_hit / n_pairs_total * 100:4.1f}%) valid={n_valid}"
        )


if __name__ == "__main__":
    main()