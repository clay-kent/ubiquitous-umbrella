"""is_valid_layout が呼ぶ intersection のうち、
「木で隣接する面どうし（= 折り線を共有し、重なり面積が厳密に 0）」の割合を測る。

これらは**幾何を計算するまでもなく非重複**と分かるので、スキップできれば
intersection 呼び出しを減らせる。どれくらい減るかを実測する。
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
        ctx = fifth.RibbonContext(v, faces)
        mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
        seen = set()
        rest = fifth.enumerate_spanning_trees(edges, len(faces))
        total_bbox = 0
        total_tree_adj = 0
        total_nonadj = 0
        nets = 0
        for chunk in iter(lambda: list(itertools.islice(rest, 2048)), []):
            for tree, (a, b) in zip(
                chunk, fifth._batch_canonical_keys(ctx, mctx, chunk)
            ):
                k = min(a, b)
                if k in seen:
                    continue
                seen.add(k)
                polys = fifth.layout_tree(v, faces, normals, tree, prep)
                nets += 1
                # tree の辺（面ペア、両向き）を集合に
                tedges = set()
                for u, w in tree:
                    tedges.add((u, w))
                    tedges.add((w, u))
                n = len(polys)
                arr = shapely.bounds(np.asarray(polys, dtype=object))
                for i in range(n):
                    for j in range(i + 1, n):
                        if (
                            arr[i, 2] < arr[j, 0]
                            or arr[j, 2] < arr[i, 0]
                            or arr[i, 3] < arr[j, 1]
                            or arr[j, 3] < arr[i, 1]
                        ):
                            continue
                        total_bbox += 1
                        if (i, j) in tedges:
                            total_tree_adj += 1
                        else:
                            total_nonadj += 1
            if nets >= 2500:
                break
        print(
            f"{name:8s} nets={nets:4d} bbox_pairs={total_bbox:6d} "
            f"tree_adjacent={total_tree_adj:6d} "
            f"({total_tree_adj / max(total_bbox, 1) * 100:4.1f}%) "
            f"nonadj={total_nonadj:6d} "
            f"per_net_nonadj={total_nonadj / nets:.1f}"
        )


if __name__ == "__main__":
    main()
