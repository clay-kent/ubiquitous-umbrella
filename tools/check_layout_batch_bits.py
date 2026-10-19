"""`_layout_batch`（窓単位のバッチ展開）が `layout_tree` とビット一致するか。

木をまとめて展開すると numpy の呼び出し回数が数十倍減るが、丸めが変わると
採用できない。ここでは実立体のランダムな木について、Polygon の全座標を
ビット単位で比べる（``tools/hash_layout_bits.py`` の全数版が本番の門）。

    uv run python tools/check_layout_batch_bits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def same(polys_a: list, polys_b: list) -> bool:
    if len(polys_a) != len(polys_b):
        return False
    for pa, pb in zip(polys_a, polys_b):
        if not np.array_equal(
            np.asarray(pa.exterior.coords), np.asarray(pb.exterior.coords)
        ):
            return False
    return True


def main() -> None:
    total = 0
    bad = 0
    for name in SOLIDS:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        edges = fifth.build_adjacency_graph(faces)
        trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
        # 全木（立体が小さいので高々 8,100 本）を 512 本ずつまとめて比べる
        for start in range(0, len(trees), 512):
            chunk = trees[start : start + 512]
            batched, _bnds = fifth._layout_batch(len(faces), chunk, prep)[:2]
            for tree, polys_b in zip(chunk, batched):
                polys_a = fifth.layout_tree(v, faces, normals, tree, prep)
                total += 1
                if not same(polys_a, polys_b):
                    bad += 1
                    if bad <= 3:
                        print(f"  differs: {name} tree={tree}")
        print(f"{name:9s} trees={len(trees):6d} ok")
    print(f"checked {total} layouts, bit-differing {bad}")


if __name__ == "__main__":
    main()
