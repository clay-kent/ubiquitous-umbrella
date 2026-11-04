"""全立体の「一意な展開図」の指標 (面積, 外周) を 9 桁丸めでハッシュする。

座標の丸め順を変える高速化では、`hash_layout_bits.py`（厳密一致）は当然ずれる。
その代わりに **API golden と同じ 9 桁丸め** で指標の多重集合が一致することを見る。
一致すれば「観測できる差はない（差は 9 桁より下の ULP のみ）」と言える。

    uv run python tools/hash_metrics_rounded.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    names = sys.argv[1:] or ["box", "frustum4", "prism5", "lshape", "prism6"]
    for name in names:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        shared_of = prep["shared_of"]
        edges = fifth.build_adjacency_graph(faces)
        trees = fifth.enumerate_spanning_trees(edges, len(faces))
        items = []
        for tree, polys in fifth.iter_unique_nets(v, faces, trees, normals):
            m = fifth.net_metrics(polys, tree, v, faces, normals, shared_of)
            items.append(
                (
                    round(m.bounding_box_area, 9),
                    round(m.perimeter, 9),
                    round(m.bounding_box_width, 9),
                    round(m.bounding_box_height, 9),
                )
            )
        items.sort()
        h = hashlib.sha256(repr(items).encode()).hexdigest()
        print(f"{name:9s} nets={len(items):6d} sha256={h[:24]}")


if __name__ == "__main__":
    main()
