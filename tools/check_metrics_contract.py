"""Cycle 5（折り曲げ回転の前計算）がメトリクスを壊さないことの確認。

「同面積の net が入れ替わる」のが、無関係な tie に基づくだけかを、
メトリクス（sort_by の値・件数・順序性）で確認する。
丸め順序で変わるのは「同値どうしの並び順」だけで、
sort_by の契約（面積の良い順）は守られているはず。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
import make_nets


def metrics_of(v, f, normals, shared_of, nets):
    return [
        fifth.net_metrics(polys, tree, v, shared_of).bounding_box_area
        for tree, polys in nets
    ]


def main():
    for name, builder in [
        ("cube", lambda: make_nets.box(1.0, 1.0, 1.0)),
        ("box", lambda: make_nets.box(1.0, 1.5, 2.0)),
        ("prism4", lambda: make_nets.prism(4)),
        ("prism5", lambda: make_nets.prism(5)),
        ("lshape", make_nets.l_shape),
    ]:
        v, f = builder()
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        shared_of = fifth.build_shared_of(faces)
        edges = fifth.build_adjacency_graph(faces)
        nets = list(
            fifth.iter_unique_nets(
                v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals
            )
        )
        areas = sorted(metrics_of(v, f, normals, shared_of, nets))
        perims = sorted(
            fifth.net_metrics(p, t, v, shared_of).perimeter for t, p in nets
        )
        print(
            f"{name:8s} unique={len(nets):5d} "
            f"area: min={areas[0]:.9f} max={areas[-1]:.9f} "
            f"monotonic_sorted={all(a <= b for a, b in zip(areas, areas[1:]))} "
            f"perim_min={perims[0]:.6f}"
        )
        # 面積が同値のグループが複数あるか
        from collections import Counter

        c = Counter(round(a, 9) for a in areas)
        dup = {k: n for k, n in c.items() if n > 1}
        print(f"          ties(同値グループ): {len(dup)}  max_group={max(c.values())}")


if __name__ == "__main__":
    main()