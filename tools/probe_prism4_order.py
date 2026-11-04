"""prism4 の net 面積を repr で比較し、tie の並びが何で決まるかを確定する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
import make_nets


def main():
    v, f = make_nets.prism(4)
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    shared_of = fifth.build_shared_of(faces)
    edges = fifth.build_adjacency_graph(faces)
    recs = []
    for tree, polys in fifth.iter_unique_nets(
        v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals
    ):
        m = fifth.net_metrics(polys, tree, v, faces, normals, shared_of)
        recs.append((m.bounding_box_area, tree))
    # 列挙順（そのまま）
    print("enumeration order (area, first 6):")
    for i, (a, t) in enumerate(recs[:6]):
        print(f"  {i}: {a!r}")
    print("first 3 trees:", recs[0][1], "|", recs[1][1], "|", recs[2][1])
    # 面積で安定ソート
    s = sorted(recs, key=lambda r: r[0])
    print("sorted; areas 2,3 with repr and tree:")
    print("  idx1:", repr(s[1][0]), s[1][1])
    print("  idx2:", repr(s[2][0]), s[2][1])
    print("  bit-identical(2,3):", s[1][0] == s[2][0])


if __name__ == "__main__":
    main()
