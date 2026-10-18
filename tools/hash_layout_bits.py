"""全立体・全全域木の layout_tree 出力をビット単位でハッシュする。

高速化の前後でこのハッシュが一致すれば、展開図の座標が**完全に同一**
（丸めを含めて1ビットも変わらない）ことを保証できる。
`git stash` で前後に切り替えて使う。

    uv run python tools/hash_layout_bits.py

標準出力に立体ごとの (木数, 座標バイト列のハッシュ) を出す。
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def main():
    names = sys.argv[1:] or ["box", "frustum4", "prism5", "lshape", "prism6"]
    for name in names:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        edges = fifth.build_adjacency_graph(faces)
        h = hashlib.sha256()
        n = 0
        for tree in fifth.enumerate_spanning_trees(edges, len(faces)):
            polys = fifth.layout_tree(v, faces, normals, tree, prep)
            for p in polys:
                h.update(np.asarray(p.exterior.coords, dtype=np.float64).tobytes())
            n += 1
        print(f"{name:9s} trees={n:6d} sha256={h.hexdigest()[:24]}")


if __name__ == "__main__":
    main()
