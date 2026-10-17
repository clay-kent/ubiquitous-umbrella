"""案A（枝刈り）の実現可能性を測る。

問: BFS で面を配置している途中（全部置き終わる前）に、
   「この木は最終的に重なりを持つ」ことが多少次早く検出できるか。

検出が常に最後（全部置いてから）なら、枝刈りは効かない。
途中で検出できるなら、重複した木の多くを途中で切れる。

出力:
  - 木ごとに「何枚目を置いた時点で重なりが確定したか」の分布
  - 全置換が要る木 / 途中で判断できる木 の比率
"""

import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS


def early_overlap_depth(vertices, faces, normals, tree, prep):
    """BFS で順に面を配置し、初めて重なりが確定したときの「配置済み面数」を返す。
    最後まで確定しなければ n_faces を返す。"""
    shared_of = prep["shared_of"]
    origin, u_x, u_y = prep["origin"], prep["u_x"], prep["u_y"]
    face_pts = prep["face_pts"]
    n_faces = len(faces)

    _parents, pairs = fifth._tree_bfs(n_faces, tree)
    eye, zero = np.eye(3), np.zeros(3)
    rot = [eye] * n_faces
    off = [zero] * n_faces
    placed = [False] * n_faces
    placed[0] = True

    polys = []
    n_placed = 1

    def make_poly(i):
        pts = face_pts[i]
        axis_x = rot[i].T @ u_x
        axis_y = rot[i].T @ u_y
        shift = off[i] - origin
        cx = pts @ axis_x + float(shift @ u_x)
        cy = pts @ axis_y + float(shift @ u_y)
        return np.column_stack([cx, cy])

    polys.append(make_poly(0))

    for u, v in pairs:
        edge = shared_of[(u, v)]
        p_a, p_b = vertices[edge[0]], vertices[edge[1]]
        k = (p_b - p_a) / np.linalg.norm(p_b - p_a)
        kx, ky, kz = k.tolist()
        nx_, ny_, nz_ = normals[u].tolist()
        mx, my, mz = normals[v].tolist()
        c = mx * nx_ + my * ny_ + mz * nz_
        s = ((ky * mz - kz * my) * nx_ + (kz * mx - kx * mz) * ny_
             + (kx * my - ky * mx) * nz_)
        v1c = 1.0 - c
        r = np.array([
            [kx * kx * v1c + c, kx * ky * v1c - kz * s, kx * kz * v1c + ky * s],
            [ky * kx * v1c + kz * s, ky * ky * v1c + c, ky * kz * v1c - kx * s],
            [kz * kx * v1c - ky * s, kz * ky * v1c + kx * s, kz * kz * v1c + c],
        ])
        rot[v] = rot[u] @ r
        off[v] = rot[u] @ (p_a - r @ p_a) + off[u]
        placed[v] = True
        polys.append(make_poly(v))
        n_placed += 1

        # 新しい面と、すでに置いた面すべての重なりを見る
        counts = [len(p) for p in polys]
        geoms = shapely.polygons(shapely.linearrings(
            np.concatenate(polys, axis=0),
            indices=np.repeat(np.arange(len(polys)), counts),
        ))
        if not fifth.is_valid_layout(list(geoms)):
            return n_placed
    return n_faces


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    prep = fifth.prepare_layout(v, faces, normals)
    edges = fifth.build_adjacency_graph(faces)
    n_faces = len(faces)

    trees = list(itertools.islice(fifth.enumerate_spanning_trees(edges, n_faces), limit))
    hist = Counter()
    early = 0
    for tree in trees:
        d = early_overlap_depth(v, faces, normals, tree, prep)
        hist[d] += 1
        if d < n_faces:
            early += 1
    print(f"{name}: faces={n_faces} sampled={len(trees)}")
    print(f"  重なりが途中で確定した木: {early}/{len(trees)} ({100*early/len(trees):.0f}%)")
    print(f"  確定位置（配置済み面数）の分布:")
    for d in sorted(hist):
        mark = "  <- 全部置いてから" if d == n_faces else ""
        print(f"    {d:3d}/{n_faces} : {hist[d]:5d}{mark}")


if __name__ == "__main__":
    main()