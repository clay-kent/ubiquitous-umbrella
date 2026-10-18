"""layout_tree の投影を「1面ずつ」vs「まとめて」で計算し、ビット一致を確かめる。

ビット一致しなければ採用しない（Cycle 5 のバグ2 = 1 ULP ずれで tie が反転、
を繰り返さないため）。matmul は GEMV と GEMM で丸め順が変わりうるので、
実際の立体・実際の木で `array_equal` を確認する。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def project_per_face(prep, rot, off, n_faces):
    origin = prep["origin"]
    u_x = prep["u_x"]
    u_y = prep["u_y"]
    face_pts = prep["face_pts"]
    xs, ys = [], []
    for i in range(n_faces):
        face_rot, t = rot[i], off[i]
        axis_x = face_rot.T @ u_x
        axis_y = face_rot.T @ u_y
        shift = t - origin
        pts = face_pts[i]
        xs.append(pts @ axis_x + float(shift @ u_x))
        ys.append(pts @ axis_y + float(shift @ u_y))
    return np.concatenate(xs), np.concatenate(ys)


def project_batched(prep, rot, off, n_faces):
    """rot/off を (F,3,3)/(F,3) にまとめ、軸も (3,2) で一括計算する案。"""
    origin = prep["origin"]
    u_x = prep["u_x"]
    u_y = prep["u_y"]
    face_pts = prep["face_pts"]
    U = np.column_stack([u_x, u_y])  # (3,2)
    R = np.stack([r for r in rot])  # (F,3,3)
    O = np.stack([o for o in off])  # (F,3)
    axes = np.einsum("fji,jk->fik", R, U)  # (F,3,2) = (rot.T) @ U
    consts = (O - origin) @ U  # (F,2)
    xs, ys = [], []
    for i in range(n_faces):
        pts = face_pts[i]
        c = pts @ axes[i] + consts[i]  # (m,2)
        xs.append(c[:, 0])
        ys.append(c[:, 1])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    cases = ["box", "frustum4", "prism5", "lshape"]
    for name in cases:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        edges = fifth.build_adjacency_graph(faces)
        n_faces = len(faces)
        mismatch = 0
        total = 0
        for tree in fifth.enumerate_spanning_trees(edges, n_faces):
            # rot/off を取り出すために layout_tree の中身を再現する
            fold_of = prep["fold_of"]
            parents, pairs = fifth._tree_bfs(n_faces, tree)
            eye = np.eye(3)
            zero = np.zeros(3)
            rot = [eye] * n_faces
            off = [zero] * n_faces
            rot[0] = eye
            for u, vtx in pairs:
                r, w = fold_of[(u, vtx)]
                rot[vtx] = rot[u] @ r
                off[vtx] = rot[u] @ w + off[u]
            x1, y1 = project_per_face(prep, rot, off, n_faces)
            x2, y2 = project_batched(prep, rot, off, n_faces)
            total += 1
            if not (np.array_equal(x1, x2) and np.array_equal(y1, y2)):
                mismatch += 1
        print(
            f"{name:9s} trees={total:5d} bit-mismatch={mismatch} "
            f"{'BIT-IDENTICAL' if mismatch == 0 else 'DIFFERS'}"
        )


if __name__ == "__main__":
    main()
