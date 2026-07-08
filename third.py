import itertools
import json

import networkx as nx
import numpy as np
from shapely.geometry import Polygon


def build_adjacency_graph(faces: list[list[int]]) -> list[tuple[int, int]]:
    """面の隣接グラフの辺リストを構築する。

    2つの面がちょうど2つの頂点を共有している場合に隣接とみなす。
    """
    edges = []
    n_faces = len(faces)
    for i in range(n_faces):
        for j in range(i + 1, n_faces):
            if len(set(faces[i]) & set(faces[j])) == 2:
                edges.append((i, j))
    return edges


def enumerate_spanning_trees(
    edges: list[tuple[int, int]], num_nodes: int
) -> list[list[tuple[int, int]]]:
    """すべての全域木を列挙する。"""
    all_trees = []
    for edge_subset in itertools.combinations(edges, num_nodes - 1):
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(edge_subset)
        if nx.is_tree(g):
            all_trees.append(list(edge_subset))
    return all_trees


def compute_normals(vertices: np.ndarray, faces: list[list[int]]) -> list[np.ndarray]:
    """各面の外向き法線ベクトルを計算する。

    各面の頂点が反時計回り(CCW)に並んでいることを前提とする。
    """
    normals = []
    for face in faces:
        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]
        n = np.cross(p1 - p0, p2 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            raise ValueError("退化した面（面積ゼロ）が検出されました。")
        normals.append(n / norm)
    return normals


def layout_tree(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    tree_edges: list[tuple[int, int]],
) -> list[Polygon]:
    """全域木に基づいて面を展開し、ShapelyのPolygonリスト（2D座標系）を返す。"""
    n_faces = len(faces)
    adj = [[] for _ in range(n_faces)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    # 各面をルート面(面0)の平面へ展開するための4x4変換行列
    m_matrices = [None] * n_faces
    m_matrices[0] = np.eye(4)
    queue = [0]
    visited = {0}

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v in visited:
                continue

            # 面u と 面v で共有している辺を探索 (面uの巡回順での進行方向 A -> B)
            shared = set(faces[u]) & set(faces[v])
            n_u = len(faces[u])
            a_idx, b_idx = None, None
            for i in range(n_u):
                if faces[u][i] in shared and faces[u][(i + 1) % n_u] in shared:
                    a_idx = faces[u][i]
                    b_idx = faces[u][(i + 1) % n_u]
                    break

            if a_idx is None or b_idx is None:
                continue

            p_a = vertices[a_idx]
            p_b = vertices[b_idx]

            # 1. 回転軸 k (共有辺の正規化ベクトル)
            edge_vec = p_b - p_a
            k = edge_vec / np.linalg.norm(edge_vec)

            # 2. 両面の法線から、回転角度(cos, sin)を算出
            nu = normals[u]
            nv = normals[v]
            c = np.dot(nv, nu)
            s = np.dot(np.cross(k, nv), nu)

            # 3. ロドリゲスの回転公式による3x3回転行列
            kx, ky, kz = k
            v_1c = 1 - c
            r_rot = np.array(
                [
                    [
                        kx * kx * v_1c + c,
                        kx * ky * v_1c - kz * s,
                        kx * kz * v_1c + ky * s,
                        0,
                    ],
                    [
                        ky * kx * v_1c + kz * s,
                        ky * ky * v_1c + c,
                        ky * kz * v_1c - kx * s,
                        0,
                    ],
                    [
                        kz * kx * v_1c - ky * s,
                        kz * ky * v_1c + kx * s,
                        kz * kz * v_1c + c,
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )

            # 4. 回転中心を p_a に合わせた4x4アフィン変換行列
            t1 = np.eye(4)
            t1[0:3, 3] = -p_a
            t2 = np.eye(4)
            t2[0:3, 3] = p_a
            transform = t2 @ r_rot @ t1

            # 親の変換行列に累積
            m_matrices[v] = m_matrices[u] @ transform

            visited.add(v)
            queue.append(v)

    # 面0 の平面上における 2D ローカル座標系の定義
    origin = vertices[faces[0][0]]
    u_x = vertices[faces[0][1]] - origin
    u_x = u_x / np.linalg.norm(u_x)
    u_z = normals[0]
    u_y = np.cross(u_z, u_x)

    polygons = []
    for i in range(n_faces):
        pts_2d = []
        for v_idx in faces[i]:
            p3d_h = np.append(vertices[v_idx], 1.0)
            p_trans = (m_matrices[i] @ p3d_h)[:3]

            # 2D平面(X-Y)への射影
            x2d = np.dot(p_trans - origin, u_x)
            y2d = np.dot(p_trans - origin, u_y)
            pts_2d.append((x2d, y2d))
        polygons.append(Polygon(pts_2d))

    return polygons


def is_valid_layout(polygons: list[Polygon]) -> bool:
    """展開図のポリゴン間に重なり（衝突）がないかを検証する。"""
    n_polys = len(polygons)
    for i in range(n_polys):
        for j in range(i + 1, n_polys):
            # 浮動小数点誤差を考慮。交差面積が微小以上なら衝突とみなす
            if polygons[i].intersection(polygons[j]).area > 1e-4:
                return False
    return True


def normalize_key(polygons: list[Polygon]) -> str:
    """展開図の形状を、回転・反転・平行移動に対して正規化し、一意な文字列キーを生成する。"""
    centers = [(p.centroid.x, p.centroid.y) for p in polygons]

    # 平面上の等長変換（回転・反転の8パターン）
    transforms = [
        lambda x, y: (x, y),
        lambda x, y: (x, -y),
        lambda x, y: (-x, y),
        lambda x, y: (-x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, x),
        lambda x, y: (y, -x),
        lambda x, y: (-y, -x),
    ]
    min_key = None
    for tf in transforms:
        transformed = [tf(x, y) for x, y in centers]
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        # 少数第4位で丸めて微小な数値誤差を吸収
        shifted = [(round(x - min_x, 4), round(y - min_y, 4)) for x, y in transformed]
        shifted.sort()
        key = json.dumps(shifted)
        if min_key is None or key < min_key:
            min_key = key
    return min_key


# ==========================================
# 2. メインインターフェース
# ==========================================


def generate_unfolding_nets(
    vertices: np.ndarray | list[list[float]], faces: list[list[int]]
) -> list[list[Polygon]]:
    """与えられた3D多面体の頂点と面情報から、重複のない有効な展開図のリスト（各展開図はShapely Polygonのリスト）を生成する。

    Args:
        vertices: (V, 3)の頂点座標配列またはリスト。
        faces: 各面の頂点インデックスのリスト（外側から見て反時計回り(CCW)で統一されていること）。

    Returns:
        valid_nets: 重なりのない有効な展開図（Shapely Polygonのリスト）のリスト。
    """
    vertices = np.array(vertices, dtype=float)
    n_faces = len(faces)

    if n_faces < 4:
        raise ValueError("多面体は少なくとも4つ以上の面を持つ必要があります。")

    edges = build_adjacency_graph(faces)
    all_trees = enumerate_spanning_trees(edges, n_faces)
    normals = compute_normals(vertices, faces)

    valid_nets = []
    unique_keys = set()
    unique_nets = []

    for tree in all_trees:
        try:
            polys = layout_tree(vertices, faces, normals, tree)
            if is_valid_layout(polys):
                key = normalize_key(polys)
                if key not in unique_keys:
                    unique_keys.add(key)
                    unique_nets.append(polys)
                    valid_nets.append(polys)
        except Exception:
            # 展開処理で不正な形状や計算エラーが生じた場合はスキップ
            continue

    return unique_nets


# ==========================================
# 3. 実行
# ==========================================

if __name__ == "__main__":
    # --- 1. 立方体 ---
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    cube_faces = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 上面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 後面
        [3, 0, 4, 7],  # 左面
    ]

    # --- 2. 正四面体---
    tetra_vertices = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=float,
    )
    tetra_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]

    # --- 3. 四角錐 ---
    pyramid_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=float,
    )
    pyramid_faces = [
        [0, 3, 2, 1],  # 底面
        [0, 1, 4],  # 側面1
        [1, 2, 4],  # 側面2
        [2, 3, 4],  # 側面3
        [3, 0, 4],  # 側面4
    ]

    shapes = {
        "立方体": (cube_vertices, cube_faces),
        "正四面体": (tetra_vertices, tetra_faces),
        "四角錐": (pyramid_vertices, pyramid_faces),
    }

    for name, (verts, fcs) in shapes.items():
        nets = generate_unfolding_nets(verts, fcs)
        print(f"--- {name} ---")
        print(f"面の数: {len(fcs)}")
        print(f"回転・反転を除いた一意な展開図の数: {len(nets)}\n")
        print(nets)
