import json
import math

import numpy as np
from shapely.geometry import Polygon

# ==================== データ定義 ====================
vertices = np.array(
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

faces = [
    {"id": 0, "indices": [0, 1, 2, 3]},  # 底面
    {"id": 1, "indices": [4, 5, 6, 7]},  # 上面
    {"id": 2, "indices": [0, 1, 5, 4]},  # 前面
    {"id": 3, "indices": [1, 2, 6, 5]},  # 右面
    {"id": 4, "indices": [2, 6, 7, 3]},  # 後面
    {"id": 5, "indices": [3, 7, 4, 0]},  # 左面
]

N = len(faces)  # 6
local_coords = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ],
    dtype=float,
)


# ==================== 2D変換行列 ====================
def translate(tx, ty):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )


def rotate(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def flip(dx, dy):
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return np.eye(3)
    ux, uy = dx / length, dy / length
    return np.array(
        [
            [ux * ux - uy * uy, 2 * ux * uy, 0],
            [2 * ux * uy, uy * uy - ux * ux, 0],
            [0, 0, 1],
        ]
    )


def apply_transform(M, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    transformed = pts_h @ M.T
    return transformed[:, :2]


# ==================== 3D空間での面の関係判定 ====================
def get_3d_relation(u, v):
    idx_u = set(faces[u]["indices"])
    idx_v = set(faces[v]["indices"])
    shared = idx_u & idx_v
    if len(shared) == 2:
        return "edge"  # 3D空間で辺を共有している
    elif len(shared) == 1:
        return "vertex"  # 3D空間で頂点のみ共有している
    return "none"  # 3D空間で全く共有していない


# ==================== 面の隣接グラフ ====================
def find_shared_edge(u, v):
    idx_u = faces[u]["indices"]
    idx_v = faces[v]["indices"]
    for i in range(4):
        a = idx_u[i]
        b = idx_u[(i + 1) % 4]
        for j in range(4):
            c = idx_v[j]
            d = idx_v[(j + 1) % 4]
            if (a == c and b == d) or (a == d and b == c):
                return (a, b)
    return None


# 全辺リスト（隣接する面ペア）
edge_list = []
for i in range(N):
    for j in range(i + 1, N):
        if find_shared_edge(i, j) is not None:
            edge_list.append((i, j))

# ==================== 全域木の列挙（手動バックトラック） ====================
import networkx as nx  # 全域木チェック用にのみ使用

all_spanning_trees = []


def backtrack(start, selected):
    if len(selected) == N - 1:
        # 選択された辺が全域木を形成するかチェック
        test_graph = nx.Graph()
        test_graph.add_nodes_from(range(N))
        test_graph.add_edges_from([edge_list[i] for i in selected])
        if nx.is_tree(test_graph):
            all_spanning_trees.append([edge_list[i] for i in selected])
        return
    remaining = len(edge_list) - start
    need = N - 1 - len(selected)
    if remaining < need:
        return
    for i in range(start, len(edge_list)):
        selected.append(i)
        backtrack(i + 1, selected)
        selected.pop()


backtrack(0, [])


# ==================== 展開図の2Dレイアウト ====================
def get_local_edge(face_idx, shared_edge):
    a, b = shared_edge
    indices = faces[face_idx]["indices"]
    for i in range(4):
        p1 = indices[i]
        p2 = indices[(i + 1) % 4]
        if p1 == a and p2 == b:
            return local_coords[i], local_coords[(i + 1) % 4]
        if p1 == b and p2 == a:
            return local_coords[i], local_coords[(i + 1) % 4]
    return None


def layout_tree(selected_edges):
    tree_adj = [[] for _ in range(N)]
    for u, v in selected_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)

    transforms = [None] * N
    transforms[0] = np.eye(3)
    face_polys = [None] * N
    face_polys[0] = apply_transform(transforms[0], local_coords)

    visited = [False] * N
    visited[0] = True
    queue = [0]

    while queue:
        u = queue.pop(0)
        Mu = transforms[u]

        for v in tree_adj[u]:
            if visited[v]:
                continue

            shared = find_shared_edge(u, v)
            edge_u = get_local_edge(u, shared)
            edge_v = get_local_edge(v, shared)
            if edge_u is None or edge_v is None:
                return None

            sU, eU = edge_u
            sV, eV = edge_v

            T1 = translate(-sV[0], -sV[1])
            vecV = np.array([eV[0] - sV[0], eV[1] - sV[1]])
            vecU = np.array([eU[0] - sU[0], eU[1] - sU[1]])
            angleV = math.atan2(vecV[1], vecV[0])
            angleU = math.atan2(vecU[1], vecU[0])
            R = rotate(angleU - angleV)
            F = flip(vecU[0], vecU[1])
            poly = apply_transform(Mv, local_coords)
            poly = np.round(poly, 10)
            face_polys[v] = poly

            visited[v] = True
            queue.append(v)

    if any(p is None for p in face_polys):
        return None
    return [np.array(p) for p in face_polys]


# ==================== 重なり判定 (Shapely) ====================
def polygons_overlap(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if p1.is_empty or p2.is_empty:
        return False
    return p1.intersects(p2) and not p1.touches(p2)


# ==================== 重なり判定 (Shapely + 3D関係性チェック) ====================
def is_valid_net(face_polys, selected_edges):
    adj = np.zeros((N, N), dtype=bool)
    for u, v in selected_edges:
        adj[u, v] = adj[v, u] = True

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                continue

            p1 = Polygon(face_polys[i])
            p2 = Polygon(face_polys[j])

            # 2D平面上で離れていれば問題ない
            if not p1.intersects(p2):
                continue

            # ここから下は、2D平面上で「何らかの形で接している（touchesまたは重なり）」状態
            rel_3d = get_3d_relation(i, j)

            if rel_3d == "none":
                # 3Dで無関係な面なのに、2Dでくっついてしまったのは明らかに無効な展開図
                return False
            else:
                # 3Dで辺や頂点を共有している場合、2Dで「面積を持って重なってしまえば」無効
                if not p1.touches(p2):
                    return False

    return True


# ==================== 重複排除（正規化） ====================
def get_centers(face_polys):
    centers = []
    for poly in face_polys:
        cx = poly[:, 0].mean()
        cy = poly[:, 1].mean()
        centers.append((cx, cy))
    return centers


def normalize_key(face_polys):
    centers = get_centers(face_polys)
    pts = [(round(x * 2), round(y * 2)) for x, y in centers]

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
        transformed = [tf(x, y) for x, y in pts]
        min_x = min(p[0] for p in transformed)
        min_y = min(p[1] for p in transformed)
        shifted = [(x - min_x, y - min_y) for x, y in transformed]
        shifted.sort()
        key = json.dumps(shifted)
        if min_key is None or key < min_key:
            min_key = key
    return min_key


# ==================== メイン実行 ====================
valid_nets = []
for selected_edges in all_spanning_trees:
    polys = layout_tree(selected_edges)
    if polys is None:
        continue
    if is_valid_net(polys, selected_edges):
        valid_nets.append((selected_edges, polys))

unique_keys = set()
for _, polys in valid_nets:
    unique_keys.add(normalize_key(polys))

print(f"全域木の総数: {len(all_spanning_trees)}")
print(f"有効な展開図（衝突なし）: {len(valid_nets)}")
print(f"回転・反転を除いた一意な展開図: {len(unique_keys)}")

if len(unique_keys) == 11:
    print("✅ 立方体の展開図は正しく11種類です。")
else:
    print(f"❌ 11種類ではありません。一覧: {(unique_keys)}")
