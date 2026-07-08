import itertools
import json
import math

import networkx as nx
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

# ========== 入力データ ==========
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
    [0, 1, 2, 3],  # 底面
    [4, 5, 6, 7],  # 上面
    [0, 1, 5, 4],  # 前面
    [1, 2, 6, 5],  # 右面
    [2, 6, 7, 3],  # 後面
    [3, 7, 4, 0],  # 左面
]

local_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

N = len(faces)

# ========== 面の隣接グラフの構築 ==========
edges = []
for i in range(N):
    for j in range(i + 1, N):
        if len(set(faces[i]) & set(faces[j])) == 2:  # 辺を共有
            edges.append((i, j))

# ========== 全域木の列挙 ==========
all_trees = []
for edge_subset in itertools.combinations(edges, N - 1):
    g = nx.Graph()
    g.add_nodes_from(range(N))
    g.add_edges_from(edge_subset)
    if nx.is_tree(g):
        all_trees.append(edge_subset)


# ========== 展開図の全組み合わせを列挙する ==========
def build_local_polygon(face_idx):
    return Polygon(local_coords[: len(faces[face_idx])])


def find_shared_edge_vertices(u, v):
    common = set(faces[u]) & set(faces[v])
    return tuple(common)


def layout_tree(tree_edges):
    adj = [[] for _ in range(N)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    polygons_abs = [None] * N
    polygons_abs[0] = build_local_polygon(0)

    queue = [0]
    visited = {0}

    while queue:
        u = queue.pop(0)
        poly_u = polygons_abs[u]

        for v in adj[u]:
            if v in visited:
                continue

            shared_vtx = find_shared_edge_vertices(u, v)

            def get_edge_line(poly, face_idx):
                coords = np.array(poly.exterior.coords[:-1])
                indices = faces[face_idx]
                n_verts = len(indices)
                for i in range(n_verts):
                    a = indices[i]
                    b = indices[(i + 1) % n_verts]
                    if set([a, b]) == set(shared_vtx):
                        return coords[i], coords[(i + 1) % n_verts]
                raise ValueError("共有辺が見つからない")

            p_start_u, p_end_u = get_edge_line(poly_u, u)

            local_poly_v = build_local_polygon(v)
            p_start_v, p_end_v = get_edge_line(local_poly_v, v)

            # 移動
            dx = p_start_u[0] - p_start_v[0]
            dy = p_start_u[1] - p_start_v[1]
            poly_v_aligned = affinity.translate(local_poly_v, dx, dy)

            # 回転
            vec_u = np.array(p_end_u) - np.array(p_start_u)
            vec_v = np.array(p_end_v) - np.array(p_start_v)
            angle_u = math.atan2(vec_u[1], vec_u[0])
            angle_v = math.atan2(vec_v[1], vec_v[0])
            rot_angle = angle_u - angle_v
            poly_v_rotated = affinity.rotate(
                poly_v_aligned,
                math.degrees(rot_angle),
                origin=(p_start_u[0], p_start_u[1]),
            )

            # 必要なら反転
            coords_u = np.array(poly_u.exterior.coords[:-1])
            other_u = None
            for pt in coords_u:
                if not (np.allclose(pt, p_start_u) or np.allclose(pt, p_end_u)):
                    other_u = pt
                    break

            coords_v_rot = np.array(poly_v_rotated.exterior.coords[:-1])
            other_v = None
            for pt in coords_v_rot:
                if not (np.allclose(pt, p_start_u) or np.allclose(pt, p_end_u)):
                    other_v = pt
                    break

            edge_vec = vec_u
            normal = np.array([-edge_vec[1], edge_vec[0]])
            side_u = np.dot(np.array(other_u) - np.array(p_start_u), normal)
            side_v = np.dot(np.array(other_v) - np.array(p_start_u), normal)
            if side_u * side_v > 0:
                dx_e, dy_e = edge_vec
                length = math.hypot(dx_e, dy_e)
                ux, uy = dx_e / length, dy_e / length
                a = ux * ux - uy * uy
                b = 2 * ux * uy
                c = 2 * ux * uy
                d = uy * uy - ux * ux
                T = affinity.translate(poly_v_rotated, -p_start_u[0], -p_start_u[1])
                T = affinity.affine_transform(T, [a, b, c, d, 0, 0])
                T = affinity.translate(T, p_start_u[0], p_start_u[1])
                poly_v_rotated = T

            polygons_abs[v] = poly_v_rotated
            visited.add(v)
            queue.append(v)

    return polygons_abs


# ========== 面の衝突判定をして、無効な展開図はフィルタ ==========
def is_valid_layout(polygons):
    for i in range(N):
        for j in range(i + 1, N):
            # 浮動小数点計算の微小な誤差を考慮し、面積が 1e-5 以上重なっていたらNGとする
            if polygons[i].intersection(polygons[j]).area > 1e-5:
                return False
    return True


# ========== 正規化(同じ展開図は同じ表現になるようにsort) ==========
def normalize_key(polygons):
    centers = [(p.centroid.x, p.centroid.y) for p in polygons]
    pts = [(round(x * 2, 6), round(y * 2, 6)) for x, y in centers]

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
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        shifted = [(x - min_x, y - min_y) for x, y in transformed]
        shifted.sort()
        key = json.dumps(shifted)
        if min_key is None or key < min_key:
            min_key = key
    return min_key


# ========== main処理 ==========
valid_nets = []
for tree in all_trees:
    polys = layout_tree(tree)
    if polys is None:
        continue

    # 判定関数に tree_edges を渡す必要すらなくなりました
    if is_valid_layout(polys):
        valid_nets.append(polys)

unique_keys = set(normalize_key(p) for p in valid_nets)

print(f"全域木の総数: {len(all_trees)}")
print(f"有効な展開図: {len(valid_nets)}")
print(f"回転・反転を除いた一意な展開図: {len(unique_keys)}")
# memo: あとで外部テスト化し、立方体以外でも検証を行う
if len(unique_keys) == 11:
    print("✅ 立方体の展開図は11種類です。")
else:
    print(f"❌ 11種類ではありません。確認: {unique_keys}")
# 評価手法は要検討
