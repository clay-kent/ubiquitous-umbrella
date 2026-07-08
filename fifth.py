import itertools
import sys

import networkx as nx
import numpy as np
from shapely.geometry import Polygon


def build_adjacency_graph(faces: list[list[int]]) -> list[tuple[int, int]]:
    """面の隣接グラフの辺リストを構築する。"""
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
    """各面の外向き法線ベクトルを計算する。"""
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

    m_matrices = [None] * n_faces
    m_matrices[0] = np.eye(4)
    queue = [0]
    visited = {0}

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v in visited:
                continue

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

            edge_vec = p_b - p_a
            k = edge_vec / np.linalg.norm(edge_vec)

            nu = normals[u]
            nv = normals[v]
            c = np.dot(nv, nu)
            s = np.dot(np.cross(k, nv), nu)

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

            t1 = np.eye(4)
            t1[0:3, 3] = -p_a
            t2 = np.eye(4)
            t2[0:3, 3] = p_a
            transform = t2 @ r_rot @ t1

            m_matrices[v] = m_matrices[u] @ transform
            visited.add(v)
            queue.append(v)

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
            if polygons[i].intersection(polygons[j]).area > 1e-4:
                return False
    return True


def build_ribbon_graph(
    vertices: np.ndarray,
    faces: list[list[int]],
    tree_edges: list[tuple[int, int]],
) -> nx.DiGraph:
    """展開図の木構造から半辺グラフ (Ribbon Graph) を構築する。

    ノード: 各面の各辺に対応する半辺 (face_index, edge_index)
      - 属性 `length`: 辺の長さ (丸め処理)
    エッジ:
      - 'next' 型の有向エッジ: 同一面上での巡回順序 (CCW)
      - 'glue' 型の無向/双方向有向エッジ: 隣接面間の接着
        - 属性 `flipped`: 接着される向き
    """
    g = nx.DiGraph()

    # 1. ハーフエッジのノードを作成し、面内の「next」エッジで巡回的に結ぶ
    for f_idx, face in enumerate(faces):
        n_v = len(face)
        for i in range(n_v):
            p1 = vertices[face[i]]
            p2 = vertices[face[(i + 1) % n_v]]
            length = np.linalg.norm(p2 - p1)
            
            node_id = (f_idx, i)
            g.add_node(node_id, length=round(length, 5))

        # nextエッジを追加 (面をCCWに一周するサイクル)
        for i in range(n_v):
            g.add_edge((f_idx, i), (f_idx, (i + 1) % n_v), type="next")

    # 2. 面間の接着 (glue) エッジを追加
    for u, v in tree_edges:
        # 面uと面vの共有頂点を3Dで探す
        shared = set(faces[u]) & set(faces[v])
        if len(shared) != 2:
            continue
        
        # 面uにおける共有辺のインデックスを探す
        idx_u = None
        n_u = len(faces[u])
        for i in range(n_u):
            a, b = faces[u][i], faces[u][(i + 1) % n_u]
            if a in shared and b in shared:
                idx_u = i
                break
        
        # 面vにおける共有辺のインデックスを探す
        idx_v = None
        n_v = len(faces[v])
        for i in range(n_v):
            a, b = faces[v][i], faces[v][(i + 1) % n_v]
            if a in shared and b in shared:
                idx_v = i
                break

        if idx_u is not None and idx_v is not None:
            # glueエッジを双方向で接続
            u_start, u_end = faces[u][idx_u], faces[u][(idx_u + 1) % n_u]
            v_start, v_end = faces[v][idx_v], faces[v][(idx_v + 1) % n_v]
            
            flipped = (u_start == v_end) and (u_end == v_start)
            
            g.add_edge((u, idx_u), (v, idx_v), type="glue", flipped=flipped)
            g.add_edge((v, idx_v), (u, idx_u), type="glue", flipped=flipped)

    return g


def is_isomorphic_net(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
    """2つの半辺グラフが同型であるかを判定する。"""
    def node_match(n1, n2):
        # 辺の長さが一致するかチェック
        return abs(n1["length"] - n2["length"]) < 1e-4

    def edge_match(e1, e2):
        # エッジのタイプ (next or glue) と flipped 属性が一致するかチェック
        if e1["type"] != e2["type"]:
            return False
        if e1["type"] == "glue":
            return e1.get("flipped") == e2.get("flipped")
        return True

    GM = nx.algorithms.isomorphism.DiGraphMatcher(
        g1, g2, node_match=node_match, edge_match=edge_match
    )
    return GM.is_isomorphic()


def reverse_ribbon_graph(g: nx.DiGraph) -> nx.DiGraph:
    """ハーフエッジグラフの面内有向エッジ(next)をすべて反転させた、鏡像（裏返し）グラフを生成する。"""
    rev_g = nx.DiGraph()
    for node, data in g.nodes(data=True):
        rev_g.add_node(node, **data)
    for u, v, data in g.edges(data=True):
        if data["type"] == "next":
            rev_g.add_edge(v, u, **data)
        else:
            rev_g.add_edge(u, v, **data)
    return rev_g


def get_unique_nets_by_ribbon_graph(
    vertices: np.ndarray,
    faces: list[list[int]],
    all_trees: list[list[tuple[int, int]]],
    normals: list[np.ndarray],
) -> list[list[tuple[int, int]]]:
    """半辺グラフの同型判定を用いて、一意な展開図に対応する全域木を抽出する。"""
    unique_trees = []
    unique_graphs = []
    unique_graphs_rev = []

    for tree in all_trees:
        try:
            polys = layout_tree(vertices, faces, normals, tree)
            if not is_valid_layout(polys):
                continue
            
            rg = build_ribbon_graph(vertices, faces, tree)
            
            # 既存のユニークなグラフ（およびその鏡像）と同型かチェック
            is_duplicate = False
            for u_g, u_g_rev in zip(unique_graphs, unique_graphs_rev):
                if is_isomorphic_net(rg, u_g) or is_isomorphic_net(rg, u_g_rev):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_graphs.append(rg)
                unique_graphs_rev.append(reverse_ribbon_graph(rg))
                unique_trees.append(tree)
        except Exception as e:
            continue

    return unique_trees


def generate_unfolding_nets(
    vertices: np.ndarray | list[list[float]], faces: list[list[int]]
) -> list[list[Polygon]]:
    """半辺グラフ同型判定を用いて一意な展開図を生成する。"""
    vertices = np.array(vertices, dtype=float)
    n_faces = len(faces)

    if n_faces < 4:
        raise ValueError("多面体は少なくとも4つ以上の面を持つ必要があります。")

    edges = build_adjacency_graph(faces)
    all_trees = enumerate_spanning_trees(edges, n_faces)
    normals = compute_normals(vertices, faces)

    unique_trees = get_unique_nets_by_ribbon_graph(vertices, faces, all_trees, normals)

    valid_nets = []
    for tree in unique_trees:
        polys = layout_tree(vertices, faces, normals, tree)
        valid_nets.append(polys)

    return valid_nets


if __name__ == "__main__":
    # 出力をUTF-8に変更 (Windows/CP932環境での文字化け対策)
    sys.stdout.reconfigure(encoding="utf-8")

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

    # --- 2. 正四面体 ---
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
        print(f"一意な展開図の数: {len(nets)}\n")
