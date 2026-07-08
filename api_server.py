import sys
import numpy as np
import fifth
from fastapi import FastAPI, Query
from fastapi.responses import Response
from shapely.geometry import Polygon
from typing import Optional, List, Dict, Any

app = FastAPI(title="Cuboid Net Generator API")

def make_cuboid(w: float, h: float, d: float) -> tuple[np.ndarray, list[list[int]]]:
    """与えられた寸法(w, h, d)の直方体の頂点と面リストを生成する。"""
    vertices = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0],
        [0, 0, d],
        [w, 0, d],
        [w, h, d],
        [0, h, d]
    ], dtype=float)
    
    faces = [
        [0, 3, 2, 1],  # 0: 底面
        [4, 5, 6, 7],  # 1: 上面
        [0, 1, 5, 4],  # 2: 前面
        [1, 2, 6, 5],  # 3: 右面
        [2, 3, 7, 6],  # 4: 後面
        [3, 0, 4, 7]   # 5: 左面
    ]
    return vertices, faces

def get_face_name(face_id: int) -> str:
    names = ["底面", "上面", "前面", "右面", "後面", "左面"]
    return names[face_id] if 0 <= face_id < 6 else f"面{face_id}"

def get_layout_parent_relation(
    faces: list[list[int]],
    tree_edges: list[tuple[int, int]]
) -> list[tuple[int, Optional[int]]]:
    """全域木から、面の親子関係（展開順序における親面）を決定する。
    面0をルートとする。
    """
    n_faces = len(faces)
    adj = [[] for _ in range(n_faces)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    parent_map = [None] * n_faces
    queue = [0]
    visited = {0}
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                parent_map[v] = u
                visited.add(v)
                queue.append(v)
    
    return parent_map

@app.get("/api/net/calculate")
def calculate_nets(
    w: float = Query(..., description="幅 (width)"),
    h: float = Query(..., description="高さ (height)"),
    d: float = Query(..., description="奥行き (depth)"),
    sort_by: Optional[str] = Query("area", description="ソート基準 (area または perimeter)")
):
    vertices, faces = make_cuboid(w, h, d)
    
    # 全域木の列挙と法線計算
    edges = fifth.build_adjacency_graph(faces)
    all_trees = fifth.enumerate_spanning_trees(edges, len(faces))
    normals = fifth.compute_normals(vertices, faces)
    
    # グラフ同型判定を用いてユニークな全域木を取得
    unique_trees = fifth.get_unique_nets_by_ribbon_graph(vertices, faces, all_trees, normals)
    
    nets_data = []
    
    for net_idx, tree in enumerate(unique_trees):
        # 2D配置
        polys = fifth.layout_tree(vertices, faces, normals, tree)
        
        # 評価指標の算出
        all_x, all_y = [], []
        for poly in polys:
            x, y = poly.exterior.xy
            all_x.extend(x)
            all_y.extend(y)
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_area = bbox_w * bbox_h
        
        # 外周長 (隣接/接着辺を除いた露出している辺の合計)
        # 簡易的に、各面の外周の合計から 2 * 接着辺の長さを引く
        total_perimeter = 0.0
        for poly in polys:
            total_perimeter += poly.length
        # 接着辺の長さの合計を引く
        shared_length = 0.0
        for u, v in tree:
            shared_nodes = set(faces[u]) & set(faces[v])
            pt_indices = list(shared_nodes)
            pA = vertices[pt_indices[0]]
            pB = vertices[pt_indices[1]]
            shared_length += np.linalg.norm(pB - pA)
        
        perimeter = total_perimeter - 2.0 * shared_length
        
        # 親面マップの取得
        parent_map = get_layout_parent_relation(faces, tree)
        
        faces_json = []
        for face_id, poly in enumerate(polys):
            x, y = poly.exterior.xy
            # poly.exterior.xy の頂点数は開始点と終了点が重複して5個になるので、最初の4個を取得
            vertices_2d = [[x[i], y[i]] for i in range(4)]
            
            # 親面との関係および折り曲げ軸の決定
            parent_id = parent_map[face_id]
            fold_axis = None
            fold_angle = 0.0
            
            if parent_id is not None:
                # 共有頂点を探す
                shared = list(set(faces[face_id]) & set(faces[parent_id]))
                if len(shared) == 2:
                    pA = vertices[shared[0]].tolist()
                    pB = vertices[shared[1]].tolist()
                    fold_axis = [pA, pB]
                    fold_angle = 90.0 # 直方体なので常に90度
            
            faces_json.append({
                "face_id": face_id,
                "name": get_face_name(face_id),
                "vertices_2d": vertices_2d,
                "parent_face_id": parent_id,
                "fold_axis": fold_axis,
                "fold_angle": fold_angle
            })
            
        nets_data.append({
            "net_id": net_idx + 1,
            "metrics": {
                "bounding_box": {
                    "width": float(bbox_w),
                    "height": float(bbox_h)
                },
                "bounding_box_area": float(bbox_area),
                "perimeter": float(perimeter)
            },
            "faces": faces_json
        })
        
    # ソート
    if sort_by == "area":
        nets_data.sort(key=lambda x: x["metrics"]["bounding_box_area"])
    elif sort_by == "perimeter":
        nets_data.sort(key=lambda x: x["metrics"]["perimeter"])
        
    return {
        "dimensions": {
            "width": w,
            "height": h,
            "depth": d
        },
        "nets": nets_data
    }

@app.get("/api/net/stl")
def get_stl(
    w: float = Query(..., description="幅 (width)"),
    h: float = Query(..., description="高さ (height)"),
    d: float = Query(..., description="奥行き (depth)")
):
    # STLテキストデータの生成
    # 直方体は12個の三角形（各長方形面に2つの三角形）で表現される。
    vertices = [
        [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
        [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]
    ]
    # 各面の三角形（外側を向くように構成）
    triangles = [
        # 底面 (z=0, 法線下向き [0, 0, -1])
        ([0, 0, -1], vertices[0], vertices[2], vertices[1]),
        ([0, 0, -1], vertices[0], vertices[3], vertices[2]),
        # 上面 (z=d, 法線上向き [0, 0, 1])
        ([0, 0, 1], vertices[4], vertices[5], vertices[6]),
        ([0, 0, 1], vertices[4], vertices[6], vertices[7]),
        # 前面 (y=0, 法線前向き [0, -1, 0])
        ([0, -1, 0], vertices[0], vertices[1], vertices[5]),
        ([0, -1, 0], vertices[0], vertices[5], vertices[4]),
        # 右面 (x=w, 法線右向き [1, 0, 0])
        ([1, 0, 0], vertices[1], vertices[2], vertices[6]),
        ([1, 0, 0], vertices[1], vertices[6], vertices[5]),
        # 後面 (y=h, 法線後向き [0, 1, 0])
        ([0, 1, 0], vertices[2], vertices[3], vertices[7]),
        ([0, 1, 0], vertices[2], vertices[7], vertices[6]),
        # 左面 (x=0, 法線左向き [-1, 0, 0])
        ([-1, 0, 0], vertices[3], vertices[0], vertices[4]),
        ([-1, 0, 0], vertices[3], vertices[4], vertices[7]),
    ]
    
    stl_lines = ["solid cuboid"]
    for normal, p1, p2, p3 in triangles:
        stl_lines.append(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
        stl_lines.append("    outer loop")
        stl_lines.append(f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}")
        stl_lines.append(f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}")
        stl_lines.append(f"      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}")
        stl_lines.append("    endloop")
        stl_lines.append("  endfacet")
    stl_lines.append("endsolid cuboid\n")
    
    stl_content = "\n".join(stl_lines)
    
    return Response(
        content=stl_content,
        media_type="application/sla",
        headers={"Content-Disposition": f"attachment; filename=cuboid_{w}_{h}_{d}.stl"}
    )
