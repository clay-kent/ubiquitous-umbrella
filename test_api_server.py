import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from shapely.geometry import Polygon

from api_server import app, make_cuboid, get_face_name, get_layout_parent_relation

client = TestClient(app, raise_server_exceptions=False)

# ==========================================
# 1. ユニットテスト (make_cuboid)
# ==========================================
def test_make_cuboid_success():
    w, h, d = 1.0, 2.0, 3.0
    vertices, faces = make_cuboid(w, h, d)
    
    # 頂点の数と形状の確認
    assert vertices.shape == (8, 3)
    # 期待される頂点の座標
    expected_vertices = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0],
        [0, 0, d],
        [w, 0, d],
        [w, h, d],
        [0, h, d]
    ], dtype=float)
    assert np.allclose(vertices, expected_vertices)
    
    # 面の数とインデックスの確認
    assert len(faces) == 6
    # 面ごとの頂点数
    for face in faces:
        assert len(face) == 4

def test_make_cuboid_degenerate():
    # 寸法が 0 や負の数の場合も make_cuboid 自体は現状エラーにならず頂点を作る
    w, h, d = 0.0, -1.0, 2.0
    vertices, faces = make_cuboid(w, h, d)
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    assert vertices[2][1] == -1.0


# ==========================================
# 2. ユニットテスト (get_face_name)
# ==========================================
@pytest.mark.parametrize("face_id, expected", [
    (0, "底面"),
    (1, "上面"),
    (2, "前面"),
    (3, "右面"),
    (4, "後面"),
    (5, "左面"),
    (-1, "面-1"),
    (6, "面6"),
    (100, "面100"),
])
def test_get_face_name(face_id, expected):
    assert get_face_name(face_id) == expected


# ==========================================
# 3. ユニットテスト (get_layout_parent_relation)
# ==========================================
def test_get_layout_parent_relation_chain():
    # 6つの面が一本鎖で繋がっているツリーを想定: 0 - 2 - 3 - 4 - 5, 2 - 1
    # tree_edges: (0, 2), (2, 3), (3, 4), (4, 5), (2, 1)
    faces = [[0]*4] * 6  # 形状は何でもよいので面数6のダミー
    tree_edges = [(0, 2), (2, 3), (3, 4), (4, 5), (2, 1)]
    
    parent_map = get_layout_parent_relation(faces, tree_edges)
    
    # 期待される親関係:
    # 0 (root) -> None
    # 2 (0から訪問) -> 0
    # 1 (2から訪問) -> 2
    # 3 (2から訪問) -> 2
    # 4 (3から訪問) -> 3
    # 5 (4から訪問) -> 4
    assert parent_map[0] is None
    assert parent_map[2] == 0
    assert parent_map[1] == 2
    assert parent_map[3] == 2
    assert parent_map[4] == 3
    assert parent_map[5] == 4

def test_get_layout_parent_relation_empty():
    faces = [[0]*4] * 6
    # 木のエッジがない場合
    parent_map = get_layout_parent_relation(faces, [])
    # ルート(0)以外は親なし (None) のままであるべき
    assert parent_map == [None] * 6


# ==========================================
# 4. API /api/net/stl のテスト
# ==========================================
def test_get_stl_success():
    response = client.get("/api/net/stl?w=1&h=2&d=3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/sla"
    assert "attachment; filename=cuboid_1.0_2.0_3.0.stl" in response.headers["content-disposition"]
    
    content = response.text
    assert content.startswith("solid cuboid")
    assert content.endswith("endsolid cuboid\n")
    # 12個の三角形(facet)が含まれること
    assert content.count("facet normal") == 12
    assert content.count("outer loop") == 12
    assert content.count("vertex") == 36
    assert content.count("endfacet") == 12

def test_get_stl_validation_error():
    # 必須パラメータがない場合
    response = client.get("/api/net/stl?w=1&h=2")
    assert response.status_code == 422


# ==========================================
# 5. API /api/net/calculate のテスト (モック使用)
# ==========================================
@patch("api_server.fifth")
def test_calculate_nets_mocked(mock_fifth):
    # fifthの挙動をモックする
    # edges, all_trees, normals, unique_trees, layout_tree を設定
    mock_fifth.build_adjacency_graph.return_value = [(0, 1)]
    mock_fifth.enumerate_spanning_trees.return_value = [[(0, 1)]]
    mock_fifth.compute_normals.return_value = [np.array([0, 0, 1])] * 6
    
    # get_unique_nets_by_ribbon_graph が返すユニークな全域木のリスト
    # 面0-面2-面3... のような全域木 (tree_edgesのリスト) を想定
    mock_tree = [(0, 2), (2, 3), (3, 4), (4, 5), (2, 1)]
    mock_fifth.get_unique_nets_by_ribbon_graph.return_value = [mock_tree]
    
    # layout_tree が返す Shapely Polygons (面ごとに4頂点のダミーポリゴンを返す)
    # 面積や外周長の計算に影響するため、具体的な座標を定義
    # 各面 1x1 の正方形とする
    dummy_polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # 面0
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # 面1
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # 面2
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),  # 面3
        Polygon([(0, 3), (1, 3), (1, 4), (0, 4)]),  # 面4
        Polygon([(0, -1), (1, -1), (1, 0), (0, 0)]), # 面5
    ]
    mock_fifth.layout_tree.return_value = dummy_polygons
    
    # API呼び出し
    response = client.get("/api/net/calculate?w=1&h=1&d=1&sort_by=area")
    assert response.status_code == 200
    
    data = response.json()
    assert "dimensions" in data
    assert data["dimensions"] == {"width": 1.0, "height": 1.0, "depth": 1.0}
    assert "nets" in data
    assert len(data["nets"]) == 1
    
    net = data["nets"][0]
    assert net["net_id"] == 1
    assert "metrics" in net
    assert "bounding_box" in net["metrics"]
    assert "bounding_box_area" in net["metrics"]
    assert "perimeter" in net["metrics"]
    
    # faces の検証
    faces = net["faces"]
    assert len(faces) == 6
    for face in faces:
        assert "face_id" in face
        assert "name" in face
        assert len(face["vertices_2d"]) == 4
        assert "parent_face_id" in face
        assert "fold_axis" in face
        assert "fold_angle" in face


# ==========================================
# 6. E2E 結合テスト (実際の fifth を使用)
# ==========================================
def test_calculate_nets_e2e_cube():
    # 1x1x1 立方体の場合、ユニークな展開図は11種類になるはず
    response = client.get("/api/net/calculate?w=1&h=1&d=1&sort_by=area")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["nets"]) == 11
    
    # 面積でソートされていることを確認
    areas = [net["metrics"]["bounding_box_area"] for net in data["nets"]]
    assert areas == sorted(areas)

def test_calculate_nets_e2e_sort_by_perimeter():
    response = client.get("/api/net/calculate?w=1&h=1&d=1&sort_by=perimeter")
    assert response.status_code == 200
    
    data = response.json()
    # 外周長でソートされていることを確認
    perimeters = [net["metrics"]["perimeter"] for net in data["nets"]]
# ==========================================
# 7. 異常系・エッジケースのテスト (追加)
# ==========================================
def test_calculate_nets_invalid_sort_by():
    # 不正な sort_by パラメータを指定した場合、エラーにならずフォールバックされる
    response = client.get("/api/net/calculate?w=1&h=1&d=1&sort_by=invalid_value")
    assert response.status_code == 200
    data = response.json()
    assert "nets" in data
    assert len(data["nets"]) == 11

def test_calculate_nets_negative_dimensions():
    # 負の寸法が渡された場合、現在 API は 200 OK を返すが、展開図が正しく作られず空になる
    response = client.get("/api/net/calculate?w=-1&h=1&d=1")
    assert response.status_code == 200
    data = response.json()
    assert "nets" in data
    # 負の寸法では有効なレイアウトが構築できず、ユニークな展開図リストは空になる
    assert len(data["nets"]) == 0


@patch("api_server.fifth")
def test_calculate_nets_empty_layout(mock_fifth):
    # layout_tree が空を返した場合に、バウンディングボックスの計算 (min/max) で
    # 例外 (ValueError) が発生することを確認するテスト
    mock_fifth.build_adjacency_graph.return_value = [(0, 1)]
    mock_fifth.enumerate_spanning_trees.return_value = [[(0, 1)]]
    mock_fifth.compute_normals.return_value = [np.array([0, 0, 1])] * 6
    mock_fifth.get_unique_nets_by_ribbon_graph.return_value = [[(0, 1)]]
    
    # layout_tree が空リストを返すようにモック
    mock_fifth.layout_tree.return_value = []
    
    # min() の引数が空のため ValueError が起きて 500 になる
    response = client.get("/api/net/calculate?w=1&h=1&d=1")
    assert response.status_code == 500

