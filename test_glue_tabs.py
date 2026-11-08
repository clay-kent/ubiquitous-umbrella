"""のりしろ（glue tab）の配置ルールのテスト。

ルールは 1 つだけ:

* 展開図で切れている共有辺（= 全域木に含まれない辺）の**両側のうち
  ちょうど片方**にのりしろが付く。

このファイルでは、その「ちょうど片方」を全展開図で数え上げ、
のりしろが面の外周の正しい辺に付いていることまで確認する。
"""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

import api_server
import fifth
import make_nets
import net_test_cache

client = TestClient(api_server.app, raise_server_exceptions=False)
URL = "/api/net/calculate"


CASES = [
    pytest.param(lambda: make_nets.box(1, 1, 1), id="cube"),
    pytest.param(lambda: make_nets.box(1, 2, 3), id="box_1x2x3"),
    pytest.param(make_nets.tetrahedron, id="tetrahedron"),
    pytest.param(lambda: make_nets.prism(4), id="square_prism"),
    pytest.param(lambda: make_nets.prism(5), id="pentagonal_prism"),
    pytest.param(lambda: make_nets.pyramid(4), id="square_pyramid"),
    # 非凸（凹角あり）。全域木が 8100 本あるので既定では slow として外す。
    pytest.param(make_nets.l_shape, id="L-shape", marks=pytest.mark.slow),
]

# のりしろ同士・のりしろと面が重ならないことを要求できるのは、外周が
# 単純（凹角の奥に別の面が潜り込んでいない）な立体だけ。L 字柱の一部の
# 展開図は面が角で接しており、どちら側にも面積のあるのりしろを置けない。
CASES_WITHOUT_TAB_OVERLAP = CASES[:6]


def _l_prism(pts):
    """L 字断面を押し出した角柱。"""
    vertices = np.array(
        [[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts]
    )
    faces = [list(range(5, -1, -1)), list(range(6, 12))]
    faces += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    return vertices, faces


def _thin_l_shape():
    """凹角のある細い L 字柱。のりしろ同士が重なる展開図を含む。"""
    return _l_prism([(0, 0), (3, 0), (3, 0.5), (1, 0.5), (1, 1), (0, 1)])


def _g_l_shape():
    """下辺が太い L 字柱。奥行きを下限まで縮めないと消えない重なりを含む。"""
    return _l_prism([(0, 0), (3, 0), (3, 0.8), (1, 0.8), (1, 1), (0, 1)])


def _point_key(point):
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def _face_has_segment(poly: Polygon, p0, p1) -> bool:
    """面の外周に、座標 ``p0``-``p1`` の辺がそのまま存在するか。"""
    coords = list(poly.exterior.coords)[:-1]
    target = {_point_key(p0), _point_key(p1)}
    return any(
        {_point_key(coords[i]), _point_key(coords[(i + 1) % len(coords)])} == target
        for i in range(len(coords))
    )


def _cut_pairs(faces, tree_edges):
    """隣接辺のうち、全域木に含まれない面対（= 切れている辺）を返す。"""
    adjacent = {
        frozenset((u, v)) for u, v, _edge in fifth.build_adjacency_with_edges(faces)
    }
    tree = {frozenset(edge) for edge in tree_edges}
    return adjacent - tree


def _nets(builder):
    vertices, faces = builder()
    normals = fifth.compute_normals(vertices, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = fifth.enumerate_spanning_trees(edges, len(faces))
    return vertices, faces, normals, net_test_cache.unique_nets(vertices, faces)


@pytest.mark.parametrize("builder", CASES)
def test_exactly_one_tab_per_cut_edge(builder):
    """"両方に付く" も "どちらにも付かない" も存在しないこと。"""
    vertices, faces, _normals, nets = _nets(builder)
    pair_to_edge = fifth._pair_to_edge(faces)

    for tree, polys in nets:
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)
        cut = _cut_pairs(faces, tree)

        # 切れている辺の本数と、のりしろの枚数が一致する。
        assert len(tabs) == len(cut)
        # 1 本の辺に 2 枚付いていないこと。
        keys = [frozenset((t.face_id, t.neighbor_face_id)) for t in tabs]
        assert len(keys) == len(set(keys))
        # 切れている辺をちょうど 1 回ずつ覆っていること。
        assert set(keys) == cut
        # のりしろは必ずその辺の両端の面のどちらかに付く。
        for tab, key in zip(tabs, keys):
            assert tab.face_id in key
            assert tab.neighbor_face_id in key
            assert tab.face_id != tab.neighbor_face_id


@pytest.mark.parametrize("builder", CASES)
def test_tabs_attach_to_the_boundary_edge(builder):
    """のりしろの貼り付け辺が、面の外周の正しい辺と一致すること。"""
    vertices, faces, _normals, nets = _nets(builder)
    pair_to_edge = fifth._pair_to_edge(faces)

    for tree, polys in nets:
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)
        for tab in tabs:
            assert _face_has_segment(polys[tab.face_id], *tab.edge_2d)
            a, b = pair_to_edge[frozenset((tab.face_id, tab.neighbor_face_id))]

            # 貼り付け辺の長さは、立体上の共有辺の長さと等しい（展開は等長）。
            length_2d = math.dist(tab.edge_2d[0], tab.edge_2d[1])
            length_3d = float(
                (vertices[a] - vertices[b]) @ (vertices[a] - vertices[b])
            ) ** 0.5
            assert length_2d == pytest.approx(length_3d, rel=1e-9, abs=1e-12)

            # 3D の辺も、その面対の共有辺と一致する。
            edge_3d = {
                tuple(round(c, 9) for c in tab.edge_3d[0]),
                tuple(round(c, 9) for c in tab.edge_3d[1]),
            }
            assert edge_3d == {
                tuple(round(float(c), 9) for c in vertices[a]),
                tuple(round(float(c), 9) for c in vertices[b]),
            }


@pytest.mark.parametrize("builder", CASES_WITHOUT_TAB_OVERLAP)
def test_tab_shape_is_positive_and_outside_faces(builder):
    """台形が潰れておらず、面の内側へはみ出していないこと。

    凹角のある立体では、外周の外側に別の面がある場合がある。その場合は
    反対側を選ぶので、ここでは「選ばれたのりしろ」だけを見る。
    """
    vertices, faces, _normals, nets = _nets(builder)
    pair_to_edge = fifth._pair_to_edge(faces)

    for tree, polys in nets:
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)

        for tab in tabs:
            tab_poly = Polygon(tab.vertices_2d)
            assert tab_poly.is_valid
            # 貼り付け辺の長さ L に対し、面積は奥行き 0.25L の台形ぶんある。
            assert tab_poly.area > 0.0
            # 面の内部と重ならない（境界で接するだけ）。
            for poly in polys:
                assert fifth._tab_overlap_area(tab_poly, poly) == 0.0

        # のりしろ同士も重ならない。
        tab_polys = [Polygon(t.vertices_2d) for t in tabs]
        for i in range(len(tab_polys)):
            for j in range(i + 1, len(tab_polys)):
                assert fifth._tab_overlap_area(tab_polys[i], tab_polys[j]) == 0.0


@pytest.mark.slow
def test_lshape_tab_geometry_is_valid():
    """凹角のある立体でも、のりしろ自体は健全で、同士で重ならないこと。

    L 字柱の一部の展開図は面が角で接しており、どちら側にも面積のある
    のりしろを置けない（別の面と重なる）。それは配置の制約であって
    台形の破損ではないので、ここでは「自分の面」「他ののりしろ」との
    重なりだけを確認する。
    """
    vertices, faces, _normals, nets = _nets(make_nets.l_shape)
    pair_to_edge = fifth._pair_to_edge(faces)

    for tree, polys in nets:
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)

        tab_polys = []
        for tab in tabs:
            points = {_point_key(p) for p in tab.vertices_2d}
            assert len(points) == 4
            tab_poly = Polygon(tab.vertices_2d)
            assert tab_poly.is_valid
            assert tab_poly.area > 0.0
            # 自分の面の内部へははみ出さない（外周で接するだけ）。
            assert fifth._tab_overlap_area(tab_poly, polys[tab.face_id]) == 0.0
            tab_polys.append(tab_poly)

        for i in range(len(tab_polys)):
            for j in range(i + 1, len(tab_polys)):
                assert fifth._tab_overlap_area(tab_polys[i], tab_polys[j]) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "builder",
    [_thin_l_shape, _g_l_shape],
    ids=["thin-L", "G-L"],
)
def test_lshape_shrinks_tabs_to_avoid_tab_tab_overlap(builder):
    """のりしろ同士が重なる展開図で、奥行きを縮めて重なりを消すこと。

    細い L 字柱と、下辺が太い L 字柱は、既定の奥行き 0.25L だと
    のりしろ同士が重なる展開図を含む。側の選び方だけでは消せないので、
    奥行きを縮める。面が角で接する展開図では面との重なりは残るが、
    のりしろ同士は重ならない（下限まで縮めても消えない場合は下限で残す）。
    """
    vertices, faces = builder()
    fifth.validate_solid(vertices, faces)
    pair_to_edge = fifth._pair_to_edge(faces)
    nets = net_test_cache.unique_nets(vertices, faces)
    shrunk = 0

    for tree, polys in nets:
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)

        tab_polys = []
        for tab in tabs:
            # 貼り付け辺は面の外周と共有したまま（縮めても外れない）。
            assert tab.vertices_2d[:2] == tab.edge_2d
            tab_poly = Polygon(tab.vertices_2d)
            assert tab_poly.area > 0.0
            tab_polys.append(tab_poly)

            length = math.dist(tab.edge_2d[0], tab.edge_2d[1])
            p0, p1, q1, q0 = tab.vertices_2d
            depth = math.dist(
                ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
                ((q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2),
            )
            if depth < 0.99 * fifth.TAB_DEPTH_RATIO * length:
                shrunk += 1

        for i in range(len(tab_polys)):
            for j in range(i + 1, len(tab_polys)):
                assert fifth._tab_overlap_area(tab_polys[i], tab_polys[j]) == 0.0

    # 実際に縮めたのりしろが存在すること（空振り防止）。
    assert shrunk > 0


@pytest.mark.slow
def test_face_overlap_alone_does_not_shrink_tabs():
    """面との重なりだけでは奥行きを縮めないこと。

    G-L の 6 番目の展開図では、面1→面4 ののりしろが面4と 0.39 重なるが、
    他ののりしろとは重ならない。面を障害物に含めると、こののりしろが
    奥行き 0.4 倍まで縮んでしまう（レビュー指摘）。縮めないことを固定する。
    """
    vertices, faces = _g_l_shape()
    pair_to_edge = fifth._pair_to_edge(faces)
    nets = net_test_cache.unique_nets(vertices, faces)
    tree, polys = nets[6]
    tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)

    tab = next(
        t for t in tabs if t.face_id == 1 and t.neighbor_face_id == 4
    )
    # 前提: 面4とは重なっている（面が角で接する展開図）。
    assert fifth._tab_overlap_area(Polygon(tab.vertices_2d), polys[4]) > 0.0

    length = math.dist(tab.edge_2d[0], tab.edge_2d[1])
    p0, p1, q1, q0 = tab.vertices_2d
    depth = math.dist(
        ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
        ((q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2),
    )
    assert depth == pytest.approx(fifth.TAB_DEPTH_RATIO * length)


def test_tab_overlap_threshold_uses_local_area():
    """短い辺ののりしろでも、実重なりを 0 と誤判定しないこと。

    展開図全体の面積を基準にすると、長い辺と短い辺が混在するときに
    短い辺の重なりが閾値の下へ隠れる（レビュー指摘）。局所面積を
    基準にしたことを直接固定する。
    """
    big = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    small = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
    # 交差面積 0.01。旧実装の閾値 1e-7 * 1000^2 = 0.1 なら 0 になっていた。
    assert fifth._tab_overlap_area(big, small) == pytest.approx(0.01)


def test_glue_tabs_does_not_mutate_faces():
    """のりしろを計算しても、面のポリゴンと指標は変わらないこと。"""
    vertices, faces, normals, nets = _nets(lambda: make_nets.box(1, 1, 1))
    shared_of = fifth.build_shared_of(faces)
    pair_to_edge = fifth._pair_to_edge(faces)

    for tree, polys in nets:
        before_wkb = [poly.wkb for poly in polys]
        before = fifth.net_metrics(polys, tree, vertices, faces, normals, shared_of)
        tabs = fifth.glue_tabs(polys, faces, tree, pair_to_edge, vertices)
        after = fifth.net_metrics(polys, tree, vertices, faces, normals, shared_of)
        assert tabs
        assert before == after
        assert [poly.wkb for poly in polys] == before_wkb


def test_api_returns_glue_tabs():
    """API の応答にのりしろが入り、辺の対応が展開図と一致すること。"""
    vertices, faces = make_nets.box(1, 1, 1)
    top_n = 3
    body = {
        "vertices": [[float(c) for c in p] for p in vertices.tolist()],
        "faces": faces,
        "top_n": top_n,
        "sort_by": "area",
    }
    response = client.post(URL, json=body)
    assert response.status_code == 200, response.text[:400]
    payload = response.json()

    adjacent = {
        frozenset((u, v)) for u, v, _edge in fifth.build_adjacency_with_edges(faces)
    }
    for net in payload["nets"]:
        tabs = net["glue_tabs"]
        assert tabs
        # 応答の parent_face_id から全域木を復元し、切れている辺を数える。
        tree = {
            frozenset((face["parent_face_id"], face["face_id"]))
            for face in net["faces"]
            if face["parent_face_id"] is not None
        }
        cut = adjacent - tree
        keys = [frozenset((t["face_id"], t["neighbor_face_id"])) for t in tabs]
        assert len(keys) == len(set(keys))
        assert set(keys) == cut

    # JSON のキー名と値が、同じ展開図に対する fifth.glue_tabs の結果と
    # 完全に一致すること。golden は glue_tabs を除外しているので、
    # ここで直列化の対応を固定する。
    oriented = fifth.orient_faces_outward(vertices, faces)
    normals = fifth.compute_normals(vertices, oriented)
    shared_of = fifth.build_shared_of(oriented)
    pair_to_edge = fifth._pair_to_edge(oriented)
    scored = [
        (
            fifth.net_metrics(polys, tree, vertices, oriented, normals, shared_of),
            tree,
            polys,
        )
        for tree, polys in net_test_cache.unique_nets(vertices, oriented)
    ]
    scored.sort(key=lambda s: fifth.metric_sort_key(s[0], "area"))

    expected = [
        api_server._glue_tabs_json(
            fifth.glue_tabs(polys, oriented, tree, pair_to_edge, vertices)
        )
        for _metrics, tree, polys in scored[:top_n]
    ]
    assert [net["glue_tabs"] for net in payload["nets"]] == expected
