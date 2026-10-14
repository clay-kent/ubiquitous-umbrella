"""api_server.py（POST /api/net/calculate）のユニット・結合テスト。"""

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_server

client = TestClient(api_server.app, raise_server_exceptions=False)

URL = "/api/net/calculate"


def detail_msg(r) -> str:
    """422 の detail は常にオブジェクトの配列なので、全 msg を繋げて返す。

    Pydantic のフィールド検証エラーと本 API のエラーはどちらも
    ``{"type", "loc", "msg"}`` の配列なので、同じ読み方で扱える。
    """
    detail = r.json()["detail"]
    assert isinstance(detail, list), f"detail が配列ではない: {detail!r}"
    assert all(isinstance(item, dict) for item in detail), detail
    return " ".join(str(item["msg"]) for item in detail)


# --------------------------------------------------------------------------
# テスト用の立体
# --------------------------------------------------------------------------


def cube_body(w=1.0, h=1.0, d=1.0):
    v = [
        [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
        [0, 0, d], [w, 0, d], [w, h, d], [0, h, d],
    ]
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return {"vertices": v, "faces": f}


def tetrahedron_body():
    return {
        "vertices": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        "faces": [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    }


def prism_body(n, r=1.0, h=1.0):
    v = [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ]
    v += [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), h]
        for i in range(n)
    ]
    f = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    f += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return {"vertices": v, "faces": f}


def pyramid_body(n, r=1.0, h=1.0):
    v = [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ]
    v += [[0.0, 0.0, h]]
    f = [list(range(n - 1, -1, -1))] + [[i, (i + 1) % n, n] for i in range(n)]
    return {"vertices": v, "faces": f}


def l_shape_body():
    """非凸の L 字柱。凹角（270 度）を持つ。"""
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
    v = [[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts]
    f = [list(range(5, -1, -1)), list(range(6, 12))]
    f += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    return {"vertices": v, "faces": f}


def cutout_body():
    """立方体を貫通する四角い穴つき（genus=1）。上面と下面は4枚の四角に分割する。"""
    outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
    inner = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    loop = outer + inner
    top = len(loop)
    v = [[x, y, z] for z in (0.0, 1.0) for x, y in loop]
    f = []
    for k in range(4):
        j = (k + 1) % 4
        f.append([k, j, top + j, top + k])                     # 外側の側面
        f.append([4 + k, 4 + j, top + 4 + j, top + 4 + k])     # 穴の側面
    for base, flip in ((0, False), (top, True)):
        for k in range(4):
            j = (k + 1) % 4
            quad = [base + k, base + j, base + 4 + j, base + 4 + k]
            f.append(list(reversed(quad)) if flip else quad)
    return {"vertices": v, "faces": f}


def octahedron_body():
    v = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    f = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    return {"vertices": v, "faces": f}


def post(body, headers=None, content=None, **params):
    """本体に solidify パラメータを載せて POST する。"""
    if content is not None:
        return client.post(URL, content=content, headers=headers or {})
    payload = dict(body or {})
    payload.update(params)
    return client.post(URL, json=payload, headers=headers or {})


# --------------------------------------------------------------------------
# ルーティング
# --------------------------------------------------------------------------


def test_get_is_not_allowed():
    """この API は POST のみ。GET は 405。"""
    assert client.get(URL).status_code == 405


def test_removed_endpoints_are_gone():
    """旧 GET エンドポイント（stl）は存在しない。"""
    assert client.get("/api/net/stl", params={"w": 1, "h": 1, "d": 1}).status_code == 404
    assert client.get("/api/net/calculate").status_code == 405


def test_openapi_only_exposes_post_calculate():
    paths = client.get("/openapi.json").json()["paths"]
    assert list(paths) == ["/api/net/calculate"]
    assert list(paths["/api/net/calculate"]) == ["post"]


# --------------------------------------------------------------------------
# 一意な展開図の個数
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, builder, expected",
    [
        ("tetrahedron", tetrahedron_body, 2),
        ("octahedron", octahedron_body, 11),
        ("cube", cube_body, 11),
        ("triangular prism", lambda: prism_body(3), 9),
        ("square prism", lambda: prism_body(4), 29),
        ("pentagonal prism", lambda: prism_body(5), 99),
        ("square pyramid", lambda: pyramid_body(4), 8),
        ("pentagonal pyramid", lambda: pyramid_body(5), 15),
    ],
)
def test_total_unique_nets(name, builder, expected):
    r = post(builder())
    assert r.status_code == 200, r.text
    assert r.json()["total_unique_nets"] == expected


def test_non_convex_solid_is_accepted():
    """凹角を持つ L 字柱も拒否せず、上位n個を返す。"""
    r = post(l_shape_body())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total_unique_nets"] == 1516
    assert data["returned"] == 1
    assert len(data["nets"][0]["faces"]) == 8


def test_genus_one_solid_is_accepted():
    """穴つき（genus=1）の立体のも、妥当なら受け付けられる。"""
    r = post(cutout_body())
    # 全域木が 4200 万通りで上限を超えるので、列挙前に 422 で弾かれる
    assert r.status_code == 422
    assert "上限" in detail_msg(r)


# --------------------------------------------------------------------------
# top_n / sort_by
# --------------------------------------------------------------------------


def test_top_n_defaults_to_one():
    r = post(cube_body())
    data = r.json()
    assert data["returned"] == 1
    assert len(data["nets"]) == 1
    assert data["nets"][0]["net_id"] == 1


@pytest.mark.parametrize("n", [1, 3, 5, 11])
def test_top_n_returns_that_many_nets(n):
    r = post(cube_body(), top_n=n)
    data = r.json()
    assert data["returned"] == n
    assert len(data["nets"]) == n
    assert [net["net_id"] for net in data["nets"]] == list(range(1, n + 1))


def test_top_n_is_capped_at_total_unique_nets():
    r = post(cube_body(), top_n=11)
    assert r.json()["returned"] == 11
    r = post(tetrahedron_body(), top_n=5)
    assert r.json()["returned"] == 2  # 2種しかない


def test_sort_by_area_is_ascending():
    r = post(cube_body(), top_n=11, sort_by="area")
    areas = [net["metrics"]["bounding_box_area"] for net in r.json()["nets"]]
    assert areas == sorted(areas)


def test_sort_by_perimeter_is_ascending():
    r = post(cube_body(), top_n=11, sort_by="perimeter")
    perims = [net["metrics"]["perimeter"] for net in r.json()["nets"]]
    assert perims == sorted(perims)


def test_sort_by_area_and_perimeter_can_pick_different_nets():
    """評価指標ごとに「最良」の意味が異なる（立方体の最小外接矩形面積と最小外周長は別物）。"""
    by_area = post(cube_body(), top_n=11, sort_by="area").json()["nets"]
    by_perimeter = post(cube_body(), top_n=11, sort_by="perimeter").json()["nets"]
    areas = [n["metrics"]["bounding_box_area"] for n in by_area]
    perims = [n["metrics"]["perimeter"] for n in by_perimeter]
    assert areas[0] == min(areas) == 10.0          # 1x5 の帯が最良（面積最小）
    assert perims[0] == min(perims) == 14.0
    # 面積最良と周長最良は別の展開図になりうる
    assert by_area[0]["metrics"] != by_perimeter[0]["metrics"]


def test_sort_by_defaults_to_area():
    r = post(cube_body())
    assert r.json()["sort_by"] == "area"


def test_unknown_sort_by_is_rejected():
    assert post(cube_body(), sort_by="volume").status_code == 422


@pytest.mark.parametrize("n", [0, -1, 10_001])
def test_out_of_range_top_n_is_rejected(n):
    assert post(cube_body(), top_n=n).status_code == 422


# --------------------------------------------------------------------------
# レスポンスの形
# --------------------------------------------------------------------------


def test_response_envelope():
    r = post(cube_body(), top_n=2)
    data = r.json()
    assert set(data) == {"solid", "sort_by", "total_unique_nets", "returned", "nets"}
    assert data["solid"]["vertex_count"] == 8
    assert data["solid"]["face_count"] == 6
    assert data["solid"]["face_vertex_counts"] == [4] * 6
    assert data["solid"]["bounding_box"] == {"width": 1.0, "height": 1.0, "depth": 1.0}
    assert data["solid"]["total_area"] == pytest.approx(6.0)
    for net in data["nets"]:
        assert set(net) == {"net_id", "metrics", "faces"}
        assert set(net["metrics"]) == {"bounding_box", "bounding_box_area", "perimeter"}
        assert set(net["metrics"]["bounding_box"]) == {"width", "height"}


def test_each_face_reports_its_own_polygon_size():
    data = post(tetrahedron_body()).json()
    face = data["nets"][0]["faces"][0]
    assert set(face) == {
        "face_id", "name", "vertices_2d", "parent_face_id",
        "dihedral_angle", "fold_angle", "fold_axis_3d", "fold_axis_2d",
    }
    assert len(face["vertices_2d"]) == 3
    assert all(len(pt) == 2 for pt in face["vertices_2d"])


def test_faces_of_different_sizes_keep_their_vertex_counts():
    """三角柱は底面が三角形・側面が四角形なので、面ごとに頂点数が違う。"""
    r = post(prism_body(3))
    data = r.json()
    sizes = sorted(len(f["vertices_2d"]) for f in data["nets"][0]["faces"])
    assert sizes == [3, 3, 4, 4, 4]


def test_every_face_appears_exactly_once():
    for builder in (cube_body, tetrahedron_body, lambda: prism_body(5)):
        data = post(builder()).json()
        faces = data["nets"][0]["faces"]
        assert [f["face_id"] for f in faces] == list(range(data["solid"]["face_count"]))
        assert len({f["face_id"] for f in faces}) == data["solid"]["face_count"]


# --------------------------------------------------------------------------
# 折り曲げ角・軸
# --------------------------------------------------------------------------


def test_root_face_has_no_fold_information():
    faces = post(cube_body()).json()["nets"][0]["faces"]
    root = faces[0]
    assert root["parent_face_id"] is None
    assert root["dihedral_angle"] is None
    assert root["fold_angle"] is None
    assert root["fold_axis_3d"] is None
    assert root["fold_axis_2d"] is None


def test_cube_fold_angles_are_ninety_degrees():
    faces = post(cube_body()).json()["nets"][0]["faces"]
    children = [f for f in faces if f["parent_face_id"] is not None]
    assert len(children) == 5  # 全域木なので根以外は必ず親を持つ
    for f in children:
        assert f["dihedral_angle"] == pytest.approx(90.0)
        assert f["fold_angle"] == pytest.approx(90.0)
        assert f["fold_angle"] == pytest.approx(180.0 - f["dihedral_angle"])


def test_reflex_edge_reports_270_degrees():
    """L 字柱の凹角棱は dihedral=270, fold=-90 になる。

    凹角棱は 1 本だけで、どの展開図にも含まれるわけではないので、
    多くの展開図をまとめて見てその値が出ることを確認する。
    """
    data = post(l_shape_body(), top_n=400).json()
    seen = set()
    for net in data["nets"]:
        for f in net["faces"]:
            if f["dihedral_angle"] is None:
                continue
            assert f["fold_angle"] == pytest.approx(180.0 - f["dihedral_angle"])
            seen.add(round(f["dihedral_angle"], 6))
    assert seen <= {90.0, 270.0}          # L 字柱の二面角は 90 と 270 だけ
    assert 90.0 in seen
    assert 270.0 in seen, "凹角棱の 270 度が見つからない"


def test_convex_solids_only_report_right_angles_for_prisms():
    data = post(cube_body(), top_n=11).json()
    for net in data["nets"]:
        for f in net["faces"]:
            if f["dihedral_angle"] is not None:
                assert f["dihedral_angle"] == pytest.approx(90.0)


def test_parent_relation_is_a_rooted_tree():
    data = post(prism_body(5)).json()
    for net in data["nets"]:
        parent = {f["face_id"]: f["parent_face_id"] for f in net["faces"]}
        roots = [k for k, v in parent.items() if v is None]
        assert len(roots) == 1
        # 親をたどると必ず根に到達する（閉路がない）
        for node in parent:
            seen = set()
            cur = node
            while parent[cur] is not None:
                assert cur not in seen
                seen.add(cur)
                cur = parent[cur]
            assert cur == roots[0]


def test_fold_axis_3d_is_a_real_edge_of_the_solid():
    body = cube_body()
    edge_points = {
        frozenset((tuple(float(c) for c in body["vertices"][e[0]]),
                   tuple(float(c) for c in body["vertices"][e[1]])))
        for e in api_server.fifth.build_edge_map(body["faces"])
    }
    checked = 0
    for net in post(body, top_n=11).json()["nets"]:
        for f in net["faces"]:
            if f["fold_axis_3d"] is None:
                continue
            a, b = (tuple(float(c) for c in p) for p in f["fold_axis_3d"])
            assert frozenset((a, b)) in edge_points
            checked += 1
    assert checked == 11 * 5  # 11 展開図 x 根以外の5面


def test_fold_axis_2d_lies_on_both_faces():
    data = post(prism_body(4)).json()
    for net in data["nets"]:
        for f in net["faces"]:
            if f["fold_axis_2d"] is None:
                continue
            pts = {tuple(round(c, 9) for c in p) for p in f["vertices_2d"]}
            for p in f["fold_axis_2d"]:
                assert tuple(round(c, 9) for c in p) in pts


# --------------------------------------------------------------------------
# 展開図の幾何
# --------------------------------------------------------------------------


def _polygon_area(coords):
    a = 0.0
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % len(coords)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def test_net_preserves_every_face_area():
    """展開しても各面の面積は変わらない。"""
    for builder in (cube_body, tetrahedron_body, lambda: prism_body(5), l_shape_body):
        body = builder()
        data = post(body, top_n=3).json()
        for net in data["nets"]:
            for f in net["faces"]:
                expected = _face_area_3d(body, f["face_id"])
                assert _polygon_area(f["vertices_2d"]) == pytest.approx(expected, abs=1e-6)


def _face_area_3d(body, face_id):
    """面の面積（3D）。wind に依存しない Newell 式で求める。"""
    import fifth

    v = np.array(body["vertices"], dtype=float)
    faces = fifth.orient_faces_outward(v, [[int(i) for i in f] for f in body["faces"]])
    acc = np.zeros(3)
    face = faces[face_id]
    for i in range(len(face)):
        acc += np.cross(v[face[i]], v[face[(i + 1) % len(face)]])
    return float(np.linalg.norm(acc)) / 2.0


def test_net_faces_do_not_overlap():
    from shapely.geometry import Polygon

    for builder in (cube_body, lambda: prism_body(4), l_shape_body):
        data = post(builder(), top_n=3).json()
        for net in data["nets"]:
            polys = [Polygon(f["vertices_2d"]) for f in net["faces"]]
            for i in range(len(polys)):
                for j in range(i + 1, len(polys)):
                    assert polys[i].intersection(polys[j]).area == pytest.approx(0.0, abs=1e-6)


def test_metrics_are_consistent_with_the_returned_geometry():
    data = post(cube_body(), top_n=11).json()
    for net in data["nets"]:
        xs = [p[0] for f in net["faces"] for p in f["vertices_2d"]]
        ys = [p[1] for f in net["faces"] for p in f["vertices_2d"]]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        assert net["metrics"]["bounding_box"]["width"] == pytest.approx(w)
        assert net["metrics"]["bounding_box"]["height"] == pytest.approx(h)
        assert net["metrics"]["bounding_box_area"] == pytest.approx(w * h)
        assert net["metrics"]["perimeter"] > 0.0


# --------------------------------------------------------------------------
# 入力の巻き順（上向き/下向きの正規化）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: [list(reversed(x)) for x in f], id="all-reversed"),
        pytest.param(lambda f: [x if i % 2 else list(reversed(x)) for i, x in enumerate(f)],
                     id="alternating"),
    ],
)
def test_result_is_independent_of_face_winding(mutate):
    """呼び出し側は巻き順を意識しなくてよい。"""
    body = cube_body()
    canonical = post(body, top_n=11).json()
    body["faces"] = mutate(body["faces"])
    other = post(body, top_n=11).json()
    assert other["total_unique_nets"] == canonical["total_unique_nets"]
    assert [n["metrics"] for n in other["nets"]] == [n["metrics"] for n in canonical["nets"]]


def test_result_is_independent_of_face_list_order():
    """面の並び順を変えても、同じ展開図が同じ指標で返る（根の選択だけが変わりうる）。"""
    body = cube_body()
    canonical = post(body, top_n=11).json()
    body["faces"] = body["faces"][::-1]
    other = post(body, top_n=11).json()
    assert other["total_unique_nets"] == canonical["total_unique_nets"]
    key = lambda n: (n["metrics"]["bounding_box_area"], n["metrics"]["perimeter"])
    assert sorted(map(key, other["nets"])) == sorted(map(key, canonical["nets"]))


def test_vertex_order_does_not_matter():
    """頂点の並べ替えと面の頂点順序の回転は結果を変えない。"""
    base = cube_body()
    reference = post(base, top_n=11).json()["total_unique_nets"]
    perm = [5, 0, 7, 2, 3, 1, 6, 4]
    shuffled = {
        "vertices": [base["vertices"][i] for i in perm],
        "faces": [[perm.index(i) for i in face] for face in base["faces"]],
    }
    assert post(shuffled).json()["total_unique_nets"] == reference


# --------------------------------------------------------------------------
# 入力検証（422）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, message",
    [
        ({"vertices": [[0, 0, 0]], "faces": [[0]]}, "4"),
        ({"vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
          "faces": [[0, 1, 2], [0, 1, 2]]}, "4"),
        ({"vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
          "faces": [[0, 1, 2], [0, 2, 3], [0, 1], [1, 2, 3]]}, "3"),
        ({"vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
          "faces": [[0, 1, 2], [0, 2, 9], [0, 1, 3], [1, 2, 3]]}, "範囲外"),
        ({"vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
          "faces": [[0, 1, 1], [1, 2, 3], [0, 2, 3], [0, 1, 2]]}, "複数回"),
    ],
)
def test_invalid_solid_is_rejected_with_422(body, message):
    r = post(body)
    assert r.status_code == 422, r.text
    assert message in detail_msg(r)


def test_open_surface_is_rejected():
    """立方体の側面を1枚外すと境界ができるので 422。"""
    body = cube_body()
    body["faces"] = body["faces"][:4]
    r = post(body)
    assert r.status_code == 422
    assert "共有" in detail_msg(r)


def test_non_planar_face_is_rejected():
    body = cube_body()
    body["vertices"][7] = [0.0, 1.0, 2.0]
    r = post(body)
    assert r.status_code == 422
    assert "平面" in detail_msg(r)


def test_disconnected_solid_is_rejected():
    """面が2つの離れた立方体だと、展開図を構成できない。"""
    a = cube_body()
    b = cube_body(5.0, 5.0, 5.0)
    merged = {
        "vertices": a["vertices"] + b["vertices"],
        "faces": a["faces"] + [[i + 8 for i in f] for f in b["faces"]],
    }
    r = post(merged)
    assert r.status_code == 422
    assert "連結" in detail_msg(r)


def test_too_many_spanning_trees_is_rejected():
    """全域木が上限を超える立体は、列挙せずに 422 で返す。

    6角柱は 8100 通りで通るが、7角柱は上限を超える。
    """
    import fifth

    assert fifth.count_spanning_trees(
        fifth.build_adjacency_graph(prism_body(6)["faces"]), 8
    ) == 8100
    assert post(prism_body(6)).status_code == 200

    n_trees = fifth.count_spanning_trees(
        fifth.build_adjacency_graph(prism_body(7)["faces"]), 9
    )
    assert n_trees > api_server.MAX_SPANNING_TREES
    r = post(prism_body(7))
    assert r.status_code == 422
    assert f"{n_trees:,}" in detail_msg(r)
    assert f"{api_server.MAX_SPANNING_TREES:,}" in detail_msg(r)


def test_cap_rejection_happens_before_enumeration(monkeypatch):
    """上限超過は列挙せずに 422 で返す（重い計算を無駄にしない）。"""
    import fifth

    def boom(*_a, **_k):
        raise AssertionError("列挙してはいけない")

    monkeypatch.setattr(fifth, "enumerate_spanning_trees", boom)
    r = post(prism_body(7))
    assert r.status_code == 422
    assert "上限" in detail_msg(r)


def test_malformed_vertices_are_rejected():
    assert post({"vertices": [[0, 0], [1, 0], [1, 1]], "faces": [[0, 1, 2]]}).status_code == 422
    assert post({"vertices": [[0, 0, 0], [1, 0]], "faces": [[0, 1]]}).status_code == 422
    assert post({"vertices": "nope", "faces": [[0]]}).status_code == 422


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_vertices_are_rejected(bad):
    """NaN / Inf は JSON としては不正なので、json= ではなく生の文字列で送る。"""
    body = cube_body()
    raw = json.dumps(body).replace("[0, 0, 0]", f"[0, 0, {bad}]", 1)
    assert bad in raw
    r = post(None, content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert "有限" in detail_msg(r)


def test_missing_fields_are_rejected():
    assert client.post(URL, json={"vertices": [[0, 0, 0]]}).status_code == 422
    assert client.post(URL, json={"faces": [[0, 1, 2]]}).status_code == 422
    assert client.post(URL, json={}).status_code == 422


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"dimensions": {"width": 1, "height": 1, "depth": 1}},
                     id="old-spec-field"),
        pytest.param({"w": 1, "h": 1, "d": 1}, id="old-query-params"),
        pytest.param({"sortby": "area"}, id="typo"),
        pytest.param({"top_n": 1, "extra": True}, id="unrelated"),
    ],
)
def test_unknown_fields_are_rejected_not_silently_dropped(extra):
    """未知フィールドは 422。typo や旧仕様フィールドを黙って捨てないため。"""
    r = post(cube_body(), **extra)
    assert r.status_code == 422, r.text
    assert "Extra inputs" in r.text or "extra" in r.text.lower()


def test_openapi_schema_forbids_extra_fields():
    schema = client.get("/openapi.json").json()["components"]["schemas"]["SolidRequest"]
    assert schema.get("additionalProperties") is False


def test_openapi_documents_422_error_shape():
    """422 の形は OpenAPI にも明記する（/docs を見ただけで気づけるように）。"""
    responses = client.get("/openapi.json").json()["paths"][URL]["post"]["responses"]
    assert "422" in responses
    schema = responses["422"]["content"]["application/json"]["schema"]
    assert schema["required"] == ["detail"]
    items = schema["properties"]["detail"]["items"]
    assert set(items["required"]) == {"type", "loc", "msg"}
    assert {"spanning_trees", "limit"} <= set(items["properties"])


# --------------------------------------------------------------------------
# エラーの構造
# --------------------------------------------------------------------------


def test_error_detail_is_always_a_list_of_objects():
    """422 の detail は、Pydantic の検証エラーでも自前のエラーでも配列。"""
    # Pydantic 側（フィールド欠落）
    r = client.post(URL, json={"vertices": [[0, 0, 0]]})
    assert r.status_code == 422
    pydantic_detail = r.json()["detail"]
    assert isinstance(pydantic_detail, list)
    assert {"type", "loc", "msg"} <= set(pydantic_detail[0])

    # 本 API 側（立体の不正）
    r = post({"vertices": [[0, 0], [1, 0], [1, 1]], "faces": [[0, 1, 2]]})
    assert r.status_code == 422
    own_detail = r.json()["detail"]
    assert isinstance(own_detail, list)
    assert {"type", "loc", "msg"} <= set(own_detail[0])
    assert own_detail[0]["loc"] == ["body"]


def test_input_errors_expose_machine_readable_type():
    """入力エラーは msg（人間向け）と type（機械可読）を分けて返す。"""
    body = cube_body()
    body["faces"] = body["faces"][:4]  # 境界を作る
    err = post(body).json()["detail"][0]
    assert err["type"] == "boundary_or_non_manifold"
    assert "共有" in err["msg"]


def test_spanning_tree_cap_error_is_structured():
    """上限超過は入力エラーと type で区別でき、木数と上限が数値で入る。

    入力自体は正当なので、クライアントは type で「入力ミス」と
    「計算量の上限」を分けて処理できる。
    """
    r = post(prism_body(7))
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["type"] == "too_many_spanning_trees"
    assert err["limit"] == api_server.MAX_SPANNING_TREES
    assert isinstance(err["spanning_trees"], int)
    assert err["spanning_trees"] > err["limit"]


# --------------------------------------------------------------------------
# make_nets.py のプリセット
# --------------------------------------------------------------------------


def test_every_make_nets_preset_is_a_valid_solid():
    """お試し CLI のプリセットはすべて正しい立体（少なくとも API で 422 にならない）。"""
    import make_nets

    assert set(make_nets.SOLIDS) == {
        "box", "tetra", "prism3", "prism4", "prism5",
        "pyr4", "pyr5", "lshape", "cutout",
    }
    for name, builder in make_nets.SOLIDS.items():
        v, f = builder()
        vertices, faces = api_server._build_solid(
            api_server.SolidRequest(vertices=v.tolist(), faces=f)
        )
        assert len(faces) == len(f), name
        assert vertices.shape == (len(v), 3), name


def test_api_and_cli_compute_the_same_perimeter():
    """API と CLI が同じ数字を出す（PR3 で指標の定義を fifth に一元化した）。

    それまでは API が「周長合計 − 2×折り曲げ軸」、CLI が「外接矩形の周長
    2*(w+h)」という**別の量**を計算していた。両者が同じ fifth.net_metrics
    を呼んでいることを確認する。
    """
    import make_nets

    import fifth

    v, f = make_nets.box(1, 1, 3)
    r = client.post(
        URL,
        json={
            "vertices": [[float(c) for c in p] for p in v.tolist()],
            "faces": f,
            "top_n": 5,
            "sort_by": "perimeter",
        },
    )
    assert r.status_code == 200, r.text[:400]
    api_perimeters = [net["metrics"]["perimeter"] for net in r.json()["nets"]]
    assert api_perimeters

    # 同じ展開図集合を CLI 側で評価して一致を確認する
    vertices, faces = api_server._build_solid(
        api_server.SolidRequest(vertices=v.tolist(), faces=f)
    )
    normals = fifth.compute_normals(vertices, faces)
    shared_of = fifth.build_shared_of(faces)
    trees = fifth.enumerate_spanning_trees(
        fifth.build_adjacency_graph(faces), len(faces)
    )
    nets = list(fifth.iter_unique_nets(vertices, faces, trees, normals))
    scored = sorted(
        (
            (
                fifth.net_metrics(polys, tree, vertices, shared_of),
                polys,
            )
            for tree, polys in nets
        ),
        key=lambda s: s[0].perimeter,
    )
    assert [m.perimeter for m, _ in scored[: len(api_perimeters)]] == pytest.approx(
        api_perimeters
    )


def test_perimeter_is_not_the_bounding_box_perimeter():
    """応答の perimeter が 2*(w+h) ではないことの直接確認。

    外周長は「全面の周長合計 − 2×折り曲げ軸」。外接矩形の周長とは別物。
    """
    profile = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
    vertices = [[x, y, float(z)] for z in (0.0, 1.0) for x, y in profile]
    faces = [[5, 4, 3, 2, 1, 0], list(range(6, 12))]
    faces += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]

    r = client.post(URL, json={"vertices": vertices, "faces": faces, "top_n": 1})
    assert r.status_code == 200, r.text[:400]
    m = r.json()["nets"][0]["metrics"]
    bbox_perimeter = 2.0 * (m["bounding_box"]["width"] + m["bounding_box"]["height"])
    assert m["perimeter"] != pytest.approx(bbox_perimeter, abs=1e-9)


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------


def test_cors_preflight_is_allowed_for_configured_origin():
    r = client.options(
        URL,
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_allows_configured_origins_on_actual_request():
    r = post(cube_body(), headers={"Origin": "http://127.0.0.1:3000"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


def test_cors_is_not_applied_to_unknown_origin():
    r = post(cube_body(), headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers
