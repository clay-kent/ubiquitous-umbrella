"""fifth.py（展開図生成コア）のユニットテスト。"""

import itertools
import math
import random
import re

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon

import fifth

# --------------------------------------------------------------------------
# テスト用の立体
# --------------------------------------------------------------------------


def cube(w=1.0, h=1.0, d=1.0):
    v = np.array(
        [
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
            [0, 0, d], [w, 0, d], [w, h, d], [0, h, d],
        ],
        dtype=float,
    )
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


def tetrahedron():
    v = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return v, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]


def prism(n, r=1.0, h=1.0):
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
    return np.array(v, dtype=float), f


def pyramid(n, r=1.0, h=1.0):
    v = [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ]
    v += [[0.0, 0.0, h]]
    f = [list(range(n - 1, -1, -1))] + [[i, (i + 1) % n, n] for i in range(n)]
    return np.array(v, dtype=float), f


def l_shape():
    """非凸の L 字柱。凹角（内側の二面角が 270 度になる辺）が含まれる。"""
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
    v = [[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts]
    f = [list(range(5, -1, -1)), list(range(6, 12))]
    f += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    return np.array(v, dtype=float), f


def cutout():
    """立方体を貫通する四角い穴つき（genus=1）。上面と下面は4枚の四角に分割する。"""
    outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
    inner = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    loop = outer + inner
    top = len(loop)  # 上面の頂点は top 番目から始まる
    v = [[x, y, z] for z in (0.0, 1.0) for x, y in loop]
    f = []
    for k in range(4):
        j = (k + 1) % 4
        f.append([k, j, top + j, top + k])                      # 外側の側面
        f.append([4 + k, 4 + j, top + 4 + j, top + 4 + k])      # 穴の側面
    for base, flip in ((0, False), (top, True)):
        for k in range(4):
            j = (k + 1) % 4
            quad = [base + k, base + j, base + 4 + j, base + 4 + k]
            f.append(list(reversed(quad)) if flip else quad)
    return np.array(v, dtype=float), f


def _signed_volume(vertices, faces, normals):
    """発散定理による符号付き体積（法線が外向きなら正）。"""
    total = 0.0
    for face, n in zip(faces, normals):
        pts = vertices[face]
        a = pts[0]
        for i in range(1, len(face) - 1):
            total += float(np.dot(np.cross(pts[i] - a, pts[i + 1] - a), n))
    return total / 6.0


def _neighbors(faces):
    """build_adjacency_graph は辺のリストを返すので、隣接リストに読み替える。"""
    nbrs = [[] for _ in range(len(faces))]
    for u, v in fifth.build_adjacency_graph(faces):
        nbrs[u].append(v)
        nbrs[v].append(u)
    return nbrs


# パラメータライズ用に「関数を返す」ラッパーを用意する
tetrahedron_b = tetrahedron
cube_b = cube
prism3_b = lambda: prism(3)
prism4_b = lambda: prism(4)
prism5_b = lambda: prism(5)
prism6_b = lambda: prism(6)
pyramid4_b = lambda: pyramid(4)
pyramid5_b = lambda: pyramid(5)
l_shape_b = l_shape
cutout_b = cutout


# --------------------------------------------------------------------------
# validate_solid / orient_faces_outward
# --------------------------------------------------------------------------


def test_valid_solids_are_accepted():
    for name, builder in [
        ("cube", cube_b),
        ("tetra", tetrahedron_b),
        ("prism3", prism3_b),
        ("prism5", prism5_b),
        ("pyr4", pyramid4_b),
        ("L-shape", l_shape_b),
        ("cutout", cutout_b),
    ]:
        v, f = builder()
        fifth.validate_solid(v, f)  # 例外が出なければ妥当


def test_face_with_fewer_than_three_vertices_is_rejected():
    v, _ = cube()
    with pytest.raises(ValueError, match="3未満"):
        fifth.validate_solid(v, [[0, 1, 2], [0, 2, 3], [0, 1], [1, 2, 3]])


def test_out_of_range_vertex_index_is_rejected():
    v, _ = cube()
    with pytest.raises(ValueError, match="範囲外"):
        fifth.validate_solid(v, [[0, 1, 2], [0, 2, 9], [0, 1, 5], [1, 2, 3]])


def test_repeated_vertex_in_a_face_is_rejected():
    v, _ = cube()
    with pytest.raises(ValueError, match="複数回"):
        fifth.validate_solid(v, [[0, 1, 1], [1, 2, 3], [0, 2, 3], [0, 1, 2]])


def test_non_planar_face_is_rejected():
    v, f = cube()
    v = v.copy()
    v[7] = [0.0, 1.0, 2.0]  # 面の1頂点だけを z 方向に動かすと平面でなくなる
    with pytest.raises(ValueError, match="平面"):
        fifth.validate_solid(v, f)


def test_open_surface_is_rejected():
    # 立方体の側面を1枚外すと境界（非多様体）になる
    v, f = cube()
    with pytest.raises(ValueError, match="共有"):
        fifth.validate_solid(v, f[:4])


def test_orient_faces_outward_is_winding_independent():
    """呼び出し側は巻き順を意識しなくてよい。"""
    v, f = cube()
    canonical = fifth.orient_faces_outward(v, [list(face) for face in f])

    flipped = [list(reversed(face)) for face in f]
    assert fifth.orient_faces_outward(v, flipped) == canonical

    mixed = [list(face) if i % 2 else list(reversed(face)) for i, face in enumerate(f)]
    assert fifth.orient_faces_outward(v, mixed) == canonical


@pytest.mark.parametrize(
    "builder",
    [cube_b, tetrahedron_b, prism4_b, pyramid4_b, l_shape_b, cutout_b],
    ids=["cube", "tetra", "prism4", "pyr4", "L-shape", "cutout"],
)
def test_orient_faces_outward_points_outward(builder):
    v, f = builder()
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    assert _signed_volume(v, faces, normals) > 0


def _reference_parents(n_faces, tree_edges):
    """layout_parent_relation の契約を独立に書き直した実装（比較用）。

    契約: 根は面0。木の無向隣接を幅優先に走査し、
    最初に訪れた未訪問の隣接面を親とする。根の親は None。
    """
    adj = [[] for _ in range(n_faces)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    parents = [None] * n_faces
    visited = {0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parents[v] = u
                queue.append(v)
    return parents


@pytest.mark.parametrize(
    "builder",
    [cube_b, tetrahedron_b, prism4_b, prism5_b, l_shape_b],
    ids=["cube", "tetra", "prism4", "prism5", "L-shape"],
)
def test_layout_parent_relation_matches_its_documented_contract(builder):
    """契約どおりに書き直した独立実装（比較用）。

    折り曲げ軸は「親と共有する辺」なので、ここが 1 つでも変わると
    API 応答の ``fold_axis_*`` と ``parent_face_id`` が黙って変わる。
    layout_tree と二重実装にしないための、意図的に独立した実装との突き合わせ。
    """
    _v, f = builder()
    edges = fifth.build_adjacency_graph(f)
    trees = list(fifth.enumerate_spanning_trees(edges, len(f)))
    assert trees  # 立体が空だと空振りになる

    for tree in trees:
        got = fifth.layout_parent_relation(len(f), tree)
        assert got == _reference_parents(len(f), tree)
        assert got[0] is None  # 根の親は None
        assert all(got[v] is not None for v in range(1, len(f)))


def test_layout_tree_and_parent_relation_agree():
    """layout_tree が親マップの唯一のソースであること。

    BFS 順が違えば回転の伝播順も変わり、展開図自体が変わる。
    """
    v, f = prism5_b()
    normals = fifth.compute_normals(v, f)
    edges = fifth.build_adjacency_graph(f)
    prep = fifth.prepare_layout(v, f, normals)
    for tree in list(fifth.enumerate_spanning_trees(edges, len(f)))[:64]:
        parents = fifth.layout_parent_relation(len(f), tree)
        # 各面は親と隣接している = 親マップが実際の隣接と整合している
        for face_id, parent_id in enumerate(parents):
            if parent_id is None:
                continue
            assert (parent_id, face_id) in edges or (face_id, parent_id) in edges
        assert len(fifth.layout_tree(v, f, normals, tree, prep)) == len(f)


def _bowtie():
    """2つの立方体が頂点1つを共有する立体。

    座標は 15 個だが頂点インデックスは 16 個ある。面はインデックスで
    頂点を指すので、同じ点に2枚の状態が被ってしまう。
    """
    v, f = cube()
    v2 = v + np.array([1.0, 0.0, 0.0])  # 頂点 (1,0,0) を共有する
    return np.vstack([v, v2]), f + [[i + 8 for i in face] for face in f]


def test_duplicate_vertex_is_rejected():
    """同じ座標を持つ頂点は不受当（2つの立方体が頂点1つを共有する bowtie）。"""
    v, f = _bowtie()
    with pytest.raises(ValueError) as ei:
        fifth.validate_solid(v, f)
    assert getattr(ei.value, "code", None) == "duplicate_vertex"


def test_duplicate_coordinates_are_reported_before_non_planarity():
    """頂点の重複は ``non_planar_face`` より先に報告される。

    重複座標の頂点を含む面は平面でなくなるので、順序を逆にすると利用者は
    「平面にするには？」という誤った結論に誘導される。
    """
    v, f = cube()
    v[7] = v[0]  # 頂点7 を 頂点0 と同じ座標にする（面インデックスは別）
    with pytest.raises(ValueError) as ei:
        fifth.validate_solid(v, f)
    assert ei.value.code == "duplicate_vertex"


def test_orient_faces_outward_rejects_boundary_input_loudly():
    """前条件を破った入力は黙って無視せず ValueError になる。

    ``orient_faces_outward`` から削除した ``if len(holders) != 2: continue``
    は、``validate_solid`` を先に通，才有この関数の到達不能な守護が存在する前提の守護だった。
    守護を外したので、破ったときはタプル展開が ValueError として現れる。
    このテストはその契約が文書どおりであることを固定する。
    """
    v, f = cube()
    # 側面を1枚外すと境界（非多様体）になる。validate_solid は弾く。
    open_faces = f[:4]
    with pytest.raises(ValueError, match="共有"):
        fifth.validate_solid(v, open_faces)

    # validate_solid を飛ばして orient_faces_outward に渡すと、
    # 黙って無視されるのではなく ValueError で落ちる（=見える失敗）。
    with pytest.raises(ValueError):
        fifth.orient_faces_outward(v, open_faces)


def test_layout_tree_rejects_a_non_spanning_tree():
    """全域木でない木は面番号を名指しして落ちる（前条件違反の可視化）。"""
    v, f = cube()
    normals = fifth.compute_normals(v, f)
    # (0,2) と (1,3) は立方体の実際の隣接だが、この2本だけでは2成分に分かれる。
    # 面0 から BFS すると到達できるのは {0,2} だけで、{1,3,4,5} は到達不能。
    forest = [(0, 2), (1, 3)]
    adjacency = fifth.build_adjacency_graph(f)
    assert all((u, v) in adjacency for u, v in forest), "前提: この2本は隣接している"

    with pytest.raises(ValueError, match="へ到達できませんでした") as ei:
        fifth.layout_tree(v, f, normals, forest)

    # メッセージは到達不能な面の番号を名指しする
    named = int(re.search(r"面(\d+)", str(ei.value)).group(1))
    assert named in {1, 3, 4, 5}, named


# --------------------------------------------------------------------------
# 法線・辺マップ・隣接グラフ
# --------------------------------------------------------------------------


def test_face_normal_uses_newell():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert np.allclose(fifth.face_normal(v, [0, 1, 2, 3]), [0, 0, 1])
    assert np.allclose(fifth.face_normal(v, [3, 2, 1, 0]), [0, 0, -1])


def test_edge_map_lists_both_faces_of_every_edge():
    _, f = cube()
    em = fifth.build_edge_map(f)
    assert len(em) == 12  # 立方体の辺は12本
    assert all(len(holders) == 2 for holders in em.values())


def test_adjacency_requires_a_shared_edge():
    _, f = cube()
    nbrs = _neighbors(f)
    # 立方体の各面は4面と隣接する（対向する面とは隣接しない）
    assert sorted(len(n) for n in nbrs) == [4, 4, 4, 4, 4, 4]
    # 共有辺は1本につき1エントリ
    assert len(fifth.build_adjacency_graph(f)) == 12
    with_edges = fifth.build_adjacency_with_edges(f)
    assert len(with_edges) == 12
    for u, ns in enumerate(nbrs):
        for v in ns:
            assert any({a, b} == {u, v} for a, b, _ in with_edges)


def test_l_shape_has_a_reflex_dihedral():
    """非凸の L 字柱の凹角棱では、二面角が 270 度になる。

    法線のなす角だけで求めると 90 度になってしまう（凸な棱しか扱えない）。
    """
    v, f = l_shape()
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    dihedrals = [
        fifth._dihedral_deg(v, faces, normals, u, w)
        for u, w, _ in fifth.build_adjacency_with_edges(faces)
    ]
    assert len(dihedrals) == 18
    reflex = [d for d in dihedrals if abs(d - 270.0) < 1e-6]
    assert len(reflex) == 1  # 凹んだ頂点に対応する棱は1本
    assert all(abs(d - 90.0) < 1e-6 for d in dihedrals if abs(d - 270.0) >= 1e-6)


@pytest.mark.parametrize(
    "builder, expected",
    [
        (cube_b, {90.0}),
        (prism3_b, {60.0, 90.0}),
        (prism4_b, {90.0}),
        (tetrahedron_b, {70.52877936550931}),  # 正四面体 = arccos(1/3)
    ],
    ids=["cube", "prism3", "prism4", "tetra"],
)
def test_interior_dihedral_angles(builder, expected):
    """立体の内側の二面角。立方体・三角柱・立方柱・正四面体で検算できる。"""
    v, f = builder()
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    got = {
        round(fifth._dihedral_deg(v, faces, normals, u, w), 6)
        for u, w, _ in fifth.build_adjacency_with_edges(faces)
    }
    assert got == {round(d, 6) for d in expected}


# --------------------------------------------------------------------------
# 全域木の数え上げと列挙
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, edges, n, expected",
    [
        ("path", [(i, i + 1) for i in range(5)], 6, 1),
        ("star", [(0, i) for i in range(1, 7)], 7, 1),
        ("cycle6", [(i, (i + 1) % 6) for i in range(6)], 6, 6),
        ("K4", [(a, b) for a in range(4) for b in range(a + 1, 4)], 4, 16),
        ("K5", [(a, b) for a in range(5) for b in range(a + 1, 5)], 5, 125),
        # 立方体の面グラフ：上下の面がそれぞれ4側面と隣接し、側面は環をつくる
        (
            "cube",
            [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
             (2, 3), (2, 5), (3, 4), (4, 5)],
            6,
            384,
        ),
        ("disconnected", [(0, 1), (2, 3), (3, 4), (5, 6)], 7, 0),
        ("vertex0 isolated", [(1, 2), (2, 3), (1, 3)], 7, 0),
        ("single node", [], 1, 1),
        ("no edges", [], 4, 0),
    ],
)
def test_count_spanning_trees(name, edges, n, expected):
    assert fifth.count_spanning_trees(edges, n) == expected


@pytest.mark.parametrize(
    "builder, expected",
    [
        (tetrahedron_b, 16),
        (prism3_b, 75),
        (prism4_b, 384),
        (prism5_b, 1805),
        (prism6_b, 8100),
    ],
    ids=["tetra", "prism3", "prism4", "prism5", "prism6"],
)
def test_count_spanning_trees_of_convex_solids(builder, expected):
    _v, f = builder()
    assert fifth.count_spanning_trees(fifth.build_adjacency_graph(f), len(f)) == expected


def _brute_force_trees(edges, n):
    """参照実装：辺の部分集合を全部試す。"""
    out = []
    for subset in itertools.combinations(edges, n - 1):
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(subset)
        if nx.is_tree(g):
            out.append(_normalize(subset))
    return sorted(out)


def _normalize(tree):
    """木の辺の集合を「向きを無視してソートした列」にする。"""
    return tuple(sorted(tuple(sorted((u, v))) for u, v in tree))


def test_enumerate_matches_brute_force_on_named_graphs():
    cases = [
        ("path", [(i, i + 1) for i in range(5)], 6),
        ("star", [(0, i) for i in range(1, 7)], 7),
        ("cycle", [(i, (i + 1) % 6) for i in range(6)], 6),
        ("K5", [(a, b) for a in range(5) for b in range(a + 1, 5)], 5),
        ("cube", fifth.build_adjacency_graph(cube()[1]), 6),
        ("self loops", [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)], 4),
        ("parallel edges", [(0, 1), (0, 1), (1, 2), (2, 0)], 3),
        ("disconnected", [(0, 1), (2, 3), (3, 4), (5, 6)], 7),
        ("isolated vertex0", [(1, 2), (2, 3), (1, 3)], 7),
    ]
    for name, edges, n in cases:
        got = sorted(_normalize(t) for t in fifth.enumerate_spanning_trees(edges, n))
        assert got == _brute_force_trees(edges, n), name
        assert len(got) == fifth.count_spanning_trees(edges, n), name


def test_enumerate_count_matches_matrix_tree_theorem():
    """列挙した木の本数が Matrix-Tree 定理の値と一致する。"""
    for builder in (cube_b, tetrahedron_b, prism4_b, prism5_b, l_shape_b):
        _v, f = builder()
        edges = fifth.build_adjacency_graph(f)
        assert len(list(fifth.enumerate_spanning_trees(edges, len(f)))) == (
            fifth.count_spanning_trees(edges, len(f))
        )


def test_enumerate_matches_brute_force_on_random_graphs():
    rng = random.Random(20260930)
    for _ in range(120):
        n = rng.randint(2, 7)
        m = rng.randint(0, min(14, n * (n - 1) // 2 + 2))
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
        got = sorted(_normalize(t) for t in fifth.enumerate_spanning_trees(edges, n))
        assert got == _brute_force_trees(edges, n), edges


def test_enumerate_produces_every_tree_exactly_once():
    edges = fifth.build_adjacency_graph(cube()[1])
    trees = list(fifth.enumerate_spanning_trees(edges, 6))
    assert len(trees) == 384
    assert len({_normalize(t) for t in trees}) == 384


def test_enumerate_is_lazy():
    """ジェネレータなので、列挙は遅延実行される。"""
    _v, f = cutout()
    gen = fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), len(f))
    assert len(next(gen)) == len(f) - 1  # 全域木の辺数は頂点数-1


def test_enumerate_distinguishes_parallel_edges():
    assert len(list(fifth.enumerate_spanning_trees([(0, 1), (0, 1), (1, 2)], 3))) == 2


# --------------------------------------------------------------------------
# 展開図のレイアウト
# --------------------------------------------------------------------------


def _newell_area(vertices, face):
    acc = np.zeros(3)
    m = len(face)
    for i in range(m):
        acc += np.cross(vertices[face[i]], vertices[face[(i + 1) % m]])
    return float(np.linalg.norm(acc)) / 2.0


@pytest.mark.parametrize(
    "builder",
    [cube_b, tetrahedron_b, prism5_b, pyramid5_b, l_shape_b, cutout_b],
    ids=["cube", "tetra", "prism5", "pyr5", "L-shape", "cutout"],
)
def test_layout_preserves_face_area(builder):
    """展開しても面積は変わらない（展開図では各面は全等である）。"""
    v, f = builder()
    normals = fifth.compute_normals(v, f)
    prep = fifth.prepare_layout(v, f, normals)
    edges = fifth.build_adjacency_graph(f)
    tree = next(iter(fifth.enumerate_spanning_trees(edges, len(f))))
    polys = fifth.layout_tree(v, f, normals, tree, prep)

    assert len(polys) == len(f)
    for i, poly in enumerate(polys):
        assert poly.area == pytest.approx(_newell_area(v, f[i]), abs=1e-9)
        assert len(poly.exterior.coords) - 1 == len(f[i])


def test_cube_layout_keeps_faces_orthogonal():
    """立方体の隣り合う面は直角に 接している。"""
    v, f = cube()
    normals = fifth.compute_normals(v, f)
    for u, nbrs in enumerate(_neighbors(f)):
        for w in nbrs:
            assert float(np.dot(normals[u], normals[w])) == pytest.approx(0.0, abs=1e-9)


def test_is_valid_layout_accepts_touching_but_not_overlapping_faces():
    unit = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    apart = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    sharing = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    assert fifth.is_valid_layout([unit, apart])
    assert fifth.is_valid_layout([unit, sharing])
    assert not fifth.is_valid_layout([unit, unit])


def test_is_valid_layout_rejects_partial_overlap():
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    assert not fifth.is_valid_layout([a, b])


def _old_bbox_perimeter(polys):
    """削除した CLI 側の周長定義 ``2*(w+h)``（外接矩形の周長）。

    旧実装は ``poly.exterior.xy`` から x と y を分けて集めていた。
    shapely 2.x の ``coords[0]`` は「最初の座標」であって x の配列では
    ないので、ここでは明示的に分解する。
    """
    xs = [x for p in polys for x, _ in p.exterior.coords]
    ys = [y for p in polys for _, y in p.exterior.coords]
    return 2.0 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))


@pytest.mark.parametrize(
    "builder",
    [cube_b, tetrahedron_b, prism4_b, prism5_b, l_shape_b],
    ids=["cube", "tetra", "prism4", "prism5", "L-shape"],
)
def test_perimeter_equals_face_sum_minus_folded_edges(builder):
    """外周長 = 全面の周長合計 − 2 × 折り曲げ軸の長さ合計。

    展開図の外周は「折ったときに内側になる棱」だけ短くなる。定義どおりの
    値が返っていることを、箱柱の 29 枚すべてで確認する。
    """
    v, f = builder()
    normals = fifth.compute_normals(v, f)
    shared_of = fifth.build_shared_of(f)
    edges = fifth.build_adjacency_graph(f)
    trees = fifth.enumerate_spanning_trees(edges, len(f))
    nets = list(fifth.iter_unique_nets(v, f, trees, normals))
    assert nets, "展開図が1つも無いと空振りになる"

    for tree, polys in nets:
        face_sum = sum(p.length for p in polys)
        glued = sum(
            float(np.linalg.norm(v[a] - v[b]))
            for u, w in tree
            for a, b in [shared_of[(u, w)]]
        )
        assert glued > 0, "全域木なので折り曲げ軸が1本以上ある"
        m = fifth.net_metrics(polys, tree, v, shared_of)
        assert m.perimeter == pytest.approx(face_sum - 2.0 * glued, abs=1e-9)


def test_unit_cube_net_perimeter_is_14():
    """単位立方体の展開図の外周長は全11枚とも 14。

    6枚の正方形の周長 4.0 を足して 24.0、折り曲げ軸 5本の長さ 1.0 を
    2 倍して引く。定義から手で計算できる値なので実装の検算になる。
    """
    v, f = cube()
    normals = fifth.compute_normals(v, f)
    shared_of = fifth.build_shared_of(f)
    trees = fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), len(f))
    nets = list(fifth.iter_unique_nets(v, f, trees, normals))

    assert len(nets) == 11
    for tree, polys in nets:
        assert fifth.net_metrics(polys, tree, v, shared_of).perimeter == pytest.approx(14.0)


def test_perimeter_is_not_the_bounding_box_perimeter():
    """外周長は外接矩形の周長以上になる（L 字型では凹みがあるので厳密に大きい）。

    凹みのない矩形なら等しい。``2*(w+h)`` は「外接矩形の周長」であって
    「展開図の外周長」ではない、というのが主張。
    """
    v, f = l_shape()
    normals = fifth.compute_normals(v, f)
    shared_of = fifth.build_shared_of(f)
    edges = fifth.build_adjacency_graph(f)
    trees = fifth.enumerate_spanning_trees(edges, len(f))
    nets = list(fifth.iter_unique_nets(v, f, trees, normals))
    assert nets, "展開図が1つも無いと空振りになる"

    strictly_bigger = 0
    for tree, polys in nets:
        m = fifth.net_metrics(polys, tree, v, shared_of)
        old = _old_bbox_perimeter(polys)
        assert m.perimeter >= old - 1e-9  # 多角形の周長は外接矩形の周長以上
        if m.perimeter > old + 1e-9:
            strictly_bigger += 1

    # 凹みのある展開図が1つでもあれば 2*(w+h) は外周長と違うと分かる
    assert strictly_bigger > 0, "L字柱なのに外周長と外接矩形の周長が一致した"


def test_net_metrics_bbox_area_matches_width_times_height():
    v, f = cube()
    normals = fifth.compute_normals(v, f)
    shared_of = fifth.build_shared_of(f)
    edges = fifth.build_adjacency_graph(f)
    trees = fifth.enumerate_spanning_trees(edges, len(f))
    tree, polys = next(iter(fifth.iter_unique_nets(v, f, trees, normals)))

    m = fifth.net_metrics(polys, tree, v, shared_of)
    assert m.bounding_box_area == pytest.approx(
        m.bounding_box_width * m.bounding_box_height
    )


# --------------------------------------------------------------------------
# 一意な展開図の抽出
# --------------------------------------------------------------------------


def _unique_nets(v, f):
    normals = fifth.compute_normals(v, f)
    edges = fifth.build_adjacency_graph(f)
    trees = fifth.enumerate_spanning_trees(edges, len(f))
    return list(fifth.iter_unique_nets(v, f, trees, normals))


@pytest.mark.parametrize(
    "name, builder, expected",
    [
        ("tetrahedron", tetrahedron_b, 2),
        ("cube", cube_b, 11),
        ("triangular prism", prism3_b, 9),
        ("square prism", prism4_b, 29),
        ("pentagonal prism", prism5_b, 99),
        ("square pyramid", pyramid4_b, 8),
        ("pentagonal pyramid", pyramid5_b, 15),
        ("L-shape", l_shape_b, 1516),
    ],
)
def test_unique_net_counts(name, builder, expected):
    """リボングラフ同型による一意な展開図の個数が既知値と一致する。"""
    v, f = builder()
    assert len(_unique_nets(v, f)) == expected


def test_cube_has_eleven_nets_not_112():
    """鏡映は同一視する。立方体の展開図は11種（回転・並進を無視）であり112個ではない。"""
    v, f = cube()
    assert len(_unique_nets(v, f)) == 11


def test_unique_nets_are_free_of_overlap():
    v, f = l_shape()
    for _, polys in _unique_nets(v, f):
        assert fifth.is_valid_layout(polys)


def test_unique_nets_are_fewer_than_all_trees():
    """重なる全域木が捨てられ、鏡映がまとめられるので一意数は全域木数より少ない。"""
    v, f = l_shape()
    n_trees = fifth.count_spanning_trees(fifth.build_adjacency_graph(f), len(f))
    assert n_trees == 8100
    assert len(_unique_nets(v, f)) == 1516


def test_errors_inside_the_tree_loop_propagate():
    """``iter_unique_nets`` のループ中の例外は握り潰されない。

    以前はループ全体を ``except ValueError: continue`` で包んでいたため、
    予算超過もその他の ValueError も「その木は捨てる」と解釈され、
    **静かに誤った一意数**を返していた。

    このテストは正常系ではなく guard である:
    ``_net_signature`` に注入して、-raising ことを確かめる。
    """
    v, f = cube()
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, f)
    trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), len(f)))

    boom = ValueError("-loop-boom-")
    original = fifth._net_signature

    calls = {"n": 0}

    def exploding(ctx, tree):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise boom
        return original(ctx, tree)

    fifth._net_signature = exploding
    try:
        with pytest.raises(ValueError, match="loop-boom"):
            list(fifth.iter_unique_nets(v, faces, iter(trees), normals))
    finally:
        fifth._net_signature = original


def test_search_budget_exceeded_raises_a_named_solid_error():
    """予算超過は ``None`` ではなく ``SolidError('search_budget_exceeded')``。

    ``None`` を返すと VF2 に落ち 있었いので、その戻り値を誰かが
    ``None`` を返すとフォールバックに落ちていたが、その戻り値を
    API ではこの例外が 422 の ``type: "search_budget_exceeded"`` になる。
    """
    ctx = fifth.RibbonContext(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        [[0, 1, 2, 3], [0, 1, 4]],
    )
    budget = [fifth.SEARCH_BUDGET + 1]
    with pytest.raises(ValueError) as ei:
        fifth._individualize_refine(ctx.nxt, [-1] * ctx.n_nodes, [False] * ctx.n_nodes, [0] * ctx.n_nodes, budget)

    assert isinstance(ei.value, fifth.SolidError)
    assert ei.value.code == "search_budget_exceeded"


def test_budget_headroom_is_large():
    """SEARCH_BUDGET は現実のプリセットに対して十分な余裕がある。

    余裕がなければ「到達しないから abort は安全」という前提が崩れる。
    """
    for builder in (cube_b, tetrahedron_b, prism4_b, prism5_b, l_shape_b):
        v, f = builder()
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, f)
        trees = fifth.enumerate_spanning_trees(
            fifth.build_adjacency_graph(faces), len(faces)
        )
        peak = {"v": 0}

        original = fifth._individualize_refine

        def counting(nxt, partner, flip, color, budget, _o=original):
            try:
                return _o(nxt, partner, flip, color, budget)
            finally:
                peak["v"] = max(peak["v"], budget[0])

        fifth._individualize_refine = counting
        try:
            list(fifth.iter_unique_nets(v, faces, trees, normals))
        finally:
            fifth._individualize_refine = original

        assert 0 < peak["v"] < fifth.SEARCH_BUDGET / 100, (builder, peak["v"])



def test_net_signature_merges_mirror_images():
    """鏡映は同一視されるので、正準形は自身と鏡像のうち小さい方をキーにする。"""
    v, f = prism(4)
    ctx = fifth.RibbonContext(v, f)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(face)) for face in f])
    trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), len(f)))

    keys = {min(fifth._net_signature(ctx, t), fifth._net_signature(mirror_ctx, t)) for t in trees}
    assert len(trees) == 384
    assert len(keys) == 29  # 立方柱の一意な展開図は29種


def test_net_signature_is_a_complete_invariant_for_the_cube():
    v, f = cube()
    ctx = fifth.RibbonContext(v, f)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(face)) for face in f])
    trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), 6))
    keys = {min(fifth._net_signature(ctx, t), fifth._net_signature(mirror_ctx, t)) for t in trees}
    assert len(keys) == 11  # 立方体は11種（鏡映同一視）
