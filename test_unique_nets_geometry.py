import itertools
import math

import pytest

import fifth
import make_nets


def _fingerprint(polys):
    """剛体移動・鏡映に不変な指紋: 展開図の全頂点対の距離をソートする。

    リボングラフ同型（fifth の正規形）とはまったく別の考え方で、
    「平面上でみた展開図そのものが合同か」だけで一意性を数える。
    """
    pts = []
    for p in polys:
        pts.extend(list(p.exterior.coords)[:-1])
    ds = sorted(
        round(math.dist(pts[i], pts[j]), 5)
        for i, j in itertools.combinations(range(len(pts)), 2)
    )
    return (len(pts), tuple(ds))


def _geometric_unique_nets(builder):
    v, f = builder()
    normals = fifth.compute_normals(v, f)
    edges = fifth.build_adjacency_graph(f)
    prep = fifth.prepare_layout(v, f, normals)
    fps = set()
    for tree in fifth.enumerate_spanning_trees(edges, len(f)):
        polys = fifth.layout_tree(v, f, normals, tree, prep)
        if fifth.is_valid_layout(polys):
            fps.add(_fingerprint(polys))
    return len(fps)


def _ribbon_unique_nets(builder):
    v, f = builder()
    normals = fifth.compute_normals(v, f)
    trees = fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f), len(f))
    return len(list(fifth.iter_unique_nets(v, f, trees, normals)))


@pytest.mark.parametrize(
    "name, builder, expected",
    [
        ("cube", lambda: make_nets.box(1, 1, 1), 11),
        ("box 1x2x3", lambda: make_nets.box(1, 2, 3), 54),
        ("box 2x2x1", lambda: make_nets.box(2, 2, 1), 29),
        ("tetrahedron", make_nets.tetrahedron, 2),
        ("triangular prism", lambda: make_nets.prism(3), 9),
        ("square prism", lambda: make_nets.prism(4), 29),
        ("pentagonal prism", lambda: make_nets.prism(5), 99),
        ("square pyramid", lambda: make_nets.pyramid(4), 8),
        ("pentagonal pyramid", lambda: make_nets.pyramid(5), 15),
    ],
    ids=lambda s: str(s).replace(" ", "_"),
)
def test_ribbon_dedup_agrees_with_geometric_congruence(name, builder, expected):
    """リボングラフ同型による一意数が、平面図形の合同による一意数と一致する。

    両者は別々の定義なので、片方だけのバグはここで落ちる。
    """
    assert _geometric_unique_nets(builder) == expected
    assert _ribbon_unique_nets(builder) == expected
