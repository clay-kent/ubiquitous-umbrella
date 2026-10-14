"""スケールの不変性を守る回帰テスト。

同じ立体を 1e-10 倍まで縮めても、一意な展開図の数と折り曲げ軸が変わらないことを
固定する。PR2 以降のリファクタで退化しても落ちるように、スケール掃引は
使い捨ての計測ではなく通常実行に含めたテストとして残す。

⚠️ 2つの主張は別物です。混ぜないこと。

  * 一意な展開図の**数** … 1e-10 まで検証済み（このファイル）
  * ``_fold_axis_2d`` の**値** … 1e-8 までしか検証していない

``_fold_axis_2d`` は2Dレイアウト座標を ``round(9)`` 桁に丸めて頂点を突き合わせる。
丸め方が絶対精度なので、展開図が大きい立体ほど早く崩れる。数だけのテストが
1e-10 のレスポンスの正しさまで保証したわけではない。
"""

import pytest

import api_server
import fifth
import make_nets

# 軽い立体も重い立体も、実測では 5 スケールすべてで一致した。
# 1e-10 を超えたら「検証していない領域」ではなく、確実に壊れている。
SCALES = [1.0, 1e-3, 1e-6, 1e-8, 1e-10]

# 全域木が 8100 本あるため 1 回の走査に数秒かかる。この 2 つだけ印を付けて
# `-m "not slow"` で外せるようにする。落とすのは禁止（過合併を見落とすのはこの 2 つ）。
HEAVY = {"lshape", "prism6"}

SOLIDS = [
    # (名前, ビルダ, scale=1 のときの一意数)
    ("cube", lambda: make_nets.box(1, 1, 1), 11),
    ("box1x2x3", lambda: make_nets.box(1, 2, 3), 54),
    ("box2x2x1", lambda: make_nets.box(2, 2, 1), 29),
    ("tetrahedron", make_nets.tetrahedron, 2),
    ("prism3", lambda: make_nets.prism(3), 9),
    ("prism4", lambda: make_nets.prism(4), 29),
    ("prism5", lambda: make_nets.prism(5), 99),
    ("pyr4", lambda: make_nets.pyramid(4), 8),
    ("pyr5", lambda: make_nets.pyramid(5), 15),
    ("lshape", make_nets.l_shape, 1516),
    ("prism6", lambda: make_nets.prism(6), 354),
]


def _params(names):
    return [
        pytest.param(builder, scale, expected, id=f"{name}@{scale:g}")
        for name, builder, expected in SOLIDS
        if name in names
        for scale in SCALES
    ]


def _unique_count(builder, scale):
    """立体を scale 倍して、一意な展開図の数を数える。"""
    v, f = builder()
    v = v.astype(float) * scale
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = fifth.enumerate_spanning_trees(edges, len(faces))
    return len(list(fifth.iter_unique_nets(v, faces, trees, normals)))


@pytest.mark.parametrize(
    ("builder", "scale", "expected"),
    _params({n for n, _, _ in SOLIDS} - HEAVY),
)
def test_unique_net_count_is_scale_invariant(builder, scale, expected):
    """座標をいくら縮めても、一意な展開図の数は変わらない。

    これが落ちるのは 3 つある。辺長の丸めがスケール依存・面法線の退化管理が
    絶対値・重なり判定の閾値が絶対値。どれか 1 つに戻しても落ちる。
    """
    assert _unique_count(builder, scale) == expected


@pytest.mark.slow
@pytest.mark.parametrize(("builder", "scale", "expected"), _params(HEAVY))
def test_unique_net_count_is_scale_invariant_slow(builder, scale, expected):
    """上の重い 2 立体（lshape / prism6）。

    一意数が 1516 / 354 と桁が違うので、正規形の粒度が崩れると
    軽い立体のテストでは気づけない。``-m "not slow"`` で外せるように分離しただけ。
    """
    assert _unique_count(builder, scale) == expected


# --------------------------------------------------------------------------
# 実装の順序を守るガード。どちらも 1 つの表明しかない。
# --------------------------------------------------------------------------


def test_normalisation_precedes_rounding():
    """丸めてから正規化してはいけない。

    順序を逆にすると、1e-5 程度の立体では全辺が同じ値に潰れて
    box(1,1.5,2) が立方体と同じ 11 枚になってしまう（54 が期待値）。
    """
    assert _unique_count(lambda: make_nets.box(1, 1.5, 2), 1e-6) == 54


@pytest.mark.parametrize("scale", [1e-8, 1e-10])
def test_face_normal_has_no_floor(scale):
    """``face_normal`` の退化管理に ``max(1.0, s)`` の下限を置いてはいけない。

    下限を置き、小さいスケールを 1.0 の床に貼り付けると、
    辺長 1e-8 の正当な立方体を ``degenerate_face`` と誤判定する。
    """
    v, f = make_nets.box(1, 1, 1)
    v = v.astype(float) * scale
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    assert len(normals) == len(faces)
    for n in normals:
        assert float((n * n).sum()) ** 0.5 == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# fold_axis_2d の主張（一意数とは別）
# --------------------------------------------------------------------------


def _fold_axes(builder, scale):
    """全展開図の折り曲げ軸を、展開図の外接矩形で正規化して集める。

    正規化しないと、スケールだけ違う同一の展開図を数値比較できない。
    丸め自体は load-bearing なので残し、正規化の**後**に丸める。
    """
    v, f = builder()
    v = v.astype(float) * scale
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = fifth.enumerate_spanning_trees(edges, len(faces))

    axes = []
    for tree, polys in fifth.iter_unique_nets(v, faces, trees, normals):
        ext = max(
            max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in polys
        )
        parent_map = api_server.get_layout_parent_relation(faces, tree)
        for face_id, parent_id in enumerate(parent_map):
            if parent_id is None:
                continue
            axis = api_server._fold_axis_2d(polys, face_id, parent_id)
            if axis is None:
                continue
            axes.append(tuple((round(x / ext, 9), round(y / ext, 9)) for x, y in axis))
    return axes


@pytest.mark.parametrize(
    "builder",
    [
        lambda: make_nets.box(1, 1, 1),
        lambda: make_nets.box(1, 1.5, 2),
        make_nets.tetrahedron,
        lambda: make_nets.prism(4),
        lambda: make_nets.prism(5),
    ],
    ids=["cube", "box1x1.5x2", "tetra", "prism4", "prism5"],
)
def test_fold_axis_is_correct_down_to_1e_8(builder):
    """折り曲げ軸は 1e-8 まで全立体で正しい。

    ⚠️ v3 計画は「1e-9 まで」と書いたが、実測は 1e-8 が限界だった。
    ``round(9)`` は2Dレイアウト座標という「展開図の大きさ」に依存する値に
    絶対精度を掛けているので、立体がデカいほど早く崩れる。
    box(1,1.5,2)（展開図が大きい）は 1e-9、prism6（最も大きい）も 1e-9 で既に壊れる。
    つまり 1e-9 は精度の限界ではなく偶然の境界で、この下限は置けない。
    """
    baseline = _fold_axes(builder, 1.0)
    assert baseline  # 空だと空振りになるので、何か出ていることを明示する
    assert _fold_axes(builder, 1e-8) == baseline


@pytest.mark.xfail(strict=True, reason="既知の未修正バグ。PR2 の bbox 正規化で直る")
@pytest.mark.parametrize(
    ("builder", "scale"),
    [
        (lambda: make_nets.box(1, 1.5, 2), 1e-9),  # 14本が見つからず、4本が誤り
        (lambda: make_nets.box(1, 1, 1), 1e-10),  # 41本が誤り
    ],
    ids=["box1x1.5x2@1e-9", "cube@1e-10"],
)
def test_fold_axis_below_1e_8_is_still_broken(builder, scale):
    """1e-8 より小さいと ``_fold_axis_2d`` が壊れることを、.Rectangle として固定する。

    strict=True なので PR2 が bbox 正規化を入れて直すと **このテストが落ちる**。
    つまり「直ったら xfail を消す」という合図であり、修正を忘れないまま残せる。
    """
    assert _fold_axes(builder, scale) == _fold_axes(builder, 1.0)