"""スケールの不変性を守る回帰テスト。

同じ立体を 1e-10 倍まで縮めても、一意な展開図の数と折り曲げ軸が変わらないことを
固定する。PR2 以降のリファクタで退化しても落ちるように、スケール掃引は
使い捨ての計測ではなくテストとして残す。重い 2 立体だけは ``slow`` を付け、
既定実行からは外す（``pytest -m ""`` では必ず回る）。

⚠️ 2つの主張は別物です。混ぜないこと。

  * 一意な展開図の**数** … 1e-10 まで検証済み（このファイル）
  * ``_fold_axis_2d`` の**値** … 1e-12 まで検証済み（このファイル）

どちらも 1e-10 より小さいスケールを断言しているが、
片方が成立してもう片方が成立する保証はない。
"""

import pytest

import fifth
import make_nets
import net_test_cache

# 実測では 1e-3 と 1e-8 を含む 5 スケールでも一致した。ここでは両端
# （等倍と 1e-10）と中間の 1 つだけを回す。1e-10 を超えたら
# 「検証していない領域」ではなく、確実に壊れている。
SCALES = [1.0, 1e-6, 1e-10]

# 全域木が 8100 本あるため 1 回の走査に数秒かかる。この 2 つは `slow` を
# 付けて既定実行から外す（CI の `pytest -m ""` では必ず回る）。
# 落とすのは禁止（過合併を見落とすのはこの 2 つ）。
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
    return len(net_test_cache.unique_nets(v, faces))


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
    軽い立体のテストでは気づけない。既定では除外され、
    CI の ``pytest -m ""`` で必ず回る。
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

    axes = []
    for tree, polys in net_test_cache.unique_nets(v, faces):
        ext = max(
            max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in polys
        )
        parent_map = fifth.layout_parent_relation(len(faces), tree)
        for face_id, parent_id in enumerate(parent_map):
            if parent_id is None:
                continue
            axis = fifth._fold_axis_2d(polys, face_id, parent_id)
            if axis is None:
                continue
            axes.append(tuple((round(x / ext, 9), round(y / ext, 9)) for x, y in axis))
    return axes


@pytest.mark.parametrize("scale", [1e-9, 1e-10, 1e-12])
@pytest.mark.parametrize(
    "builder",
    [
        lambda: make_nets.box(1, 1, 1),
        lambda: make_nets.box(1, 1.5, 2),
        make_nets.tetrahedron,
        lambda: make_nets.prism(3),
        lambda: make_nets.prism(4),
        lambda: make_nets.prism(5),
        lambda: make_nets.pyramid(4),
        lambda: make_nets.pyramid(5),
    ],
    ids=["cube", "box1x1.5x2", "tetra", "prism3", "prism4", "prism5", "pyr4", "pyr5"],
)
def test_fold_axis_is_scale_invariant(builder, scale):
    """折り曲げ軸は 1e-12 まで正しい（PR2 で bbox 正規化を入れてから）。

    PR1 の時点では 1e-8 が限界だった。``_fold_axis_2d`` が2Dレイアウト座標に
    絶対精度の ``round(9)`` を掛けていたため、展開図が物理的に大きい立体ほど
    早く崩れていた（box(1,1.5,2) は 1e-9 で軸が14本消えていた）。

    正規化を挟むと実測は 1e-14 まで通る。1e-12 まで断言して、
    1e-10 で確実に壊れないこと（当初の主張）を固定する。
    1e-16 では一意数自体が変わり始めるので、それ以上は検証しない。
    """
    baseline = _fold_axes(builder, 1.0)
    assert baseline  # 空だと空振りになるので、何か出ていることを明示する
    assert _fold_axes(builder, scale) == baseline


# --------------------------------------------------------------------------
# のりしろの側選択（一意数・折り曲げ軸とは別）
# --------------------------------------------------------------------------


def _tab_sides(builder, scale):
    """各展開図で、切れている辺をどちらの面に貼ったかを集める。"""
    v, f = builder()
    v = v.astype(float) * scale
    faces = fifth.orient_faces_outward(v, f)
    pair_to_edge = fifth._pair_to_edge(faces)
    return [
        {
            tuple(sorted((t.face_id, t.neighbor_face_id))): (
                t.face_id,
                t.neighbor_face_id,
            )
            for t in fifth.glue_tabs(polys, faces, tree, pair_to_edge, v)
        }
        for tree, polys in net_test_cache.unique_nets(v, faces)
    ]


@pytest.mark.slow
def test_glue_tab_sides_are_scale_invariant():
    """重なりコストが同値のとき、縮尺で貼る側が変わらないこと。

    のりしろの重なり面積はスケール^2 で変わる。厳密な ``<`` で比べると
    同値候補の差が浮動小数ノイズで逆転し、L 字柱 1e-6 倍では 1516 枚中
    280 枚で同じ辺の貼り付け面が反転していた。相対トレランスの同値判定と
    「同値なら面番号の小さい側」の規則を固定する。
    """
    assert _tab_sides(make_nets.l_shape, 1.0) == _tab_sides(make_nets.l_shape, 1e-6)