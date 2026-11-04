"""リボングラフ同型による一意数と、平面図形の合同による一意数の一致。

2 つはまったく別の定義なので、片方だけのバグはここで落ちる。
重い計算は :mod:`net_test_cache` を経由して 1 回だけ走らせる。
"""

import pytest

import make_nets
import net_test_cache


def _geometric_unique_nets(builder):
    v, f = builder()
    return net_test_cache.geometric_unique_count(v, f)


def _ribbon_unique_nets(builder):
    v, f = builder()
    return len(net_test_cache.unique_nets(v, f))


CASES = [
    pytest.param(lambda: make_nets.box(1, 1, 1), 11, id="cube"),
    pytest.param(lambda: make_nets.box(1, 2, 3), 54, id="box_1x2x3"),
    pytest.param(lambda: make_nets.box(2, 2, 1), 29, id="box_2x2x1"),
    pytest.param(make_nets.tetrahedron, 2, id="tetrahedron"),
    pytest.param(lambda: make_nets.prism(3), 9, id="triangular_prism"),
    pytest.param(lambda: make_nets.prism(4), 29, id="square_prism"),
    pytest.param(lambda: make_nets.prism(5), 99, id="pentagonal_prism"),
    pytest.param(lambda: make_nets.pyramid(4), 8, id="square_pyramid"),
    pytest.param(lambda: make_nets.pyramid(5), 15, id="pentagonal_pyramid"),
    # 以下の 2 つは最も重いケース（どちらも全域木 8100 本、実測 7〜9 秒）。
    # 正規形の粒度が崩れると軽い立体のテストでは気づけないので、必ず含める。
    pytest.param(make_nets.l_shape, 1516, id="L-shape"),
    pytest.param(lambda: make_nets.prism(6), 354, id="hexagonal_prism"),
]


@pytest.mark.parametrize(("builder", "expected"), CASES)
def test_ribbon_dedup_agrees_with_geometric_congruence(builder, expected):
    """リボングラフ同型による一意数が、平面図形の合同による一意数と一致する。"""
    assert _geometric_unique_nets(builder) == expected
    assert _ribbon_unique_nets(builder) == expected
