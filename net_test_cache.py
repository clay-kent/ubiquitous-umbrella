"""テスト専用の「重い展開図計算」キャッシュ。

同じ立体の一意な展開図を、テストファイルをまたいで何度も列挙しないためのもの。
本体（fifth.py / api_server.py）からは import しないこと。

なぜ必要か
----------
L 字柱と 6 角柱は面グラフの全域木が 8,100 本あり、1 回の
``iter_unique_nets`` に実測 7〜9 秒かかる。にもかかわらず、
「周長」「折る回数」「一意数」「API の形」などを別々のテストが
同じ立体で確かめるため、以前は同じ計算を 20 回以上繰り返していた。

キャッシュのキーは「頂点座標のバイト列 + 面の頂点列」。
座標を丸めないので、値が 1 ビットでも違えば別エントリになる
（テストの意味を変えないための保守的な選択）。同じ ``make_nets``
のビルダを共有しているファイル間ではヒットする。
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np

import fifth

# (vertices.shape, vertices.tobytes(), faces) のタプル。
SolidKey = tuple[tuple[int, ...], bytes, tuple[tuple[int, ...], ...]]


def solid_key(vertices: np.ndarray, faces: list[list[int]]) -> SolidKey:
    """立体をハッシュ可能なキーに落とす。座標は丸めない。"""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    return (v.shape, v.tobytes(), tuple(tuple(int(i) for i in face) for face in faces))


def _decode(key: SolidKey) -> tuple[np.ndarray, list[list[int]]]:
    shape, raw, faces = key
    # frombuffer は読み取り専用なので、呼び出し先が書き換えても
    # キャッシュを壊さないようコピーしてから渡す。
    vertices = np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()
    return vertices, [list(face) for face in faces]


@functools.cache
def _unique_nets(key: SolidKey) -> tuple[tuple[list[tuple[int, int]], list], ...]:
    vertices, faces = _decode(key)
    normals = fifth.compute_normals(vertices, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = fifth.enumerate_spanning_trees(edges, len(faces))
    return tuple(fifth.iter_unique_nets(vertices, faces, trees, normals))


def unique_nets(
    vertices: np.ndarray, faces: list[list[int]]
) -> list[tuple[list[tuple[int, int]], list]]:
    """一意な展開図 ``[(全域木, ポリゴン列), ...]`` を共有キャッシュ経由で返す。

    戻り値の内側はキャッシュと共有しているので、**書き換えないこと**。
    テストは読むだけ（``len`` / ``net_metrics`` / ``is_valid_layout``）に使う。
    """
    return list(_unique_nets(solid_key(vertices, faces)))


def polygon_fingerprint(polygons: list) -> tuple[int, tuple[float, ...]]:
    """剛体移動・鏡映に不変な指紋: 全頂点対の距離をソートしたもの。

    リボングラフ同型（``fifth._net_signature``）とはまったく別の考え方で、
    「平面上でみた展開図そのものが合同か」だけを見る。
    """
    points = [
        point
        for polygon in polygons
        for point in list(polygon.exterior.coords)[:-1]
    ]
    distances = sorted(
        round(math.dist(points[i], points[j]), 5)
        for i, j in itertools.combinations(range(len(points)), 2)
    )
    return (len(points), tuple(distances))


@functools.cache
def _geometric_fingerprints(key: SolidKey) -> frozenset[tuple[int, tuple[float, ...]]]:
    """全全域木を展開し、重ならないものの図形指紋を集める。"""
    vertices, faces = _decode(key)
    normals = fifth.compute_normals(vertices, faces)
    prep = fifth.prepare_layout(vertices, faces, normals)
    fingerprints: set[tuple[int, tuple[float, ...]]] = set()
    for tree in fifth.enumerate_spanning_trees(
        fifth.build_adjacency_graph(faces), len(faces)
    ):
        polygons = fifth.layout_tree(vertices, faces, normals, tree, prep)
        if fifth.is_valid_layout(polygons):
            fingerprints.add(polygon_fingerprint(polygons))
    return frozenset(fingerprints)


def geometric_unique_count(vertices: np.ndarray, faces: list[list[int]]) -> int:
    """平面図形の合同だけで数えた一意数（独立的な検算用）。"""
    return len(_geometric_fingerprints(solid_key(vertices, faces)))
