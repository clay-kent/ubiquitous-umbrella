import math
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import Polygon

# 辺長の正規化後の丸め桁数。10 桁だと浮動小数点の最後のビット差を
# 同一視できず、9 桁で「同じ辺長は必ず同じ色」に収まる。
EDGE_LEN_DECIMALS = 9

# 面法線の退化管理の相対許容値。面積(長さ^2)と同じ単位なので座標スケール^2 を掛ける。
# 絶対値 1e-12 だと辺長 1e-7 以下の正当な立体を退化和誤判定する。
FACE_NORMAL_REL_TOL = 1e-12

# 重なり判定の相対許容値（2つの面の和集合 bbox の対角の長さ^2 に対する比）。
# 精度指定ではなく「実重なりと浮動ノイズの間の空白に置く」ための値。
# 実測: 実重なりは最小 1.0、ダストは 1e-9 未満なので 9 桁の余裕がある。
OVERLAP_AREA_REL_TOL = 1e-7

# 個別化-細分の仕事量上限。これを超えると 422 で失敗させる。
# 根拠（実測）: どのプリセットでも最大でも 126 程度（prism5）。最も重い
# lshape でも数千程度なので 1500 倍以上の余裕があり、実入力では到達しない。
# 到達しないからこそ、唯一の妥当な停止条件は中断である
# （以前は「超えたら 1 枚捨てて VF2 同型判定に回る」フォールバックがあるが、
#   1500 倍の余裕に対して発動しない＝死んだ経路であり、消した）。
SEARCH_BUDGET = 200_000

# 色細分における「1 つの色」。初期値は (辺長, 接着の有無, 接着の向き) の
# タプルで、1 回目の細分で hash() の結果（int）に置き換わり、その後は
# 個別化でさらに int が挿入される。つまり**実行時に型が変わる**ので、
# 和集合で書くより Any のほうが正直。
Color = Any

# prepare_layout の戻り値。6 つのキーをまとめた dict で、使う側ごとに
# 必要なキーだけ引く。クラス化しても実際の問題は解決しないので dict のまま。
Layout = dict[str, Any]


class SolidError(ValueError):
    """入力された立体が不正なときに投げられる。

    message は人間向け（日本語）、code は機械可読な分類。HTTP エラーの
    ``detail`` に載せるとき、クライアントは ``code`` で分岐して
    ``message`` を表示に回せる。ValueError のサブクラスなので、
    既存の ``except ValueError`` はそのまま機能する。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_vertices(raw: object) -> np.ndarray:
    """API が受け取った頂点配列を numpy にし、形と有限性を検査する。

    不正はすべて SolidError にして、code で機械可読に区別できるようにする。
    呼び出し側は ``except SolidError`` だけで済み、422 への切り分けを
    1 箇所に集約できる。
    """
    try:
        vertices = np.array(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SolidError("vertices_invalid", f"vertices を解釈できません: {exc}") from exc

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise SolidError(
            "vertices_shape", "vertices は [[x, y, z], ...] の形式である必要があります。"
        )
    if not np.all(np.isfinite(vertices)):
        raise SolidError("non_finite_vertex", "vertices に有限でない値があります。")
    return vertices


def validate_solid(
    vertices: np.ndarray, faces: list[list[int]], *, planarity_tol: float = 1e-6
) -> None:
    """入力された立体の妥当性を検査し、不正なら SolidError を投げる。"""
    n_v = len(vertices)
    n_f = len(faces)
    if n_v < 4:
        raise SolidError("too_few_vertices", "立体には4頂点以上が必要です。")
    if n_f < 4:
        raise SolidError("too_few_faces", "立体には4面以上が必要です。")

    # 同じ座標を持つ頂点が別インデックスで存在する。
    # 面はインデックスで頂点を指すので、重複頂点は「同じ点に2枚の状態が被る」ことになり、
    # 面の隣接グラフが定義したものと食い違って非連結に見える。
    # 頂点を1つに統合してから入力しなければならない。
    #
    # これを先に弾かないと、原因と無関係な downstream のエラー
    # （例: 重複頂点を含む面が non_planar_face になる、bowtie が not_connected になる）
    # が返って、利用者は原因を推測できません。
    if len(np.unique(vertices, axis=0)) != n_v:
        raise SolidError(
            "duplicate_vertex",
            "同じ座標を持つ頂点が複数あります。頂点を1つにまとめてください。",
        )

    # 全頂点が原点なら zero_length_edge が先に弾くので 0 にはならない。
    scale = float(np.max(np.abs(vertices)))
    for f_idx, face in enumerate(faces):
        if len(face) < 3:
            raise SolidError("face_too_few_vertices", f"面{f_idx} の頂点数が3未満です。")
        if len(set(face)) != len(face):
            raise SolidError(
                "duplicate_face_vertex", f"面{f_idx} に同じ頂点が複数回含まれています。"
            )
        for v_idx in face:
            if not 0 <= v_idx < n_v:
                raise SolidError(
                    "vertex_index_out_of_range",
                    f"面{f_idx} の頂点インデックス {v_idx} が範囲外です。",
                )

        pts = vertices[face]
        for i in range(len(face)):
            if np.allclose(pts[i], pts[(i + 1) % len(face)], atol=1e-12):
                raise SolidError("zero_length_edge", f"面{f_idx} に長さ0の辺があります。")

        normal = face_normal(vertices, face)
        offsets = (pts - pts[0]) @ normal
        if float(np.max(np.abs(offsets))) > planarity_tol * scale:
            raise SolidError("non_planar_face", f"面{f_idx} は平面ではありません。")

    # 閉じた2多様体であること（各辺がちょうど2面に共有される）
    for edge, holders in build_edge_map(faces).items():
        if len(holders) != 2:
            raise SolidError(
                "boundary_or_non_manifold",
                f"辺{edge} が{len(holders)}個の面に共有されています。"
                "閉じていないか非多様体です。",
            )

    # 各頂点まわりの面が1つのサイクルをなすこと（面の飛び）を検出
    for v_idx in range(n_v):
        incident = [i for i, f in enumerate(faces) if v_idx in f]
        if len(incident) < 3:
            raise SolidError("vertex_underused", f"頂点{v_idx} を含む面が3未満です。")
        link: dict[int, set[int]] = {i: set() for i in incident}
        for f_idx in incident:
            face = faces[f_idx]
            pos = face.index(v_idx)
            prev_v = face[pos - 1]
            next_v = face[(pos + 1) % len(face)]
            for other in incident:
                if other == f_idx:
                    continue
                of = faces[other]
                op = of.index(v_idx)
                if of[op - 1] in (prev_v, next_v) or of[(op + 1) % len(of)] in (prev_v, next_v):
                    link[f_idx].add(other)
        seen = {incident[0]}
        stack = [incident[0]]
        while stack:
            cur = stack.pop()
            for nxt in link[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(seen) != len(incident):
            raise SolidError(
                "non_manifold_vertex", f"頂点{v_idx} まわりの面が連結ではありません（非多様体）。"
            )


def count_spanning_trees(edges: list[tuple[int, int]], num_nodes: int) -> int:
    """隣接グラフの全域木の個数を Matrix-Tree 定理で求める（列挙せずに計算）。"""
    if num_nodes <= 1:
        return 1
    lap = np.zeros((num_nodes - 1, num_nodes - 1))
    for u, v in edges:
        if u < num_nodes - 1:
            lap[u, u] += 1
        if v < num_nodes - 1:
            lap[v, v] += 1
        if u < num_nodes - 1 and v < num_nodes - 1:
            lap[u, v] -= 1
            lap[v, u] -= 1
    return round(float(np.linalg.det(lap)))


def build_edge_map(faces: list[list[int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """無向辺 -> [(その辺を含む面インデックス, 進行方向)] を構築する。

    進行方向は面のループがその辺を (min, max) の順に通れば +1、逆なら -1。
    「2頂点を共有する」のではなく「辺を共有する」で隣接を判定するための土台。
    """
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for f_idx, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            direction = 1 if a < b else -1
            edge_map.setdefault(key, []).append((f_idx, direction))
    return edge_map


def build_adjacency_with_edges(
    faces: list[list[int]],
) -> list[tuple[int, int, tuple[int, int]]]:
    """面の隣接を (面u, 面v, 共有辺) のリストで返す。"""
    result = []
    for edge, holders in build_edge_map(faces).items():
        if len(holders) == 2:
            (u, _), (v, _) = holders
            result.append((u, v, edge))
    return result


def build_adjacency_graph(faces: list[list[int]]) -> list[tuple[int, int]]:
    """面の隣接グラフの辺リスト（(面u, 面v)）を構築する。"""
    return [(u, v) for u, v, _ in build_adjacency_with_edges(faces)]


def _is_connected(cur: list[tuple[int, int]], num_nodes: int) -> bool:
    """辺集合が全 num_nodes 頂点を連結にするかを BFS で調べる。"""
    nbrs: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in cur:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen = {0}
    stack = [0]
    while stack:
        c = stack.pop()
        for w in nbrs[c]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == num_nodes


def enumerate_spanning_trees(
    edges: list[tuple[int, int]], num_nodes: int
) -> Iterator[list[tuple[int, int]]]:
    """すべての全域木を、1つずつ返す（ジェネレータ）。

    辺の部分集合を全部試す方法では C(E, V-1) 回の組み合わせ判定が必要になり、
    面数の多い立体では列挙だけで時間がかかりすぎる。
    ここでは辺の「削除・縮約」による分割統治を使う。

        τ(G) = τ(G - e)  ⊎  { T ∪ {e} : T ∈ τ(G / e) }

    と分解することで、探索する木の個数がちょうど τ(G) 個に収まる。
    出力する辺は入力順（位置）に昇順に並べる。
    """
    if num_nodes <= 0:
        return
    if num_nodes == 1:
        yield []
        return

    # 自己ループは全域木に含まれ得ないので除く。多重辺は別の辺なので残す。
    # (元の辺番号, u, v) の組で持つ。縮約しても元の番号を失わないため。
    base = [(i, u, v) for i, (u, v) in enumerate(edges) if u != v]
    endpoints = {i: (u, v) for i, u, v in base}
    if not _is_connected([(u, v) for _, u, v in base], num_nodes):
        return

    def rec(
        cur: list[tuple[int, int, int]],
        n: int,
        chosen: list[int],
    ) -> Iterator[list[tuple[int, int]]]:
        """cur の全域木を列挙する。chosen はこの木に含めた辺の元の番号。"""
        # 全域木に使え得る辺を1本選ぶ（ループは使わない）
        pivot = -1
        for i, (_, a, b) in enumerate(cur):
            if a != b:
                pivot = i
                break
        if pivot < 0:
            if n == 1:  # 使える辺が尽き、頂点が1つなら全域木が完成している
                # 端点は元の辺定義から取り出すので、縮約の過程に影響されない
                yield [endpoints[i] for i in sorted(chosen)]
            return

        orig, pu, pv = cur[pivot]
        rest = cur[:pivot] + cur[pivot + 1:]

        # (1) pivot を使わない全域木
        if _is_connected([(a, b) for _, a, b in rest], n):
            yield from rec(rest, n, chosen)

        # (2) pivot を使う全域木：pivot の両端を縮約して、縮約後の全域木に対応させる
        # 縮約後の頂点数は n-1 個。n-2 が pu と pv の統合先、残りの番号を詰める。
        merged = n - 2

        # remap をインライン化する（関数呼び出しを避ける）。
        # ⚠️ 整数演算のみで**幾何の丸め演算には触れない**。列挙順序も
        # 変わらないので、出力される辺列は bit 単位で一致する。
        contracted = []
        for i, a, b in rest:
            a2 = merged if (a == pu or a == pv) else a - (pu < a) - (pv < a)
            b2 = merged if (b == pu or b == pv) else b - (pu < b) - (pv < b)
            contracted.append((i, a2, b2))
        yield from rec(contracted, n - 1, chosen + [orig])

    yield from rec(base, num_nodes, [])


def face_normal(vertices: np.ndarray, face: list[int]) -> np.ndarray:
    """1つの面の法線を Newell の方法で計算する。

    先頭3頂点の外積ではなく面全体のループを使うため、
    凹角 N-gon や先頭3頂点が一直線状の面でも正しく計算できる。
    """
    acc = np.zeros(3)
    m = len(face)
    for i in range(m):
        a = vertices[face[i]]
        b = vertices[face[(i + 1) % m]]
        acc[0] += (a[1] - b[1]) * (a[2] + b[2])
        acc[1] += (a[2] - b[2]) * (a[0] + b[0])
        acc[2] += (a[0] - b[0]) * (a[1] + b[1])
    norm = np.linalg.norm(acc)
    # acc は面積（長さ^2）の単位なので、許容値にも座標スケール^2 を掛ける。
    # 絶対値の閾値を置くと辺長 1e-7 以下の正当な立体を「退化した」と誤判定する。
    # max(1.0, s) の下限は置かない。小さいスケールを 1.0 の床に貼り付けてしまう。
    s = float(np.max(np.abs(vertices)))
    if norm < FACE_NORMAL_REL_TOL * s * s:
        raise SolidError("degenerate_face", "退化した面（面積ゼロ）が検出されました。")
    return acc / norm


def compute_normals(vertices: np.ndarray, faces: list[list[int]]) -> list[np.ndarray]:
    """各面の法線ベクトルを計算する。

    面の向きが入力に依存しないよう、build_adjacency_graph と組み合わせて
    巻き順を統一してから外向き法線を返す。
    """
    return [face_normal(vertices, f) for f in orient_faces_outward(vertices, faces)]


def signed_volume(vertices: np.ndarray, faces: list[list[int]]) -> float:
    """発散定理で符号付き体積を求める。巻き順が正しければ正の値になる。"""
    total = 0.0
    for face in faces:
        m = len(face)
        for i in range(1, m - 1):
            a, b, c = vertices[face[0]], vertices[face[i]], vertices[face[i + 1]]
            total += float(np.dot(a, np.cross(b, c)))
    return total / 6.0


def orient_faces_outward(
    vertices: np.ndarray, faces: list[list[int]]
) -> list[list[int]]:
    """面の巻き順を「外向き・互いに整合」に正規化して返す。

    呼び出し側が巻き順（Hole の向き）を意識しなくて済むようにする。

    **前条件**: 呼び出し側は先に :func:`validate_solid` を通していること。
    各辺がちょうど2面に共有される保証はそこに閉じている。
    これを破った入力（境界あり・非多様体）は下のタプル展開が
    ``ValueError: too many values to unpack`` になる。黙って無視しない。
    非可向きの立体は従来どおり ``SolidError("non_orientable")``。
    """
    oriented = [list(f) for f in faces]
    edge_map = build_edge_map(faces)

    # 1) 各辺を2つの面が逆向きに通るようにする（整合性の強制）
    #    parity union-find で「どちらかの面を裏返す必要がある」制約を解く
    parent: dict[int, int] = {i: i for i in range(len(faces))}
    parity: dict[int, int] = {i: 0 for i in range(len(faces))}

    def find(x: int) -> tuple[int, int]:
        if parent[x] != x:
            p = parent[x]
            root, q = find(p)
            parent[x] = root
            parity[x] ^= q
        return parent[x], parity[x]

    for holders in edge_map.values():
        (f1, d1), (f2, d2) = holders
        # 両面が同じ向きに通っているなら、ちょうど片方を裏返す必要がある
        need_flip = 1 if d1 == d2 else 0
        r1, p1 = find(f1)
        r2, p2 = find(f2)
        if r1 == r2:
            if (p1 ^ p2) != need_flip:
                raise SolidError("non_orientable", "非可向きの立体です（面の整合性を解消できません）。")
        else:
            parent[r1] = r2
            parity[r1] = p1 ^ p2 ^ need_flip

    for i in range(len(faces)):
        find(i)  # 経路圧縮して parity を根に対して確定させる
    for i in range(len(faces)):
        if parity[i]:
            oriented[i] = list(reversed(faces[i]))

    # 2) 全体を裏返して外向きにする
    if signed_volume(vertices, oriented) < 0:
        oriented = [list(reversed(f)) for f in oriented]

    return oriented


def build_shared_of(faces: list[list[int]]) -> dict[tuple[int, int], tuple[int, int]]:
    """(面u, 面v) -> 共有辺 の両方向マップを作る。

    隣接する 2 面が共有する辺はちょうど 1 本なので、(u, v) と (v, u) は
    同じ辺を指す。親子関係も評価指標もこれで「どの面とどの面が折線で
    つながっているか」を引く。
    """
    shared_of: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v, edge in build_adjacency_with_edges(faces):
        shared_of[(u, v)] = edge
        shared_of[(v, u)] = edge
    return shared_of


def _build_fold_tables(
    vertices: np.ndarray, faces: list[list[int]], normals: list[np.ndarray]
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """隣接面対ごとに折り曲げの回転行列と並進ベクトルを前計算する。

    ``layout_tree`` の BFS はこの表を引くだけになる。``r`` の値は木に依存しない
    ので、列挙する全域木が何本でも計算は1回で済む。

    ``r`` は共有辺の軸 ``k`` まわりで面 v を面 u と同じ平面まで倒す回転
    （Rodrigues）。``k`` は共有辺の端点順で決まるが、``build_adjacency_with_edges``
    が返す辺は常に ``(小, 大)`` に正規化されているので、``(u, v)`` と ``(v, u)``
    で ``k`` の向きは共通。法線を入れ替えると
    ``s = dot(cross(k, n_v), n_u)`` の符号だけが反転し、回転角の符号が反転する。
    つまり ``r(v, u) == r(u, v).T``（逆回転）である。

    ``w = p_a - r @ p_a`` は「回転の中心を共有辺へ移す」並進で、
    ``p_a``（辺の端点の一方）**または** ``p_b`` のどちらを使っても数学的には
    同じ結果になる（``p_a - p_b`` は回転軸 ``k`` に平行で、``(I - r)(p_a - p_b) = 0``
    だから）。ただし浮動小数では一致しないので、
    ``(u, v)``・``(v, u)`` の**両方で ``p_a``（= edge[0]）を使う**。
    こうすると前計算前の実装と**ビット単位で同じ** ``off`` を再現でき、
    同値の指標を持つ展開図の並び順まで含めて応答が完全に一致する
    （prism4 の golden がこの性質に依存していた。``p_b`` を使うと 1 ULP
    ずれて安定ソートの tie が反転し、golden が落ちる）。
    """
    fold_of: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for u, v, edge in build_adjacency_with_edges(faces):
        p_a = vertices[edge[0]]
        p_b = vertices[edge[1]]
        k = (p_b - p_a) / np.linalg.norm(p_b - p_a)
        kx, ky, kz = k.tolist()
        nx_, ny_, nz_ = normals[u].tolist()
        mx, my, mz = normals[v].tolist()

        c = mx * nx_ + my * ny_ + mz * nz_
        # s = dot(cross(k, n_v), n_u) （np.cross は微小配列では極端に遅い）
        s = (
            (ky * mz - kz * my) * nx_
            + (kz * mx - kx * mz) * ny_
            + (kx * my - ky * mx) * nz_
        )
        v1c = 1.0 - c
        r = np.array(
            [
                [kx * kx * v1c + c, kx * ky * v1c - kz * s, kx * kz * v1c + ky * s],
                [ky * kx * v1c + kz * s, ky * ky * v1c + c, ky * kz * v1c - kx * s],
                [kz * kx * v1c - ky * s, kz * ky * v1c + kx * s, kz * kz * v1c + c],
            ]
        )
        fold_of[(u, v)] = (r, p_a - r @ p_a)
        # (v, u) は逆回転 r.T。並進も p_a 基準（上のコメント参照）。
        rt = r.T
        fold_of[(v, u)] = (rt, p_a - rt @ p_a)
    return fold_of


def prepare_layout(
    vertices: np.ndarray, faces: list[list[int]], normals: list[np.ndarray]
) -> Layout:
    """layout_tree の呼び出しごとに変わらない前計算をまとめる。

    全域木1つにつき1回しか使わない値だが、候補が数千本あるため先にまとめておく。
    """
    origin = vertices[faces[0][0]]
    u_x = vertices[faces[0][1]] - origin
    u_x = u_x / np.linalg.norm(u_x)
    nx_, ny_, nz_ = normals[0].tolist()
    ux, uy, uz = u_x.tolist()
    u_y = np.array([ny_ * uz - nz_ * uy, nz_ * ux - nx_ * uz, nx_ * uy - ny_ * ux])

    return {
        "shared_of": build_shared_of(faces),
        "origin": origin,
        "u_x": u_x,
        "u_y": u_y,
        # 面の3D座標も変わらないので、毎回の多少次索引を避ける
        "face_pts": [vertices[f] for f in faces],
        # LinearRing を一括生成するときの「どの座標がどの面か」の対応。
        # 面の頂点数だけから決まり木に依存しないので、ここで1回だけ作る。
        "flat_indices": np.repeat(
            np.arange(len(faces)), [len(f) for f in faces]
        ),
        # 全 face の 3D 座標を1本に連結したもの（投影のバッチ計算用）。
        "flat_pts": np.concatenate([vertices[f] for f in faces]),
        # 全面の頂点数が同じなら、まとめて Polygon を作れる
        # (面u, 面v) -> (折り曲げ回転行列 r, w = p_a - r @ p_a)。
        # r は共有辺と両面の法線だけで決まるので木に依存しない。
        "fold_of": _build_fold_tables(vertices, faces, normals),
    }


def _tree_bfs(
    n_faces: int, tree_edges: list[tuple[int, int]]
) -> tuple[list[int | None], list[tuple[int, int]]]:
    """全域木の親子関係と、親が子より先に並ぶ ``(親, 子)`` の列を返す。

    **これが唯一の走査定義**。``layout_tree`` の回転の伝播も
    ``layout_parent_relation`` の親子関係も、ここから作られる。

    契約: 根は面0。親は木の無向隣接を幅優先に走査し、
    最初に訪れた未訪問の隣接面を親として割り当てる。根の親は None。

    ``(親, 子)`` の列も返すのは、呼び出し側で ``parents[v]`` を引くと
    ``int | None`` がリストへの添字として型エラーになるため。
    走査順そのものを渡す。
    """
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    parents: list[int | None] = [None] * n_faces
    pairs: list[tuple[int, int]] = []
    visited = {0}
    queue = [0]
    for u in queue:
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parents[v] = u
                pairs.append((u, v))
                queue.append(v)
    return parents, pairs

def layout_parent_relation(
    n_faces: int, tree_edges: list[tuple[int, int]]
) -> list[int | None]:
    """全域木から面の親子関係を求める（面0が根、ルートは None）。

    契約は :func:`_tree_bfs` に書いてある。折り曲げ軸は「親と共有する辺」なので、
    ここが変わると API 応答の ``fold_axis_*`` も変わる。契約として固定すること。
    """
    return _tree_bfs(n_faces, tree_edges)[0]


def layout_tree(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    tree_edges: list[tuple[int, int]],
    precomputed: Layout | None = None,
) -> list[Polygon]:
    """全域木に基づいて面を展開し、ShapelyのPolygonリスト（2D座標系）を返す。

    平行移動・回転は 4x4 行列ではなく (回転行列 R, 移動ベクトル t) で持ち、
    最終的な 2D への投影もまとめてベクトル演算する。
    全域木の数だけ呼び出されるため、ここは定数倍でも効く。

    折り曲げの回転行列 ``r`` は**木に依存しない**（共有辺と両面の法線だけで決まる）ので、
    :func:`_build_fold_tables` で隣接面対ごとに1回だけ計算し、ここでは引くだけにする。
    ループに残るのは行列積2回だけになる。
    """
    if precomputed is None:
        precomputed = prepare_layout(vertices, faces, normals)
    fold_of = precomputed["fold_of"]
    n_faces = len(faces)

    _parents, pairs = _tree_bfs(n_faces, tree_edges)

    # 親が子より先に現れる幅優先順で、回転と平行移動を伝播させる。
    # placed は到達した面だけ True にする。
    #
    # rot / off を最初から numpy 配列に貯める（list -> array 変換を無くすため）。
    # 値そのものは list 版と同一で、丸め演算も変わらない。
    rot = np.empty((n_faces, 3, 3))
    off = np.empty((n_faces, 3))
    rot[0] = np.eye(3)
    off[0] = 0.0
    placed = [False] * n_faces
    placed[0] = True

    for u, v in pairs:
        # tree_edges は build_adjacency_graph 由来の全域木なので共有辺は必ず存在する。
        # そうでない呼び出し（利用者が前条件を破った場合）はここで KeyError になる。
        #
        # r（折り曲げ回転）と w = p_a - r @ p_a は前計算済み。
        # ループに残るのは行列積2回だけ。
        r, w = fold_of[(u, v)]

        rot[v] = rot[u] @ r
        off[v] = rot[u] @ w + off[u]
        placed[v] = True

    origin = precomputed["origin"]
    u_x = precomputed["u_x"]
    u_y = precomputed["u_y"]

    if not all(placed):
        missing = next(i for i in range(n_faces) if not placed[i])
        raise ValueError(f"面{missing} へ到達できませんでした（全域木が不正です）。")

    # 軸 ``u @ rot[i]`` は einsum で一括計算しても**ビット単位で同じ**
    # （``tools/check_projection_axes_bits.py``: 2万回で不一致 0）。
    # 平行移動の定数項 ``(off - origin) @ u`` も成分和にすれば同じ。
    axes_x = np.einsum("b,fba->fa", u_x, rot)
    axes_y = np.einsum("b,fba->fa", u_y, rot)
    shift = off - origin
    cs_x = (shift * u_x).sum(axis=1)
    cs_y = (shift * u_y).sum(axis=1)

    # 全 face の頂点を1本の (M,3) に積み、面ごとの軸を「行ごとに繰り返した」
    # 配列 (M,3) と成分ごとに掛けて足す。face 数ぶんの GEMV 呼び出しが
    # 「数回のベクトル演算」に置き換わる（実測 projection が約2倍）。
    #
    # ⚠️ これは GEMV (`pts @ axis`) とは**丸めが一致しない**
    # （``tools/check_projection_batch_bits.py`` / ``..._explicit_bits.py``）。
    # 座標が 1 ULP 変わっても重なり判定はしきい値（相対1e-7）で堅牢、
    # 面積指標の 9 桁丸めも変わらないが、**同値タイの並び順は変わり得る**。
    # API golden（同値タイを含む）で観測差がないことを確認して採る。
    idx_rows = precomputed["flat_indices"]
    P = precomputed["flat_pts"]
    AX = axes_x[idx_rows]
    AY = axes_y[idx_rows]
    xs = P[:, 0] * AX[:, 0] + P[:, 1] * AX[:, 1] + P[:, 2] * AX[:, 2] + cs_x[idx_rows]
    ys = P[:, 0] * AY[:, 0] + P[:, 1] * AY[:, 1] + P[:, 2] * AY[:, 2] + cs_y[idx_rows]

    flat = np.column_stack([xs, ys])

    # 面を shapely の Polygon に変換する。
    #
    # 頂点数が揃っていなくても ``indices`` でまとめて作れる（shapely 2.x）。
    # Polygon() を1枚ずつ呼ぶと、頂点数が混在する立体（角錐台の底面など、
    # 8角形と四角形が混ざる）では Python 呼び出しが面数だけかかる。
    # ``indices`` は面の頂点数だけで決まるため prepare_layout で前計算済み。
    indices = precomputed["flat_indices"]
    built = cast(
        list[Polygon],
        shapely.polygons(shapely.linearrings(flat, indices=indices)),
    )
    return list(built)


def is_valid_layout(
    polygons: list[Polygon],
    folded_pairs: set[tuple[int, int]] | None = None,
) -> bool:
    """展開図のポリゴン間に重なり（衝突）がないかを検証する。

    外面の数が少ないので、空間索引（STRtree）よりも外接矩形で絞り込んだほうが
    intersection を呼ぶ回数が減って速い。

    ``bounds`` を ``p.bounds`` の Python ループではなく shapely のベクトル演算で
    一括取得する（実測: stairs で 3.4s）。判定は**逐次のまま**なので、
    重なりを見つけた時点で ``return False`` する早期終了が効く
    （stairs では 79% のネットが途中で落ちる。ベクトル化して
    全ペア計算するとこの早期終了を失って**遅くなる**）。

    ``folded_pairs`` は「展開図で折り線を共有する面対 ``(i, j)``（``i < j``）」の
    集合。折り線を共有する面は**その線の反対側に配置される**ので、内部が
    重なることはなく交差面積は厳密に 0（浮動ノイズのみ）。よって交差計算を
    丸ごと省略できる。実測では bbox を通過するペアの **50〜72%** がこれで、
    intersection 呼び出しを半減できる。

    ⚠️ 省略してよい根拠は「折り曲げの構成」であって「たまたま重ならない」では
    ない。tree に無い面対（ポリゴン上で辺を接しているだけの非折り線ペア）は
    除外対象に**含めてはならない**。
    """
    n_polys = len(polygons)
    if n_polys < 2:
        return True
    folds = folded_pairs if folded_pairs is not None else frozenset()
    # 外接矩形は Python float に落としてから使う。numpy のスカラは演算ごとに
    # ディスパッチ費用が乗るため、ペア数（面数の2乗）だけ回るこのループでは
    # float のほうがはっきり速い。値はビット単位で同じ（float64 のまま）。
    bnds = shapely.bounds(np.asarray(polygons, dtype=object)).tolist()
    for i in range(n_polys):
        ax0, ay0, ax1, ay1 = bnds[i]
        for j in range(i + 1, n_polys):
            if (i, j) in folds:
                continue
            bx0, by0, bx1, by1 = bnds[j]
            # 外接矩形が重ならなければ交差面積も 0 なので交差計算を省略できる
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            # 閾値を面積の絶対値にすると、立体が小さいと実重なりを見落とす。
            # 2つの面の和集合の外接矩形の対角の長さ^2 を基準にした比で判定する。
            # 和集合を使うのは「接しているだけ」のとき交差側の幅が0になり、
            # 閾値も0になって浮動ノイズを実重なりと誤判定するため。
            dx = max(ax1, bx1) - min(ax0, bx0)
            dy = max(ay1, by1) - min(ay0, by0)
            thresh = OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
            # 交差面積は「外接矩形の重なり面積」を超えない。重なり面積がしきい値
            # 以下なら交差面積もしきい値以下と確定するので、交差計算を省略できる。
            # ここに来る時点で bbox は交差しているので ox, oy は非負。
            # 辺や点で**接するだけ**のペア（ox か oy が 0）はここで丸ごと落ちる。
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox * oy <= thresh:
                continue
            if polygons[i].intersection(polygons[j]).area > thresh:
                return False
    return True


class NetMetrics(NamedTuple):
    """展開図 1 枚分の評価指標。"""

    bounding_box_width: float
    bounding_box_height: float
    bounding_box_area: float
    perimeter: float
    fold_count: int


def _fold_count(
    polys: list[Polygon],
    tree_edges: list[tuple[int, int]],
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    shared_of: dict[tuple[int, int], tuple[int, int]],
) -> int:
    """折る回数 = 同じ直線上で同じ折り方の折り曲げ軸をまとめた本数。

    同じ直線上に並んでいても、山折りと谷折り（折り角の符号）や折り角の
    大きさが違えば 1 回では折れないので、別の折り操作として数える
    （仕様 3.5）。

    各ポリゴンの頂点は ``faces[i]`` と同じ順に並んでいるので、共有辺の
    3D 頂点インデックスからそのまま 2D 座標を引ける。
    """
    # 直線の係数を展開図の大きさで正規化してから丸める。こうしないと
    # 立体を拡大・縮小したときに値が変わる（_fold_axis_2d と同じ方針）。
    extent = max(
        max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in polys
    )
    extent = extent or 1.0

    pos_in_face = [{vertex: k for k, vertex in enumerate(face)} for face in faces]
    coords = [list(poly.exterior.coords) for poly in polys]

    folds: set[tuple[float, float, float, float]] = set()
    for u, v in tree_edges:
        a, b = shared_of[(u, v)]
        index = pos_in_face[u]
        p0 = coords[u][index[a]]
        p1 = coords[u][index[b]]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy)
        # 法線 (nx, ny) と原点からの符号付き距離 offset で直線を表す。
        nx, ny = -dy / norm, dx / norm
        offset = nx * p0[0] + ny * p0[1]
        # 同じ直線が向きの違いで別物にならないよう、法線の向きを揃える。
        if abs(nx) > 1e-9:
            if nx < 0.0:
                nx, ny, offset = -nx, -ny, -offset
        elif ny < 0.0:
            nx, ny, offset = -nx, -ny, -offset
        # 折り角 = 180 − 二面角。正なら谷折り、負なら山折り。
        # 角度は 1e-6 度までで丸める（9 桁だと計算経路の差が残り、
        # 同じ折り方の軸が別扱いになりうる）。
        fold_angle = 180.0 - _dihedral_deg(vertices, faces, normals, u, v)
        folds.add(
            (
                round(nx, EDGE_LEN_DECIMALS),
                round(ny, EDGE_LEN_DECIMALS),
                round(offset / extent, EDGE_LEN_DECIMALS),
                round(fold_angle, 6),
            )
        )
    return len(folds)


def net_metrics(
    polys: list[Polygon],
    tree_edges: list[tuple[int, int]],
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    shared_of: dict[tuple[int, int], tuple[int, int]],
) -> NetMetrics:
    """展開図の評価指標を求める。

    **外周長 = 全面の周長合計 − 2 × 折り曲げ軸の長さ合計**。

    展開図の外周は「折ったときに消える内側の棱」だけ短い。
    したがって外接矩形の周長 ``2*(w+h)`` は外周長ではない
    （L 字型の展開図では外接矩形の空いた部分だけ周を含む）。

    **折る回数 = 同じ直線上で同じ折り方の折り曲げ軸をまとめた本数**（仕様 3.5）。
    山折りと谷折り、折り角の大きさが違う軸は別の操作として数える。

    ここが指標の**唯一の定義**。API と CLI はどちらもこれを呼ぶ。
    """
    # 指標は軽いので全部つないでもきる。重い JSON 化は上位 n 個だけで行う。
    # 全頂点を Python で回ると一意な展開図の数だけ遅くなるので、
    # 外接矩形（bounds）だけから求める。
    boxes = [p.bounds for p in polys]
    bbox_w = max(b[2] for b in boxes) - min(b[0] for b in boxes)
    bbox_h = max(b[3] for b in boxes) - min(b[1] for b in boxes)

    glued = 0.0
    for u, v in tree_edges:
        # tree_edges は build_adjacency_graph 由来なので共有辺は必ず存在する。
        a, b = shared_of[(u, v)]
        glued += float(np.linalg.norm(vertices[a] - vertices[b]))

    return NetMetrics(
        bounding_box_width=float(bbox_w),
        bounding_box_height=float(bbox_h),
        bounding_box_area=float(bbox_w * bbox_h),
        perimeter=float(sum(p.length for p in polys) - 2.0 * glued),
        fold_count=_fold_count(
            polys, tree_edges, vertices, faces, normals, shared_of
        ),
    )


def metric_sort_key(metrics: NetMetrics, sort_by: str) -> float:
    """``sort_by`` に対応する並べ替えキー。昇順が「良い順」。

    どの指標で並べるかは API と CLI で同じでなければならないので、
    ここに一本化する。
    """
    if sort_by == "area":
        return metrics.bounding_box_area
    if sort_by == "perimeter":
        return metrics.perimeter
    if sort_by == "folds":
        return float(metrics.fold_count)
    raise ValueError(f"未知の sort_by: {sort_by}")


def _face_area_3d(vertices: np.ndarray, face: list[int]) -> float:
    """面の面積を Newell の式で求める（3D 上の値）。"""
    acc = np.zeros(3)
    m = len(face)
    for i in range(m):
        acc += np.cross(vertices[face[i]], vertices[face[(i + 1) % m]])
    return float(np.linalg.norm(acc)) / 2.0


def _dihedral_deg(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    u: int,
    v: int,
) -> float:
    """棱 u-v を挟む、立体の内側の二面角（度）を求める。

    単純な 180 - 角度(法線u, 法線v) では、L字柱のような非凸な立体の
    凹角（270 度）を 90 度として扱ってしまう。
    そこで棱に垂直な断面で見たときの「立体の内側の張る角」を求める:

      * 外向き法線 n と、面を一周する向きの棱方向 t の外積が
        その面の内側（立体側）を指す。
      * 2つの面の内側方向のなす角を phi（0..180）とする。
      * その2方向の bisector が立体の内側を向いていれば張る角は phi、
        外側（凹角）を向いていれば 360 - phi。
    """
    n_u = len(faces[u])
    shared = set(faces[u]) & set(faces[v])
    # faces[u] の周回順に沿った棱の向きを求める
    i = next(
        k for k in range(n_u) if {faces[u][k], faces[u][(k + 1) % n_u]} == shared
    )
    j = (i + 1) % n_u
    t = vertices[faces[u][j]] - vertices[faces[u][i]]
    # 長さ0の辺は validate_solid の zero_length_edge が先に弾くので、ここは 0 にならない。
    t = t / np.linalg.norm(t)

    in_u = np.cross(normals[u], t)   # 面 u の内側（立体側）を指す
    in_v = -np.cross(normals[v], t)  # 面 v の内側。面 v では棱を逆向きに歩く
    cos_phi = max(-1.0, min(1.0, float(np.dot(in_u, in_v))))
    phi = math.degrees(math.acos(cos_phi))

    bisector = in_u + in_v
    inside = float(np.dot(bisector, normals[u])) < 0.0 and (
        float(np.dot(bisector, normals[v])) < 0.0
    )
    return phi if inside else 360.0 - phi


def _pair_to_edge(faces: list[list[int]]) -> dict[frozenset[int], tuple[int, int]]:
    """{(面u, 面v): 共有辺} のマップを作る。"""
    return {
        frozenset((u, v)): edge for u, v, edge in build_adjacency_with_edges(faces)
    }


def _fold_axis_2d(
    polys: list[Polygon], face_id: int, parent_id: int
) -> list[list[float]] | None:
    """展開図平面上での折り曲げ軸（2D）を、親面と共有する2頂点から求める。

    共有頂点は「同じ座標を持つ頂点」で照合する。面は独立に計算されるので
    共有頂点の座標は下位ビットだけずれるので、丸めなしでは一致しない。

    ⚠️ **正規化のあと**に丸める。先に丸めると、展開図が小さいとき
    異なる頂点が同じキーに潰れて軸が見つからなくなる（実測: scale 1e-9 で
    box(1,1.5,2) の軸が14本消えた）。
    """
    ext = max(
        max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in polys
    )
    ext = ext or 1.0

    def key(pt: tuple[float, ...]) -> tuple[float, float]:
        return (
            round(pt[0] / ext, EDGE_LEN_DECIMALS),
            round(pt[1] / ext, EDGE_LEN_DECIMALS),
        )

    parent_pts = {key(c) for c in polys[parent_id].exterior.coords}
    shared = [c for c in polys[face_id].exterior.coords if key(c) in parent_pts]
    if len(shared) < 2:
        return None
    return [
        [float(shared[0][0]), float(shared[0][1])],
        [float(shared[1][0]), float(shared[1][1])],
    ]


class FaceGeometry(NamedTuple):
    """1 面分の幾何情報。

    JSON のキー名は api_server 側で付ける。純粋な幾何だけをここに置き、
    「どう JSON に書くか」は持ち込まない。
    """

    face_id: int
    parent_id: int | None
    coords: list[list[float]]
    fold_axis_3d: list[list[float]] | None
    fold_axis_2d: list[list[float]] | None
    dihedral_angle: float | None
    fold_angle: float | None


def face_geometries(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    tree_edges: list[tuple[int, int]],
    polys: list[Polygon],
    pair_to_edge: dict[frozenset[int], tuple[int, int]],
) -> list[FaceGeometry]:
    """各面の2D形状・親子関係・折り曲げ軸/角を求める。

    ``tree_edges`` は :func:`layout_parent_relation` と同じ定義で親子を決める。
    """
    parents = layout_parent_relation(len(faces), tree_edges)

    out: list[FaceGeometry] = []
    for face_id, poly in enumerate(polys):
        x, y = poly.exterior.xy
        # shapely は末尾に開始点を重複して持つので取り除く
        coords = [[float(x[i]), float(y[i])] for i in range(len(x) - 1)]

        parent_id = parents[face_id]
        axis_3d = axis_2d = None
        dihedral = fold_angle = None

        if parent_id is not None:
            # 親子関係は tree_edges（= 面グラフの全域木）から来るので、
            # 親とこの面は必ず隣接し、共有辺が必ず存在する。
            a, b = pair_to_edge[frozenset((face_id, parent_id))]
            axis_3d = [vertices[a].tolist(), vertices[b].tolist()]
            axis_2d = _fold_axis_2d(polys, face_id, parent_id)
            dihedral = _dihedral_deg(vertices, faces, normals, face_id, parent_id)
            # 平らな展開図を立体に畳み込むのに必要な回転角
            fold_angle = 180.0 - dihedral

        out.append(
            FaceGeometry(
                face_id=face_id,
                parent_id=parent_id,
                coords=coords,
                fold_axis_3d=axis_3d,
                fold_axis_2d=axis_2d,
                dihedral_angle=dihedral,
                fold_angle=fold_angle,
            )
        )
    return out


class RibbonContext:
    """半辺構造の前計算結果。全域木ごとに変わらないので一度だけ作る。

    Python リスト側のフィールド（nxt / edge_len / node_id / idx_map / shared）は
    :func:`_glue_arrays` / :func:`_wl_refine` / :func:`_individualize_refine`
    （木ごとの正規形計算）が使う。バッチ化された正規形計算
    :func:`_batch_canonical_keys` は numpy フィールド（np_nxt / np_edge_len /
    adj_* / adj_pos_of）を使う。どちらも同じ半辺グラフを指すので、
    値の定義は1つに保つ（np_nxt = np.asarray(nxt) など）。
    """

    __slots__ = (
        "adj_fl",
        "adj_nu",
        "adj_nv",
        "adj_pos_matrix",
        "adj_pos_of",
        "edge_len",
        "faces",
        "idx_map",
        "n_faces",
        "n_nodes",
        "node_id",
        "np_edge_len",
        "np_nxt",
        "nxt",
        "shared",
        "vertices",
    )

    def __init__(self, vertices: np.ndarray, faces: list[list[int]]):
        self.vertices = vertices
        self.faces = faces
        self.n_faces = len(faces)

        nodes: list[tuple[int, int]] = []
        nxt: list[int] = []
        raw_len: list[float] = []
        node_id: list[list[int]] = []
        idx_map: list[dict[frozenset[int], int]] = []
        for f_idx, face in enumerate(faces):
            k = len(face)
            base = len(nodes)
            m: dict[frozenset[int], int] = {}
            row: list[int] = []
            for i in range(k):
                m.setdefault(frozenset((face[i], face[(i + 1) % k])), i)
                row.append(base + i)
                nodes.append((f_idx, i))
                raw_len.append(
                    float(
                        np.linalg.norm(vertices[face[i]] - vertices[face[(i + 1) % k]])
                    )
                )
            for i in range(k):
                nxt.append(row[(i + 1) % k])
            node_id.append(row)
            idx_map.append(m)

        # 辺長は「最大の辺長を1とする比」で持つ。座標のスケールに依存させては
        # いけないので、丸める前の値で最大を取り、正規化してから丸める。
        # 順序を逆にすると 1e-5 程度の立体では全辺が同じ値に潰れて情報が消える。
        mx = max(raw_len)
        edge_len = [round(e / mx, EDGE_LEN_DECIMALS) for e in raw_len]

        self.n_nodes = len(nodes)
        self.nxt = nxt
        self.edge_len = edge_len
        self.idx_map = idx_map
        self.node_id = node_id

        # --- バッチ化用の numpy 表現 ---------------------------------------
        self.np_nxt = np.asarray(nxt, dtype=np.int64)
        self.np_edge_len = np.asarray(edge_len, dtype=float)

        # 接着表（隣接エッジ位置 -> 半辺番号と向き）。enumerate_spanning_trees
        # が返す木の辺は (u, v) のタプルで、位置 p = build_adjacency_with_edges
        # の列挙順に一致する（両者とも build_edge_map の列挙順を辿る）。
        with_edges = build_adjacency_with_edges(faces)
        adj_nu = np.empty(len(with_edges), dtype=np.int64)
        adj_nv = np.empty(len(with_edges), dtype=np.int64)
        adj_fl = np.empty(len(with_edges), dtype=np.int64)
        adj_pos_of: dict[tuple[int, int], int] = {}
        for p, (u, v, edge) in enumerate(with_edges):
            adj_pos_of[(u, v)] = p
            iu = idx_map[u][frozenset(edge)]
            iv = idx_map[v][frozenset(edge)]
            adj_nu[p] = node_id[u][iu]
            adj_nv[p] = node_id[v][iv]
            fu, fv = faces[u], faces[v]
            ku, kv = len(fu), len(fv)
            adj_fl[p] = 1 if (
                fu[iu] == fv[(iv + 1) % kv] and fu[(iu + 1) % ku] == fv[iv]
            ) else 0
        self.adj_pos_of = adj_pos_of
        self.adj_nu = adj_nu
        self.adj_nv = adj_nv
        self.adj_fl = adj_fl
        # (面u, 面v) -> 位置 p の行列版。adj_pos_of の dict 引きを
        # 列挙木のバッチで numpy のファンシー索引に置き換えるために持つ。
        # 隣接しない面対は -1（参照されない）。
        adj_pos_matrix = np.full((self.n_faces, self.n_faces), -1, dtype=np.int64)
        for (u, v), p in adj_pos_of.items():
            adj_pos_matrix[u, v] = p
        self.adj_pos_matrix = adj_pos_matrix

        shared: dict[tuple[int, int], frozenset[int]] = {}
        for u, v, edge in build_adjacency_with_edges(faces):
            key = frozenset(edge)
            shared[(u, v)] = key
            shared[(v, u)] = key
        self.shared = shared


def _glue_arrays(
    ctx: RibbonContext, tree_edges: list[tuple[int, int]]
) -> tuple[list[int], list[bool]]:
    """半辺の「接着相手」と「接着の向き」を求める。

    接着は双方向なので無向辺になっている。ここでは向きを区別せず
    「接着相手」として扱う。
    """
    faces = ctx.faces
    n_nodes = ctx.n_nodes
    partner = [-1] * n_nodes
    flip = [False] * n_nodes

    for u, v in tree_edges:
        # tree_edges は面グラフの全域木なので、(u, v) は必ず隣接する。
        edge = ctx.shared[(u, v)]
        iu = ctx.idx_map[u][edge]
        iv = ctx.idx_map[v][edge]
        nu = ctx.node_id[u][iu]
        nv = ctx.node_id[v][iv]
        fu, fv = faces[u], faces[v]
        ku, kv = len(fu), len(fv)
        fl = fu[iu] == fv[(iv + 1) % kv] and fu[(iu + 1) % ku] == fv[iv]
        partner[nu] = nv
        partner[nv] = nu
        flip[nu] = fl
        flip[nv] = fl

    return partner, flip


def _wl_refine(
    nxt: list[int], partner: list[int], flip: list[bool], color: list[Color]
) -> list[Color]:
    """色細分（1-WL）。分割が安定するまで繰り返す。

    細分化しか起きないため「色数が変わらなくなった」時点で安定している。
    """
    n = len(nxt)
    while True:
        new = [
            hash(
                (
                    color[i],
                    color[nxt[i]],
                    None if partner[i] < 0 else (flip[i], color[partner[i]]),
                )
            )
            for i in range(n)
        ]
        if len(set(new)) == len(set(color)):
            return new
        color = new


def _individualize_refine(
    nxt: list[int],
    partner: list[int],
    flip: list[bool],
    color: list[Color],
    budget: list[int],
) -> tuple[Color, ...]:
    """個別化-細分による正規形。

    予算超過時は ``None`` を返さず ``SolidError`` を投げる。
    返すと「その木は重複 Didn't と判断された」ように見えてしまい、
    結果として**静かに誤った一意数**を返すことになる。
    """
    n = len(nxt)
    if budget[0] > SEARCH_BUDGET:
        raise SolidError(
            "search_budget_exceeded",
            "展開図の同型判定が計算量上限を超えました。"
            "面の数を減らしてください。",
        )
    budget[0] += n

    color = _wl_refine(nxt, partner, flip, color)
    if len(set(color)) == n:
        # 全ての半辺が別々の色になった = 色列そのものが完全な同型判定になる
        return tuple(sorted(color))

    cells: dict[Color, list[int]] = {}
    for i, c in enumerate(color):
        cells.setdefault(c, []).append(i)
    multi = {c: members for c, members in cells.items() if len(members) > 1}
    if not multi:
        return tuple(sorted(color))

    # 色の値が最小のセルを1つ選んで個別化する（この選択は同型で変わらない）。
    # members は必ず2要素以上（multi に入るのは len>1 のセルだけ）なので、
    # 先頭の要素で初期化してから残りを試せる。Optional を持ち回る必要はない。
    members = multi[min(multi)]

    def individualise(v: int) -> list[Color]:
        branch = list(color)
        branch[v] = min(color) - 1  # 既存の色と衝突しない新しい色
        return branch

    best = _individualize_refine(
        nxt, partner, flip, individualise(members[0]), budget
    )
    for v in members[1:]:
        got = _individualize_refine(
            nxt, partner, flip, individualise(v), budget
        )
        if got < best:
            best = got
    return best


def _net_signature(
    ctx: RibbonContext, tree_edges: list[tuple[int, int]]
) -> tuple[Color, ...]:
    """展開図の正規形（同型なら必ず一致するキー）。

    個別化-細分が全ての半辺を区別できた場合、その色は完全な同型判定になる。
    予算超過は :func:`_individualize_refine` が ``SolidError`` で報告する。

    これは**木ごと**の正規形計算。``iter_unique_nets`` は窓単位で
    バッチ化した :func:`_batch_canonical_keys` を使う。
    """
    partner, flip = _glue_arrays(ctx, tree_edges)
    color = [(ctx.edge_len[i], partner[i] >= 0, flip[i]) for i in range(ctx.n_nodes)]
    return _individualize_refine(ctx.nxt, partner, flip, color, [0])


# ---------------------------------------------------------------------------
# バッチ化された正規形計算
#
# ボトルネック実測 (prism7, 35,287 木): 全時間の約7割が _wl_refine の
# 「木ごと・半辺ごと」の Python ループで、そこでの hash() 呼び出しは 2,600 万回。
# 安定化WLは木ごとに独立な計算なので、窓（既定 2048 木）単位で同時に回せる。
#
# 正しさの契約:
#   * 各行（木）は**初期状態から固定の動学分かち合い**を辿り、
#     「色数が初めて増えなかったラウンド」で凍結する。収束ラウンド数は
#     同型に対して不変なので、同型な木は同一の色列に到達する。
#   * 1行の計算が他の行に影響しないため、窓の区切りや行の並びでは
#     結果が変わらない。
#   * 各ラウンドの色は「(自分, 折り先, 接着相手) の合成キー」をソートして
#     退止に刻印する（ties は潰す: 等しいキー=等しい要素＝同一色）。
#     成分は必ず [0, 2M) に収まるので int64 合成は衝突しない。
# ---------------------------------------------------------------------------

_LADDER_WINDOW = 2048

# WL の色更新に使う係数（mod 2^64 で折り返す）。
#
# Python の ``hash()`` と同じく「多項式を 2^64 で畳み込む」ので、等しい入力は
# 等しい色になり、異なる入力が異なる色になる確率は 2^-64 程度（同程度）。
# 参考実装（_wl_refine の hash((color[i], color[nxt], ...))）と同じ性質のまま、
# numpy の uint64 演算（配列は黙って折り返す）でベクトル計算できるようにしたもの。
#
#   新しい色 = A*自分の色 + B*折り先の色 + C*接着相手の色 + D*接着の有無 + E*接着の向き
#
# 接着が無い半辺は「接着相手の色」を 0 にし、「接着の有無」=0 で別の項にまとめる。
# 0 も色として現れるので、接着ありの項とは係数が違う（横軸が別）ことを使う。
_WL_A = np.uint64(0x9E3779B97F4A7C15)
_WL_B = np.uint64(0xC2B2AE3D27D4EB4F)
_WL_C = np.uint64(0x165667B19E3779F9)
_WL_D = np.uint64(0x27D4EB2F165667C5)
_WL_E = np.uint64(0x85EBCA77C2B2AE63)
_WL_E0 = np.uint64(0xD6E8FEB86659FD93)
_WL_D0 = np.uint64(0xA0761D6478BD642F)


def _glue_batch(
    ctx: RibbonContext, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(窓内の木数, F-1) の隣接位置から、(窓数, n_nodes) の partner / flip を作る。

    partner は接着していない半辺で -1。_glue_arrays のバッチ版で、
    値の定義は同一（:func:`_glue_arrays` の docstring 参照）。
    """
    rows = np.arange(len(positions))[:, None]
    nu = ctx.adj_nu[positions]
    nv = ctx.adj_nv[positions]
    fl = ctx.adj_fl[positions]

    partner = np.full((len(positions), ctx.n_nodes), -1, dtype=np.int64)
    partner[rows, nu] = nv
    partner[rows, nv] = nu

    flip = np.zeros((len(positions), ctx.n_nodes), dtype=np.int64)
    flip[rows, nu] = fl
    flip[rows, nv] = fl
    return partner, flip


def _distinct_counts(h: np.ndarray) -> npt.NDArray[np.int64]:
    """各行の異なる値の個数を数える（行ごとのソートが要るがベクトル1発）。

    ``np.sort(h, axis=1)`` は作業配列を確保してからソートするが、
    コピーして in-place ソートするほうが約1割速い（実測 (2048,10) uint64 で
    164 -> 148 us）。値は完全に同じ（等しい要素は同値なので安定性は無関係）。
    """
    srt = h.copy()
    srt.sort(axis=1)
    return np.asarray(1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1), dtype=np.int64)


def _batch_stabilize(
    ctx: RibbonContext,
    partner: np.ndarray,
    flip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """各行を plain WL の安定化まで回す（個別化はしない）。

    戻り値: (安定化後の色 (uint64), 各行の色数)。各行は「色数が初めて増えなかった
    ラウンド」の値で凍結する。_wl_refine のバッチ版。

    partner の -1（接着なし）は「接着相手の色」を 0 に落として区別する。
    ラウンドの回数は行ごとに違うので、安定した行は以降の計算から外す。
    """
    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    partner_or_self = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    # 接着のある半辺だけ C を掛ける（ない半辺は 0 倍 = 0 になる）。
    partner_coef = glued * _WL_C
    # 接着の有無と向きは行で変わらないので先にまとめておく。
    tag = glued * _WL_D + flip.astype(np.uint64) * _WL_E

    # 初期色: (辺長, 接着の有無, 接着の向き)。辺長は ctx 内で順位に置き換える
    # （長さが同じなら同じ色、長さが違えば違う色 = 参照実装のタプルと同じ性質）。
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * _WL_E0 + tag
    counts = _distinct_counts(h)

    nxt = ctx.np_nxt
    n_rows = len(h)
    # 安定した行は以降のラウンドで値が変化しない（分割も値も固定）ので、
    # 計算対象から外しても結果は変わらない。実際の分割は木ごと版と一致することを
    # tools/compare_partitions.py で確認している（box / frustum4 で 96 クラス一致）。
    # 行を間引くことで「安定が早い行を最後まで計算し続けるのを避ける」。
    row_ids = np.arange(n_rows, dtype=np.int64)
    out_colors = np.empty((n_rows, n_nodes), dtype=np.uint64)
    out_counts = np.empty(n_rows, dtype=np.int64)

    while True:
        nh = (
            h * _WL_A
            + h[:, nxt] * _WL_B
            + np.take_along_axis(h, partner_or_self, axis=1) * partner_coef
            + tag
        )
        new_counts = _distinct_counts(nh)
        # 安定した行は値を確定して以降から外す。h / counts / row_ids は
        # 常に同じ長さで、keep で同時に縮める（片方だけ縮めると壊れる）。
        done = new_counts == counts
        out_colors[row_ids[done]] = nh[done]
        out_counts[row_ids[done]] = new_counts[done]
        # ⚠️ 終了条件は **done.all()**。keep（= ~done）が全部 True なのは
        # 「まだ1行も安定していない」状態で、ここで break すると結果が空になる。
        if done.all():
            break
        keep = ~done
        # ⚠️ 残すのは **新しい**色 nh。h（古い色）を残すと counts だけ新しくなり、
        # 次のラウンドで必ず「色数変化なし」と判定されて 1〜2 ラウンドで止まる。
        h = nh[keep]
        partner_or_self = partner_or_self[keep]
        partner_coef = partner_coef[keep]
        tag = tag[keep]
        counts = new_counts[keep]
        row_ids = row_ids[keep]

    return out_colors, out_counts


def _batch_canonical_keys(
    ctx: RibbonContext,
    mirror_ctx: RibbonContext,
    trees: list[list[tuple[int, int]]],
) -> list[tuple[tuple[Color, ...], tuple[Color, ...]]]:
    """窓内の全木について (自身の正規形, 鏡像の正規形) を返す。

    項目の並びは ``trees`` の順。各行は plain WL の安定化までバッチで回し、
    その色がすべて別色ならそのまま正規形キーにする。まだ残ったセルがある場合は
    既存の :func:`_individualize_refine`（個別化-細分）に回す。予算超過の判定と
    ``SolidError`` の送出手続きはそちらと同一。

    キーは ``(ソート済みの色)`` のタプル。木ごと版（_net_signature）と違い、
    ハッシュ関数が違うので**キーの中身は一致しない**。一致するのは
    「同型なら同じキーになる」「同じキーになる同型は同値」という性質で、
    これは batch の挙動が木ごと版と同じであることから従う。
    """
    n = len(trees)
    keys: list[tuple[Color, ...]] = [None] * (2 * n)  # type: ignore[list-item]

    # すべての木は F-1 本の辺を持つ（全域木の定義）。辺の (u, v) を
    # (n, F-1, 2) の配列に一括変換すれば、位置 p は adj_pos_matrix の
    # ファンシー索引1回で求まる（以前は木×辺の Python 内包 + dict 引き）。
    edges_arr = np.asarray(trees, dtype=np.int64)

    for ctx_i, context in enumerate((ctx, mirror_ctx)):
        # 位置の付け直しは**文脈ごと**に行う。build_edge_map は Dict の挿入順で
        # 辺を列挙するので、面の巻き順を反転すると同じ辺でも位置番号が変わる。
        # ctx の位置を鏡像文脈でそのまま使うと別の辺を指し、
        # 鏡像が同一視されずに一意数が倍増する（実測: cube が 11 -> 30）。
        positions = context.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
        partner, flip = _glue_batch(context, positions)
        colors, counts = _batch_stabilize(context, partner, flip)
        offset = ctx_i * n
        nxt_list = context.nxt
        n_nodes = context.n_nodes
        rows_need_individualization = np.flatnonzero(counts != n_nodes)
        for idx in rows_need_individualization.tolist():
            # 色細分が止まったセルから個別化する。
            keys[offset + idx] = _individualize_refine(
                nxt_list,
                partner[idx].tolist(),
                flip[idx].tolist(),
                colors[idx].tolist(),
                [0],
            )
        # 全半辺が別色になった行 = 色列そのものが完全な同型判定になる。
        # ソートして並べた色列がそのまま正規形キー。行ごとの sorted() を
        # 呼ばず、軸1の numpy ソートで一括してから tuple 化する。
        rows_distinct = np.flatnonzero(counts == n_nodes)
        if rows_distinct.size:
            sorted_colors = np.sort(colors, axis=1).tolist()
            for idx in rows_distinct.tolist():
                keys[offset + idx] = tuple(sorted_colors[idx])
    return list(zip(keys[:n], keys[n:]))


def iter_unique_nets(
    vertices: np.ndarray,
    faces: list[list[int]],
    all_trees: Iterable[list[tuple[int, int]]],
    normals: list[np.ndarray],
) -> Iterator[tuple[list[tuple[int, int]], list[Polygon]]]:
    """一意な展開図を (全域木, 2Dポリゴン) の組として順に返す。

    重複は「正規形（同型なら必ず一致するキー）」で潰す。
    鏡像も同一視するので、自身と鏡像の正規形の小さい方をキーにする。

    シグネ計算は窓単位でバッチ化する（:_batch_canonical_keys 参照）。
    窓内では木の順序に従って重複判定・展開・重なり判定をするので、
    返り値の順序は木ごとの列挙順と一致する。

    ⚠️ このループを ``except ValueError`` で包まないこと。
    そうするとループ中のあらゆる ValueError（予算超過を含む）が
    「この木は捨てる」と解釈され、**静かに誤った一意数**を返す。
    前条件を破った呼び出しは、投げられたまま呼び出し元へ届くのが正しい。
    """
    ctx = RibbonContext(vertices, faces)
    mirror_ctx = RibbonContext(vertices, [list(reversed(f)) for f in faces])
    layout_prep = prepare_layout(vertices, faces, normals)

    exact_seen: set[tuple[Color, ...]] = set()

    rest = iter(all_trees)
    while True:
        window = list(islice(rest, _LADDER_WINDOW))
        if not window:
            break
        # レイアウトより先に重複を判定する。
        # 全域木の多くは別の木に既に含まれる重複なので、ここで弾けるほど
        # 展開と重なり判定の回数が減る（実測で全体の約7割を占めていた）。
        # 同じ正規形の木は展開図が合同であり重なり判定の結果まで一致する
        # （test_unique_nets_geometry.py で独立に検証済み）ため、順序を入れ替えても
        # 返り値は変わらない。
        key_pairs = _batch_canonical_keys(ctx, mirror_ctx, window)

        for tree, (sig, mirror_sig) in zip(window, key_pairs):
            # キーを先に登録してあるので、同型な木は展開せずに弾ける。
            key = min(sig, mirror_sig)
            if key in exact_seen:
                continue
            exact_seen.add(key)

            polys = layout_tree(vertices, faces, normals, tree, layout_prep)
            # 折り線を共有する面対は展開図でも線の反対側に並ぶので重ならない
            # （交差面積は浮動ノイズのみ）。交差計算を省く（実測で半分以上）。
            folded_pairs = {(min(u, v), max(u, v)) for u, v in tree}
            if not is_valid_layout(polys, folded_pairs):
                continue
            yield tree, polys
