import math
from collections.abc import Iterable, Iterator
from itertools import chain, islice
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

# のりしろ（glue tab）の奥行き。貼り付ける辺の長さ L に対する比で決める。
# 絶対長にすると立体を拡大・縮小しただけで形が変わり、同じ展開図とみなせなくなる。
TAB_DEPTH_RATIO = 0.25

# 台形の上辺をどれだけ内側へ逃がすか（奥行きに対する比）。
# 1.0 で側面が 45 度になり、直角の頂点で隣り合うのりしろが重なりにくい。
# 0.45L を上限にして、短い辺でも上辺が潰れないようにする。
TAB_SIDE_SLANT = 1.0
TAB_MAX_END_INSET_RATIO = 0.45

# のりしろ同士が重なるときに奥行きを縮める下限（元の奥行きに対する比）。
# 0 にすると面積 0 の「のりしろ」になるので、貼り合わせに使える大きさを残す。
# 下限まで縮めても重なりが消えない場合は、縮めず元の大きさのまま返す
# （面が角で接している展開図では、そもそもどちら側にも置けない）。
TAB_MIN_SHRINK_RATIO = 0.1
TAB_SHRINK_ITERS = 12

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
    # あわせて、同じ2面が2本以上の辺を共有していないことを確認する。
    # 展開図の親子・共有辺の対応（shared_of / _pair_to_edge）は
    # 「隣接する2面の共有辺はちょうど1本」を前提にしているため。
    shared_pairs: set[frozenset[int]] = set()
    for edge, holders in build_edge_map(faces).items():
        if len(holders) != 2:
            raise SolidError(
                "boundary_or_non_manifold",
                f"辺{edge} が{len(holders)}個の面に共有されています。"
                "閉じていないか非多様体です。",
            )
        pair = frozenset((holders[0][0], holders[1][0]))
        if pair in shared_pairs:
            u, v = sorted(pair)
            raise SolidError(
                "multiple_shared_edges",
                f"面{u} と面{v} が2本以上の辺を共有しています。"
                "1組の面が共有できる辺は1本までです。",
            )
        shared_pairs.add(pair)

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


def _is_connected_edges(cur: list[tuple[int, int, int]], num_nodes: int) -> bool:
    """``(番号, a, b)`` の3つ組の列が全 num_nodes 頂点を連結にするかを調べる。

    隣接リストを組んで BFS する代わりに、「到達した頂点のビットマスク」を辺の
    走査で伸ばして不動点まで回す（Bellman-Ford 風）。列挙の再帰が渡す入力は
    小さい（stairs で平均 4.8 辺）ので、vertex 数ぶんの list と set を確保する
    ほうが高くつく。

    実測（実際に呼ばれた入力 30,000 件を捕まえて交互計測した最小値）:
    0.90us → **0.59us（1.5x）**。判定は 30,000 件すべて一致
    （tools/probe_is_connected.py）。呼び出し側は元の辺番号付きの3つ組を
    持っているので、``_is_connected`` と違いペアのリストを作り直さずに済む。
    """
    full = (1 << num_nodes) - 1
    reach = 1
    while True:
        added = reach
        for _i, a, b in cur:
            if reach >> a & 1:
                added |= 1 << b
            elif reach >> b & 1:
                added |= 1 << a
        if added == reach:
            return reach == full
        reach = added
        if reach == full:
            return True


def _is_connected(cur: list[tuple[int, int]], num_nodes: int) -> bool:
    """辺集合が全 num_nodes 頂点を連結にするかを調べる（``(a, b)`` の列）。"""
    return _is_connected_edges([(0, a, b) for a, b in cur], num_nodes)


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
    # 辺番号 -> 端点。1 本につき 1 回引くだけなので dict より list が速い。
    endpoint_of: list[tuple[int, int]] = list(edges)
    if not _is_connected_edges(base, num_nodes):
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
                # 端点は元の辺定義から取り出すので、縮約の過程に影響されない。
                # ⚠️ chosen は既に昇順（再帰の各段で「cur の先頭にある非ループ辺」
                # を取るので、辺番号は必ず増える）。sorted() は要らない
                # （tools/check_enum_bits.py で HEAD と列として一致することを確認）。
                yield [endpoint_of[i] for i in chosen]
            return

        orig, pu, pv = cur[pivot]
        # pivot は cur の先頭にあることが多い（cur[:0] + cur[1:] の写しを1回減らす）。
        rest = cur[1:] if pivot == 0 else cur[:pivot] + cur[pivot + 1:]

        # (1) pivot を使わない全域木
        # 連結なグラフは辺が n-1 本以上必要なので、足りなければ判定ごと省く
        # （判定結果は必ず False になるので、出力は変わらない）。
        if len(rest) >= n - 1 and _is_connected_edges(rest, n):
            yield from rec(rest, n, chosen)

        # (2) pivot を使う全域木：pivot の両端を縮約して、縮約後の全域木に対応させる
        # 縮約後の頂点数は n-1 個。n-2 が pu と pv の統合先、残りの番号を詰める。
        merged = n - 2

        # remap をインライン化する（関数呼び出しを避ける）。
        # ⚠️ 整数演算のみで**幾何の丸め演算には触れない**。列挙順序も
        # 変わらないので、出力される辺列は bit 単位で一致する。
        contracted = [
            (
                i,
                merged if (a == pu or a == pv) else a - (pu < a) - (pv < a),
                merged if (b == pu or b == pv) else b - (pu < b) - (pv < b),
            )
            for i, a, b in rest
        ]
        # chosen は「今の枝の辺」を共有して push/pop する（毎段で list を写さない）。
        # 葉では新しい list を作って返すので、yield 後に書き換えても安全。
        chosen.append(orig)
        yield from rec(contracted, n - 1, chosen)
        chosen.pop()

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


# 重なり判定で GEOS を呼ばずに「面積 > 閾値」を証明するための探針。
# 面ごとに「内部に確実に入る円板」を _PROBE_COUNT 個だけ前計算しておく。
# 実測（stairs の実重なり 31,990 ペアで証明できた割合）:
# 1個 62.9% / 3個 80.3% / 4個 82.7%（tools/probe_probe_rates.py）。
_PROBE_COUNT = 4
_PROBE_GRID = 28
# 証明に使う半径は僅かに縮める（配置の浮動小数ぶんの余裕）。
_PROBE_SAFETY = 1.0 - 1e-12


def _face_probes(
    pts3: npt.NDArray[np.float64],
    count: int = _PROBE_COUNT,
    grid: int = _PROBE_GRID,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """面の内部に確実に入る円板の中心と半径を返す。

    戻り値は ``(ab, rad)``。``ab`` は面の先頭3頂点 ``(v0, v1, v2)`` を基準に
    したアフィン係数 ``(k, 2)``（点 ``v0 + a*(v1-v0) + b*(v2-v0)``）、
    ``rad`` はその点を中心とする円板の半径 ``(k,)``。

    半径は「面の境界までの距離」を格子点で評価した値そのものなので、その
    円板は面の内部に完全に含まれる（境界に接するだけ）。剛体変換（折り曲げ）
    で面を配置しても半径と係数の意味は変わらないので、展開図の座標から
    探針の位置を作れる（:func:`_layout_batch`）。

    点は「境界から遠い順」に、離れすぎない範囲で散らして選ぶ。面の形が
    細い場合でも半径 0 の点は選ばない（``dist > 0`` のみ）。
    """
    origin = pts3[0]
    e1 = pts3[1] - origin
    e1 = e1 / np.linalg.norm(e1)
    normal = np.cross(pts3[1] - origin, pts3[2] - origin)
    normal = normal / np.linalg.norm(normal)
    e2 = np.cross(normal, e1)
    rel = pts3 - origin
    pts = np.column_stack([rel @ e1, rel @ e2])  # (m, 2)

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = hi - lo
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid)
    )
    flat = np.column_stack([gx.ravel(), gy.ravel()])  # (G, 2)
    px = flat[:, 0:1]
    py = flat[:, 1:2]
    ax = pts[:, 0][None, :]
    ay = pts[:, 1][None, :]
    bx = np.roll(pts[:, 0], -1)[None, :]
    by = np.roll(pts[:, 1], -1)[None, :]
    # 偶数奇数則で内外（境界から離れた点だけを使うので縮退しない）
    cond = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_x = ax + (py - ay) * (bx - ax) / (by - ay)
    inside = (cond & (px < cross_x)).sum(axis=1) % 2 == 1
    # 境界（各辺の線分）までの距離
    ex = bx - ax
    ey = by - ay
    ee = ex * ex + ey * ey
    deg = ee <= 0.0
    safe = np.where(deg, 1.0, ee)
    t = np.clip(((px - ax) * ex + (py - ay) * ey) / safe, 0.0, 1.0)
    t = np.where(deg, 0.0, t)
    dist = np.sqrt((px - (ax + t * ex)) ** 2 + (py - (ay + t * ey)) ** 2).min(axis=1)

    ok = inside & (dist > 0.0)
    cand = flat[ok]
    clear = dist[ok]
    if len(cand) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    sep = float(np.hypot(span[0], span[1])) / (3.0 * count)
    sep2 = sep * sep
    chosen: list[int] = []
    for idx in np.argsort(-clear, kind="stable").tolist():
        if len(chosen) >= count:
            break
        cx_, cy_ = cand[idx]
        if all(
            (cx_ - cand[j, 0]) ** 2 + (cy_ - cand[j, 1]) ** 2 >= sep2 for j in chosen
        ):
            chosen.append(idx)
    sel = cand[chosen]
    rad = clear[chosen]
    mat = np.column_stack([pts[1] - pts[0], pts[2] - pts[0]])
    ab = np.linalg.solve(mat, (sel - pts[0]).T).T
    return ab, rad


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

    fold_of = _build_fold_tables(vertices, faces, normals)

    # 面ごとの探針（内部に確実に入る円板）。重なり判定で GEOS を呼ばずに
    # 「面積 > 閾値」を証明するのに使う（:func:`_proven_overlaps`）。
    n_faces = len(faces)
    lens = [len(f) for f in faces]
    max_len = max(lens)
    probe_ab = np.zeros((n_faces, _PROBE_COUNT, 2))
    probe_r = np.zeros((n_faces, _PROBE_COUNT))
    probe_anchor = np.zeros((n_faces, 3), dtype=np.intp)
    flat_pad = np.zeros((n_faces, max_len), dtype=np.intp)
    off = 0
    for fi, f in enumerate(faces):
        ab, rad = _face_probes(vertices[f])
        probe_ab[fi, : rad.size] = ab
        probe_r[fi, : rad.size] = rad
        # 投影は 3D 点についてアフィンなので、先頭3頂点の像から探針の像が出る。
        probe_anchor[fi] = (off, off + 1, off + 2)
        flat_pad[fi, : lens[fi]] = np.arange(off, off + lens[fi])
        # 余りは先頭頂点で埋める。縮退辺なので内外判定の交点は作らず、
        # 距離は「実際の辺までの距離」以上にしかならない（証明は保守側）。
        flat_pad[fi, lens[fi] :] = off
        off += lens[fi]

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
        # 面ごとの探針（(a, b) 係数と半径）と、配置用の添字。
        "probe_ab": probe_ab,
        "probe_r": probe_r,
        "probe_anchor": probe_anchor,
        "flat_pad": flat_pad,
        # 全面の頂点数が同じなら、まとめて Polygon を作れる
        # (面u, 面v) -> (折り曲げ回転行列 r, w = p_a - r @ p_a)。
        # r は共有辺と両面の法線だけで決まるので木に依存しない。
        "fold_of": fold_of,
        # 上と同じものを (u, v) の2次元配列にしたもの。窓単位のバッチ展開
        # （_layout_batch）で (親, 子) から1回のファンシー索引で引くため。
        "fold_r_tab": _fold_table(fold_of, len(faces), 0),
        "fold_w_tab": _fold_table(fold_of, len(faces), 1),
    }


def _fold_table(
    fold_of: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    n_faces: int,
    part: int,
) -> np.ndarray:
    """``fold_of`` を (面, 面, ...) の密な配列にする（隣接しない対は 0）。

    ``part=0`` が回転行列 (F, F, 3, 3)、``part=1`` が並進 (F, F, 3)。
    バッチ展開で ``tab[親, 子]`` と1回のファンシー索引で引くために使う。
    """
    sample = next(iter(fold_of.values()))[part]
    table = np.zeros((n_faces, n_faces) + sample.shape)
    for (u, v), rw in fold_of.items():
        table[u, v] = rw[part]
    return table


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

def _trees_depth_order(
    n_faces: int, trees: list[list[tuple[int, int]]]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
    """窓内の各木を面0で根付きにし、(親, 子) の列を**深さ順**で返す。

    戻り値は ``(par, child)``。どちらも ``(木数, F-1)`` で、同じ行の中では
    親が子より先に現れる（列 ``k`` を処理した時点で親の回転が確定している）。

    :func:`_tree_bfs` のバッチ版だが、**列の順序は幅優先の訪問順ではない**。
    深さが同じ面の並びは木ごとに違ってよい。回転の伝播は
    「親を先に処理する」ことだけを要求し、値は親の回転だけで決まるので、
    列の順序を変えても :func:`_layout_batch` の出力はビット単位で同じ。

    到達できない面がある木（= 全域木でない）が 1 本でもあれば ``None`` を返す。
    呼び出し側は木ごとの :func:`_tree_bfs` にフォールバックする（例外の文言まで
    従来どおりにするため）。

    親の割り当ては木の中の根からの経路で一意に決まる（同じ面を2つの親が
    同時に発見することは木では起こらない）ので、ベクトル化しても木ごと版と
    同じ親子関係になる。
    """
    n = len(trees)
    n_edges = n_faces - 1
    edges = _trees_to_array(trees, n_edges)  # (n, n_edges, 2)
    eu = edges[:, :, 0]
    ev = edges[:, :, 1]

    depth = np.full((n, n_faces), -1, dtype=np.int64)
    par = np.full((n, n_faces), -1, dtype=np.int64)
    depth[:, 0] = 0
    for _ in range(n_edges):
        du = np.take_along_axis(depth, eu, axis=1)
        dv = np.take_along_axis(depth, ev, axis=1)
        fwd = (du >= 0) & (dv < 0)
        bwd = (dv >= 0) & (du < 0)
        disc = fwd | bwd
        if not disc.any():
            break
        rows, cols = np.nonzero(disc)
        take_fwd = fwd[rows, cols]
        src = np.where(take_fwd, eu[rows, cols], ev[rows, cols])
        tgt = np.where(take_fwd, ev[rows, cols], eu[rows, cols])
        depth[rows, tgt] = depth[rows, src] + 1
        par[rows, tgt] = src
    if (depth < 0).any():
        return None

    # 深さの昇順に並べれば親が子より先に来る。同じ深さの中の順序は
    # 出力に影響しないので、安定ソートは要らない（既定のクイックソート）。
    order = np.argsort(depth, axis=1)
    child = order[:, 1:]
    return np.take_along_axis(par, child, axis=1), child


def layout_parent_relation(
    n_faces: int, tree_edges: list[tuple[int, int]]
) -> list[int | None]:
    """全域木から面の親子関係を求める（面0が根、ルートは None）。

    契約は :func:`_tree_bfs` に書いてある。折り曲げ軸は「親と共有する辺」なので、
    ここが変わると API 応答の ``fold_axis_*`` も変わる。契約として固定すること。
    """
    return _tree_bfs(n_faces, tree_edges)[0]


_EYE3 = np.eye(3)


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
    rot[0] = _EYE3
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


class LayoutBatch(NamedTuple):
    """窓単位の展開結果（Polygon は必要な行だけ後で作る）。

    ``xs``/``ys`` は窓内の全頂点の投影座標（``prepare_layout`` の ``flat_pts``
    と同じ順＝面の順に連結）。``bnds`` は面ごとの外接矩形 ``(B, F, 4)``。

    Polygon をここで作らないのは、重なり判定を通るネットが全体の 1〜2 割しか
    ないため（stairs で 37,704 中 5,714）。GEOS が要るネットだけ
    :func:`_build_polys` でまとめて作る。
    """

    xs: npt.NDArray[np.float64]
    ys: npt.NDArray[np.float64]
    bnds: npt.NDArray[np.float64]


def _build_polys(
    precomputed: Layout,
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp] | None = None,
) -> list[Polygon]:
    """選択した行の Polygon をまとめて作る（``rows=None`` なら全行）。

    面を shapely の Polygon に変換する。1 枚ずつ ``Polygon()`` を呼ぶと、頂点数が
    混在する立体（角錐台の底面など 8角形と四角形）では Python 呼び出しが
    面数だけかかる。``indices``（座標ごとの「どの面か」）は面の頂点数だけで
    決まるので前計算済み（``flat_indices``）で、行を選んだときは行番号ぶんを
    足すだけ（ring の生成は ring ごとに独立なので結果は 1 本ずつと同じ）。
    """
    if rows is None:
        x = xs
        y = ys
    else:
        x = xs[rows]
        y = ys[rows]
    n_sel = x.shape[0]
    n_faces = precomputed["probe_anchor"].shape[0]
    flat = np.column_stack([x.ravel(), y.ravel()])
    ring_idx = (
        np.arange(n_sel, dtype=np.intp)[:, None] * n_faces
        + precomputed["flat_indices"][None, :]
    ).ravel()
    built = shapely.polygons(shapely.linearrings(flat, indices=ring_idx))
    return cast(list[Polygon], built)


def _layout_batch(
    n_faces: int,
    trees: list[list[tuple[int, int]]],
    precomputed: Layout,
) -> LayoutBatch:
    """窓内の複数の木をまとめて展開する（:func:`layout_tree` と同じ結果）。

    1 本ずつ ``layout_tree`` を呼ぶと、面数が 10 程度では**計算量ではなく
    numpy 呼び出しの固定費**が支配的になる（3x3 行列積 1 回が約 1.1us、
    1 本あたり 50 回超）。木の本数ぶんまとめて軸を 1 次元増やせば、
    同じ演算を数十回の呼び出しに畳める。

    ⚠️ ビット単位で同じ結果になることが条件。以下はすべて実測で確認済み
    （``tools/check_layout_batch_bits.py``）:

    - ``np.matmul((B,3,3), (B,3,3))`` は 1 本ずつの ``(3,3)@(3,3)`` と一致
      （numpy は内側で 1 枚ずつ同じカーネルを呼ぶ）。
    - 軸 ``u @ rot`` は ``np.matmul(u, rot)`` が per-tree の einsum と一致。
    - ``(shift*u).sum(axis=2)`` は per-tree の ``sum(axis=1)`` と一致。
    - 親子の列（``_trees_depth_order``）は幅優先の訪問順とは違う並びになるが、
      回転の伝播は「親が子より先」だけを要求し、値は親の回転だけで決まるので
      同じ結果になる（``tools/check_layout_batch_bits.py``）。
    - 投影の成分ごとの積和は要素ごとの演算なので一致。
    - ``linearrings`` は ring ごとに独立なので、座標を連結しても同じ ring になる。

    ``trees`` の順に、投影座標 ``xs``/``ys`` と面ごとの外接矩形 ``bnds``
    （``(B, F, 4)`` の ``[minx, miny, maxx, maxy]``）を :class:`LayoutBatch` で返す。
    Polygon は :func:`_build_polys` で必要な行だけ作る（GEOS まで残るネットは
    全体の 1〜2 割なので、ここで全部作ると損）。
    """
    n_batch = len(trees)
    if n_batch == 0:
        return LayoutBatch(
            np.empty((0, 0)),
            np.empty((0, 0)),
            np.empty((0, n_faces, 4)),
        )
    n_edges = n_faces - 1

    par = np.empty((n_batch, n_edges), dtype=np.int64)
    child = np.empty((n_batch, n_edges), dtype=np.int64)
    ordered = _trees_depth_order(n_faces, trees)
    if ordered is None:
        # 全域木でない木が混ざっていた。木ごとの経路にフォールバックして、
        # layout_tree と同じ例外・同じメッセージに揃える（通常はここへ来ない）。
        for b, tree in enumerate(trees):
            _parents, pairs = _tree_bfs(n_faces, tree)
            if len(pairs) != n_edges:
                reached = {0}
                reached.update(v for _u, v in pairs)
                missing = next(i for i in range(n_faces) if i not in reached)
                raise ValueError(f"面{missing} へ到達できませんでした（全域木が不正です）。")
            par[b] = [u for u, _v in pairs]
            child[b] = [v for _u, v in pairs]
    else:
        par, child = ordered

    # (親, 子) -> (r, w) は 1 回のファンシー索引で全部引ける。
    rot_tab = precomputed["fold_r_tab"]
    off_tab = precomputed["fold_w_tab"]
    rr = rot_tab[par, child]  # (B, n_edges, 3, 3)
    ww = off_tab[par, child]  # (B, n_edges, 3)

    rot = np.empty((n_batch, n_faces, 3, 3))
    off = np.empty((n_batch, n_faces, 3))
    rot[:, 0] = _EYE3
    off[:, 0] = 0.0
    rows = np.arange(n_batch)

    for k in range(n_edges):
        u = par[:, k]
        v = child[:, k]
        ru = rot[rows, u]  # (B, 3, 3)
        rot[rows, v] = np.matmul(ru, rr[:, k])
        off[rows, v] = (
            np.matmul(ru, ww[:, k, :, None])[:, :, 0] + off[rows, u]
        )

    u_x = precomputed["u_x"]
    u_y = precomputed["u_y"]
    origin = precomputed["origin"]

    axes_x = np.matmul(u_x, rot)  # (B, F, 3)
    axes_y = np.matmul(u_y, rot)
    shift = off - origin
    cs_x = (shift * u_x).sum(axis=2)  # (B, F)
    cs_y = (shift * u_y).sum(axis=2)

    idx_rows = precomputed["flat_indices"]
    P = precomputed["flat_pts"]
    AX = axes_x[:, idx_rows]  # (B, M, 3)
    AY = axes_y[:, idx_rows]
    xs = (
        P[None, :, 0] * AX[:, :, 0]
        + P[None, :, 1] * AX[:, :, 1]
        + P[None, :, 2] * AX[:, :, 2]
        + cs_x[:, idx_rows]
    )
    ys = (
        P[None, :, 0] * AY[:, :, 0]
        + P[None, :, 1] * AY[:, :, 1]
        + P[None, :, 2] * AY[:, :, 2]
        + cs_y[:, idx_rows]
    )

    # 外接矩形は投影座標からそのまま作る（shapely.bounds を呼ばない）。
    # 面ごとの頂点数は違うので、最大頂点数まで先頭頂点で埋めた添字を使う
    # （先頭頂点の重複は min/max を変えない）。
    pad = precomputed["flat_pad"]
    vx = xs[:, pad]  # (B, F, max_len)
    vy = ys[:, pad]
    bnds = np.stack([vx.min(axis=2), vy.min(axis=2), vx.max(axis=2), vy.max(axis=2)], axis=2)

    return LayoutBatch(xs, ys, bnds)


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


def pair_table(n_faces: int) -> tuple[list[tuple[int, int]], npt.NDArray[np.intp]]:
    """面対 ``(i, j)``（``i < j``）を :func:`is_valid_layout` と同じ順に並べる。

    順序は ``i`` 昇順 → ``j`` 昇順。返すのは (面対のリスト, ``(i, j)`` から
    その添字を引く表)。表は ``(n_faces, n_faces)`` で、``i > j`` でも同じ
    添字を返す（折り線の面対は ``(min, max)`` で来るため）。
    """
    pairs = [(i, j) for i in range(n_faces) for j in range(i + 1, n_faces)]
    idx = np.full((n_faces, n_faces), -1, dtype=np.intp)
    for p, (i, j) in enumerate(pairs):
        idx[i, j] = p
        idx[j, i] = p
    return pairs, idx


def _probe_disk_hits(
    qx: npt.NDArray[np.float64],
    qy: npt.NDArray[np.float64],
    rad: npt.NDArray[np.float64],
    wx: npt.NDArray[np.float64],
    wy: npt.NDArray[np.float64],
    thresh: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """探針の円板が多角形の内部に完全に入るかを判定する（GEOS を呼ばない）。

    ``(qx, qy, rad)`` は探針（``(n, K)``）、``(wx, wy)`` は相手の多角形の頂点
    （``(n, M)``、余りは先頭頂点で埋めてある）。返り値は ``(n,)`` で、
    「交差面積 > thresh」を証明できた候補が ``True``。

    証明の形: 探針の中心 ``p`` が相手の内部にあり、相手の境界までの距離 ``d``
    と自分の半径 ``r`` の小さいほうを ``ρ`` とすると、半径 ``ρ`` の円板は
    両方の内部に完全に入る。よって ``area(交差) >= π ρ²``。``π ρ² > thresh``
    なら ``area(交差) > thresh`` が証明できる。

    ``p`` が内部にあるかは偶数奇数則で見る。``ρ`` が正である限り境界から
    離れているので、レイが頂点を通る等の縮退は起きない（起きた場合は
    交点の数え方が 0/2 の対で変わるだけで偶奇は保たれる）。
    """
    ex = np.roll(wx, -1, axis=1)[:, None, :] - wx[:, None, :]
    ey = np.roll(wy, -1, axis=1)[:, None, :] - wy[:, None, :]
    rx = qx[:, :, None] - wx[:, None, :]
    ry = qy[:, :, None] - wy[:, None, :]
    ee = ex * ex + ey * ey
    # 埋め草の縮退辺（長さ 0）は「先頭頂点までの距離」になる。先頭頂点も
    # 境界上の点なので、距離は実際の境界までの距離以上（証明は保守側）。
    deg = ee <= 0.0
    t = np.clip((rx * ex + ry * ey) / np.where(deg, 1.0, ee), 0.0, 1.0)
    t = np.where(deg, 0.0, t)
    dist = np.sqrt((rx - t * ex) ** 2 + (ry - t * ey) ** 2).min(axis=2)
    # 偶数奇数則（+x 方向のレイが辺を横切る回数）
    y0 = wy[:, None, :]
    y1 = np.roll(wy, -1, axis=1)[:, None, :]
    x0 = wx[:, None, :]
    x1 = np.roll(wx, -1, axis=1)[:, None, :]
    qy3 = qy[:, :, None]
    cond = (y0 > qy3) != (y1 > qy3)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (qy3 - y0) * (x1 - x0) / (y1 - y0)
    inside = ((cond & (qx[:, :, None] < xc)).sum(axis=2) % 2) == 1
    rho = np.minimum(rad, dist) * _PROBE_SAFETY
    hit = inside & (rho * rho * np.pi > thresh[:, None])
    return np.asarray(hit.any(axis=1))


def _placed_probes(
    fx: npt.NDArray[np.float64],
    fy: npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    faces: npt.NDArray[np.intp],
    precomputed: Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """候補ペアの「探針の位置」を配置済み座標から作る（(n, K) を2本）。

    投影は 3D 点についてアフィンなので、面の先頭3頂点の像とアフィン係数だけで
    探針の像が出る（:func:`_face_probes`）。
    """
    anchor = precomputed["probe_anchor"][faces]  # (n, 3)
    ab = precomputed["probe_ab"][faces]  # (n, K, 2)
    x0 = fx[rows, anchor[:, 0]][:, None]  # (n, 1)
    x1 = fx[rows, anchor[:, 1]][:, None]
    x2 = fx[rows, anchor[:, 2]][:, None]
    y0 = fy[rows, anchor[:, 0]][:, None]
    y1 = fy[rows, anchor[:, 1]][:, None]
    y2 = fy[rows, anchor[:, 2]][:, None]
    qx = x0 + ab[:, :, 0] * (x1 - x0) + ab[:, :, 1] * (x2 - x0)
    qy = y0 + ab[:, :, 0] * (y1 - y0) + ab[:, :, 1] * (y2 - y0)
    return qx, qy


def _prove_direction(
    rows: npt.NDArray[np.intp],
    probe_of: npt.NDArray[np.intp],
    poly_of: npt.NDArray[np.intp],
    layout: LayoutBatch,
    precomputed: Layout,
    ks: slice,
    thresh: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """「面 probe_of の探針（``ks`` の範囲）が面 poly_of の内部に深く入る」を判定する。"""
    fx = layout.xs
    fy = layout.ys
    pad = precomputed["flat_pad"]
    qx, qy = _placed_probes(fx, fy, rows, probe_of, precomputed)
    return _probe_disk_hits(
        qx[:, ks],
        qy[:, ks],
        precomputed["probe_r"][probe_of][:, ks],
        fx[rows[:, None], pad[poly_of]],
        fy[rows[:, None], pad[poly_of]],
        thresh,
    )


def _proven_overlaps(
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
    pair_i: npt.NDArray[np.intp],
    pair_j: npt.NDArray[np.intp],
    thresh: npt.NDArray[np.float64],
    layout: LayoutBatch,
    precomputed: Layout,
) -> npt.NDArray[np.bool_]:
    """候補ペアのうち「交差面積 > 閾値」を GEOS 無しで証明できたものを返す。

    面ごとに「内部に確実に入る円板」を前計算してあるので（:func:`_face_probes`）、
    剛体変換で配置したあとの中心と半径を使えばいい。両方向を試す。

    探針は**1本目を先に**試す（1本で 62.9%、4本で 82.7% を証明できるので、
    残り3本は証明できなかった行だけに使う。numpy の計算量がほぼ半分になる）。

    健全性: 証明できたペアは**本当に**面積 > 閾値なので、判定（重なりが1つでも
    あれば不正）は変わらない。証明できなかったペアは従来どおり
    ``Polygon.intersection`` で判定するので、呼び出し回数は増えない。
    """
    n = rows.size
    proven = np.zeros(n, dtype=bool)
    if n == 0:
        return proven
    pi = pair_i[cols]
    pj = pair_j[cols]
    # (向きを反転するか, 探針の範囲)。安い1本目を両方向→残りを両方向。
    passes: list[tuple[bool, slice]] = [(False, slice(0, 1)), (True, slice(0, 1))]
    if precomputed["probe_r"].shape[1] > 1:
        passes.append((False, slice(1, None)))
        passes.append((True, slice(1, None)))
    todo = np.ones(n, dtype=bool)
    for reverse, ks in passes:
        idx = np.nonzero(todo)[0]
        if idx.size == 0:
            break
        c = cols[idx]
        if reverse:
            probe_of, poly_of = pair_j[c], pair_i[c]
        else:
            probe_of, poly_of = pair_i[c], pair_j[c]
        proven[idx] |= _prove_direction(
            rows[idx], probe_of, poly_of, layout, precomputed, ks, thresh[idx]
        )
        todo[idx] = ~proven[idx]
    return proven


def _valid_batch(
    layout: LayoutBatch,
    trees_arr: npt.NDArray[np.intp],
    precomputed: Layout,
) -> tuple[list[bool], list[list[Polygon] | None]]:
    """窓内の全展開図の重なり判定を、**外接矩形の絞り込みだけ**一括で行う。

    :func:`is_valid_layout` の判定式をそのまま numpy で、しかも窓の全ネットに
    まとめて適用する。1 本ずつだと面数の2乗（面数 10 で 45）回る Python の
    二重ループと、ネットごとの ``shapely.bounds`` / ``.tolist()`` の固定費が
    支配的になる（実測 stairs で 2.5 秒のうち 8 割）。

    交差計算そのものは**逐次・早期終了**のまま（重なりを見つけた時点で
    打ち切るので、全ペアを計算するより速い。stairs では交差 51,605 回に対し
    候補 124,039 ペア）。

    まず候補ペアのうち「交差面積 > 閾値」を numpy だけで証明できたものを落とす
    （:func:`_proven_overlaps`）。判定は「重なりが1つでもあれば不正」なので
    順序に依らず、証明できた候補を持つネットはその場で不正が確定する。
    証明できなかった候補は従来どおり :meth:`Polygon.intersection` で判定する。
    よって**同じ判定**を、より少ない GEOS 呼び出しで得られる
    （証明の健全性は :func:`_probe_disk_hits` を参照）。

    Polygon は**GEOS まで残ったネットと、候補が 1 つも無いネットだけ**作る
    （前者は判定に、後者は出力に要る。候補が無いネットは必ず有効）。
    stairs では 37,704 ネットのうち 5,714 ネットぶんで済む。

    ``trees_arr`` は ``(B, F-1, 2)`` の辺（折り線を共有する面対は重ならないので
    候補から外す）。返り値は (有効フラグ, ネットごとの Polygon か None)。
    """
    n_batch = layout.bnds.shape[0]
    if n_batch == 0:
        return [], []
    n_faces = layout.bnds.shape[1]
    if n_faces < 2:
        built = _build_polys(precomputed, layout.xs, layout.ys)
        return [True] * n_batch, [
            built[b * n_faces : (b + 1) * n_faces] for b in range(n_batch)
        ]
    bnds = layout.bnds
    pairs, pair_idx = pair_table(n_faces)
    pair_i = np.array([i for i, _ in pairs], dtype=np.intp)
    pair_j = np.array([j for _, j in pairs], dtype=np.intp)

    # 面対ごとの外接矩形を (B, n_pairs) に引き伸ばし、判定を全部まとめて行う。
    bi = bnds[:, pair_i, :]
    bj = bnds[:, pair_j, :]
    ax0, ay0, ax1, ay1 = bi[:, :, 0], bi[:, :, 1], bi[:, :, 2], bi[:, :, 3]
    bx0, by0, bx1, by1 = bj[:, :, 0], bj[:, :, 1], bj[:, :, 2], bj[:, :, 3]
    near = (ax1 >= bx0) & (bx1 >= ax0) & (ay1 >= by0) & (by1 >= ay0)
    dx = np.maximum(ax1, bx1) - np.minimum(ax0, bx0)
    dy = np.maximum(ay1, by1) - np.minimum(ay0, by0)
    thresh = OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
    ox = np.minimum(ax1, bx1) - np.maximum(ax0, bx0)
    oy = np.minimum(ay1, by1) - np.maximum(ay0, by0)
    cand = near & (ox * oy > thresh)
    # 折り線を共有する面対は展開図でも線の反対側に並ぶので重ならない
    # （交差面積は浮動ノイズのみ）。交差計算を丸ごと省く。
    cand[np.arange(n_batch, dtype=np.intp)[:, None], pair_idx[trees_arr[:, :, 0], trees_arr[:, :, 1]]] = False

    # 交差計算に進む面対を、:func:`is_valid_layout` と同じ順（ネットごとに
    # 添字昇順）で取り出して、ネットごとに逐次・早期終了で回す。
    rows, cols = np.nonzero(cand)
    out = [True] * n_batch
    counts_all = (
        np.bincount(rows, minlength=n_batch)
        if rows.size
        else np.zeros(n_batch, dtype=np.intp)
    )
    if rows.size:
        # 候補を「ネットごとに1つずつ」波状に処理する。証明できたネットはその時点で
        # 不正が確定するので打ち切り、残りは次の候補へ進む（判定は順序に依らない）。
        # 全候補（stairs で 124,039 件）を一度に証明しようとすると、証明の numpy 計算
        # そのものが `intersection` より重くなる（実測 0.67s）。ネットの平均候補数は
        # 3.3 なので、波状にすると計算は約 1/2.3 で同じネットが証明できる。
        starts = np.zeros(n_batch, dtype=np.intp)
        np.cumsum(counts_all[:-1], out=starts[1:])
        bad = np.zeros(n_batch, dtype=bool)
        alive = counts_all > 0
        wave = 0
        while True:
            nets = np.nonzero(alive)[0]
            if nets.size == 0:
                break
            sel = starts[nets] + wave
            hit = _proven_overlaps(
                rows[sel],
                cols[sel],
                pair_i,
                pair_j,
                thresh[rows[sel], cols[sel]],
                layout,
                precomputed,
            )
            if hit.any():
                bad[nets[hit]] = True
            alive[:] = False
            alive[nets[(~hit) & (counts_all[nets] > wave + 1)]] = True
            wave += 1
        if bad.any():
            out = (~bad).tolist()
            keep = ~bad[rows]
            rows = rows[keep]
            cols = cols[keep]

    # Polygon を作るネット: GEOS まで残ったネット（判定に要る）と、
    # 候補が 1 つも無いネット（必ず有効なので出力に要る）。
    need = counts_all == 0
    need[rows] = True
    need_idx = np.flatnonzero(need)
    built = _build_polys(precomputed, layout.xs, layout.ys, need_idx)
    polys_by_net: list[list[Polygon] | None] = [None] * n_batch
    for b in range(need_idx.size):
        base = b * n_faces
        polys_by_net[int(need_idx[b])] = built[base : base + n_faces]
    if rows.size == 0:
        return out, polys_by_net

    counts = np.bincount(rows, minlength=n_batch)
    col_list = cols.tolist()
    thr_list = thresh[rows, cols].tolist()
    pos = 0
    for k in range(n_batch):
        n_cand = int(counts[k])
        if n_cand == 0:
            continue
        polys = polys_by_net[k]
        assert polys is not None
        end = pos + n_cand
        for t in range(pos, end):
            i, j = pairs[col_list[t]]
            if polys[i].intersection(polys[j]).area > thr_list[t]:
                out[k] = False
                break
        pos = end
    return out, polys_by_net


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
    """{(面u, 面v): 共有辺} のマップを作る。

    同じ2面が2本以上の辺を共有する入力は受け付けない。辞書の内包表記で
    黙って上書きすると、切れている辺の片方がのりしろ無しになるため。
    """
    pair_to_edge: dict[frozenset[int], tuple[int, int]] = {}
    for u, v, edge in build_adjacency_with_edges(faces):
        pair = frozenset((u, v))
        if pair in pair_to_edge:
            a, b = sorted(pair)
            raise SolidError(
                "multiple_shared_edges",
                f"面{a} と面{b} が2本以上の辺を共有しています。"
                "1組の面が共有できる辺は1本までです。",
            )
        pair_to_edge[pair] = edge
    return pair_to_edge


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


class GlueTab(NamedTuple):
    """展開図の外周に付けるのりしろ 1 枚分。

    のりしろは「展開図では直接つながっていないが、立体にすると接する辺」
    （= 全域木に含まれない共有辺）の**片側だけ**に付ける。両側に付けると
    貼り合わせが二重になり、どちらにも付けないと閉じない。

    ``vertices_2d`` は ``[貼り付け辺の始点, 終点, 外側の終点側, 外側の始点側]``
    の順（台形）。最初の 2 点が面の外周と共有する辺になる。
    """

    face_id: int
    neighbor_face_id: int
    edge_3d: list[list[float]]
    edge_2d: list[list[float]]
    vertices_2d: list[list[float]]


class _TabCandidate(NamedTuple):
    """1 本の共有辺について、片側の面に付けた場合ののりしろ。"""

    face_id: int
    neighbor_face_id: int
    edge_3d: list[list[float]]
    edge_2d: list[list[float]]
    vertices_2d: list[list[float]]
    polygon: Polygon
    # 重なり判定で何度も呼ぶので、外接矩形と面積を先に持っておく。
    bounds: tuple[float, float, float, float]
    area: float


def _tab_loop_coords(poly: Polygon) -> list[tuple[float, float]]:
    """Polygon の外周を、閉じるための重複点を除いて返す。"""
    x, y = poly.exterior.xy
    return [(float(x[i]), float(y[i])) for i in range(len(x) - 1)]


def _tab_signed_area(coords: list[tuple[float, float]]) -> float:
    """外周の符号付き面積。正なら反時計回り（内側は進行方向の左）。"""
    total = 0.0
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % len(coords)]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _make_tab_candidate(
    coords: list[tuple[float, float]],
    signed_area: float,
    face: list[int],
    face_id: int,
    neighbor_face_id: int,
    a: int,
    b: int,
    vertices: np.ndarray,
) -> _TabCandidate:
    """共有辺 ``(a, b)`` を面 ``face_id`` 側に貼る場合の台形を作る。

    ``coords`` は ``polys[face_id]`` の外周で、頂点の並びは ``face`` と同じ。
    法線の向きは外周の符号付き面積から決める（面の内外を点内含判定で
    調べないので、細い面や丸めの影響を受けない）。
    """
    n = len(face)
    idx = -1
    for i in range(n):
        if {face[i], face[(i + 1) % n]} == {a, b}:
            idx = i
            break
    if idx < 0:
        # pair_to_edge は build_adjacency_with_edges の結果なので、
        # ここに来るのは呼び出し側が面と辺の対応を壊したときだけ。
        raise ValueError(
            f"面{face_id} に辺({a}, {b}) が見つかりません（pair_to_edge が不正です）。"
        )

    p0x, p0y = coords[idx]
    p1x, p1y = coords[(idx + 1) % n]
    dx = p1x - p0x
    dy = p1y - p0y
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError(f"面{face_id} の辺({a}, {b}) の長さが 0 です。")
    ux, uy = dx / length, dy / length

    # CCW なら内側は進行方向の左なので、外側は右。CW ならその逆。
    if signed_area >= 0.0:
        ox, oy = uy, -ux
    else:
        ox, oy = -uy, ux

    depth = TAB_DEPTH_RATIO * length
    inset = min(TAB_SIDE_SLANT * depth, TAB_MAX_END_INSET_RATIO * length)

    # 台形: 貼り付け辺 p0->p1 から外側へ depth 出し、上辺を左右に inset 逃がす。
    q0x = p0x + ox * depth + ux * inset
    q0y = p0y + oy * depth + uy * inset
    q1x = p1x + ox * depth - ux * inset
    q1y = p1y + oy * depth - uy * inset

    verts = [
        [p0x, p0y],
        [p1x, p1y],
        [q1x, q1y],
        [q0x, q0y],
    ]
    ea, eb = face[idx], face[(idx + 1) % n]
    polygon = Polygon(verts)
    return _TabCandidate(
        face_id=face_id,
        neighbor_face_id=neighbor_face_id,
        edge_3d=[
            [float(c) for c in vertices[ea]],
            [float(c) for c in vertices[eb]],
        ],
        edge_2d=[[p0x, p0y], [p1x, p1y]],
        vertices_2d=verts,
        polygon=polygon,
        bounds=polygon.bounds,
        area=polygon.area,
    )


def _intersection_area(
    a: Polygon,
    b: Polygon,
    a_bounds: tuple[float, float, float, float],
    b_bounds: tuple[float, float, float, float],
    a_area: float,
    b_area: float,
) -> float:
    """外接矩形で棄却してから交差面積を求める（閾値は局所面積）。"""
    if (
        a_bounds[2] < b_bounds[0]
        or b_bounds[2] < a_bounds[0]
        or a_bounds[3] < b_bounds[1]
        or b_bounds[3] < a_bounds[1]
    ):
        return 0.0
    area = a.intersection(b).area
    threshold = OVERLAP_AREA_REL_TOL * min(a_area, b_area)
    return float(area) if area > threshold else 0.0


def _tab_overlap_area(a: Polygon, b: Polygon) -> float:
    """2 つのポリゴンの交差面積。浮動ノイズは 0 とみなす。

    閾値は**小さいほうのポリゴンの面積**に対する比にする。展開図全体の
    大きさを基準にすると、長い辺と短い辺が混在する展開図で、短い辺の
    のりしろが丸ごと面に乗っていても「重なり 0」と誤判定してしまう。
    """
    return _intersection_area(a, b, a.bounds, b.bounds, a.area, b.area)


def _candidate_overlap(a: _TabCandidate, b: _TabCandidate) -> float:
    """のりしろ候補同士の交差面積（外接矩形と面積は候補が持つ）。"""
    return _intersection_area(
        a.polygon, b.polygon, a.bounds, b.bounds, a.area, b.area
    )


def _scale_tab_depth(c: _TabCandidate, ratio: float) -> _TabCandidate:
    """台形を貼り付け辺に近づけて、奥行きを ``ratio`` 倍に縮める。

    上辺の 2 点を貼り付け辺の両端へ線形に寄せる。奥行きと上辺の逃げ幅が
    同じ比で縮むので、側面 45 度の台形のまま小さくなる（元の台形に含まれる）。
    """
    p0, p1, q1, q0 = c.vertices_2d

    def lerp(p: list[float], q: list[float]) -> list[float]:
        return [p[0] + ratio * (q[0] - p[0]), p[1] + ratio * (q[1] - p[1])]

    verts = [p0, p1, lerp(p1, q1), lerp(p0, q0)]
    polygon = Polygon(verts)
    return _TabCandidate(
        face_id=c.face_id,
        neighbor_face_id=c.neighbor_face_id,
        edge_3d=c.edge_3d,
        edge_2d=c.edge_2d,
        vertices_2d=verts,
        polygon=polygon,
        bounds=polygon.bounds,
        area=polygon.area,
    )


def _tab_clear(
    c: _TabCandidate,
    obstacles: list[tuple[Polygon, tuple[float, float, float, float], float]],
) -> bool:
    """のりしろが障害物（面・他ののりしろ）と重ならないか。"""
    for poly, bounds, area in obstacles:
        if _intersection_area(c.polygon, poly, c.bounds, bounds, c.area, area) > 0.0:
            return False
    return True


def _refine_tab_depths(
    chosen: list[_TabCandidate],
    polys: list[Polygon],
) -> list[_TabCandidate]:
    """重なっているのりしろの奥行きを、重なりが消えるまで縮める。

    奥行きを縮めた台形は元の台形に含まれるので、重なりは単調に減る。
    2 分探索で「重ならない最大の奥行き」を探す。1 枚ずつ処理し、後の
    のりしろは元の大きさで判定する（後で縮むと重なりは減るだけなので、
    先に決めた側が壊れることはない）。

    下限まで縮めても消えない重なりは、面が角で接しているため置き場所が
    ない。その場合は縮めず元の大きさのまま返す。
    """
    face_obstacles = [(poly, poly.bounds, poly.area) for poly in polys]

    result: list[_TabCandidate] = []
    for i, c in enumerate(chosen):
        obstacles = [
            obstacle
            for face_id, obstacle in enumerate(face_obstacles)
            if face_id != c.face_id
        ]
        obstacles += [(r.polygon, r.bounds, r.area) for r in result]
        obstacles += [(r.polygon, r.bounds, r.area) for r in chosen[i + 1 :]]

        if _tab_clear(c, obstacles):
            result.append(c)
            continue

        smallest = _scale_tab_depth(c, TAB_MIN_SHRINK_RATIO)
        if not _tab_clear(smallest, obstacles):
            result.append(c)
            continue

        lo, hi = TAB_MIN_SHRINK_RATIO, 1.0
        for _ in range(TAB_SHRINK_ITERS):
            mid = (lo + hi) / 2.0
            if _tab_clear(_scale_tab_depth(c, mid), obstacles):
                lo = mid
            else:
                hi = mid
        result.append(_scale_tab_depth(c, lo))
    return result


def _cost_less(a: float, b: float) -> bool:
    """``a`` が ``b`` より「意味のある差」で小さいか。

    展開図を拡大・縮小すると重なり面積はスケール^2 で変わる。厳密な
    ``<`` で比べると、数学的には同値な候補の差が浮動小数ノイズで
    逆転し、選ぶ側がスケールで変わってしまう（実測: L 字柱 1e-6 倍で
    1516 枚中 280 枚）。相対トレランス内の差は同値とみなす。
    """
    return a < b and not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def _pick_side(costs: list[float]) -> int:
    """コストが最小の候補を選ぶ。同値なら先頭（面番号の小さい側）。"""
    best = 0
    for k in range(1, len(costs)):
        if _cost_less(costs[k], costs[best]):
            best = k
    return best


def _choose_tab_sides(
    candidates: list[list[_TabCandidate]],
    polys: list[Polygon],
) -> tuple[list[_TabCandidate], bool]:
    """各共有辺について、重なりが小さくなる側を 1 つ選ぶ。

    のりしろ同士・のりしろと面の重なり面積をコストにし、まず貪欲に決めて
    から 1 枚ずつ反転して改善する。``candidates`` の各要素は必ず 2 枚
    （両側）なので、結果は「各辺ちょうど 1 枚」を構造的に満たす。

    戻り値は ``(選んだ候補, 重なりが残っているか)``。重なりが残って
    いれば、呼び出し側が :func:`_refine_tab_depths` で奥行きを縮める。
    """
    if not candidates:
        return [], False

    # 面の外接矩形と面積は何度も引くので先に取る（GEOS 呼び出しを減らす）。
    face_bounds = [poly.bounds for poly in polys]
    face_areas = [poly.area for poly in polys]

    def unary_of(c: _TabCandidate) -> float:
        return sum(
            _intersection_area(
                c.polygon, poly, c.bounds, fb, c.area, fa
            )
            for poly, fb, fa in zip(polys, face_bounds, face_areas)
        )

    unary = [[unary_of(c) for c in pair] for pair in candidates]

    chosen: list[_TabCandidate] = []
    chosen_idx: list[int] = []
    pair_overlap = False
    for i, pair in enumerate(candidates):
        pair_costs = [
            sum(_candidate_overlap(pair[k], chosen[j]) for j in range(len(chosen)))
            for k in range(len(pair))
        ]
        costs = [unary[i][k] + pair_costs[k] for k in range(len(pair))]
        # 同コストなら先頭（面番号の小さい側）を選ぶ。出力を決定的にするため。
        k = _pick_side(costs)
        if pair_costs[k] > 0.0:
            pair_overlap = True
        chosen.append(pair[k])
        chosen_idx.append(k)

    face_overlap = any(
        unary[i][chosen_idx[i]] > 0.0 for i in range(len(chosen))
    )
    # のりしろ同士が重なっていなければ、各辺は自分の重なりが最小の側を
    # 選んでいる。反転しても面との重なりは増えるだけなので、改善しない。
    if pair_overlap:
        # 貪欲の後ろで選んだ側が、後から来たのりしろと重なることがある。
        # 1 枚ずつ反転してコストが下がる限り繰り返す（局所最適）。
        for _ in range(8):
            improved = False
            for i, pair in enumerate(candidates):
                if len(pair) < 2:
                    continue
                other_unary = [
                    sum(
                        _candidate_overlap(pair[k], chosen[j])
                        for j in range(len(chosen))
                        if j != i
                    )
                    for k in range(len(pair))
                ]
                costs = [unary[i][k] + other_unary[k] for k in range(len(pair))]
                best = _pick_side(costs)
                cur = chosen_idx[i]
                # 意味のある改善があるとき、または同値で先頭側に寄せられるときに
                # 切り替える。同値判定に相対トレランスを使うので、拡大・縮小しても
                # 選ぶ側が変わらない。
                if best != cur and not _cost_less(costs[cur], costs[best]):
                    chosen[i] = pair[best]
                    chosen_idx[i] = best
                    improved = True
            if not improved:
                break
        face_overlap = any(
            unary[i][chosen_idx[i]] > 0.0 for i in range(len(chosen))
        )
        pair_overlap = any(
            _candidate_overlap(chosen[i], chosen[j]) > 0.0
            for i in range(len(chosen))
            for j in range(i + 1, len(chosen))
        )
    return chosen, face_overlap or pair_overlap


def glue_tabs(
    polys: list[Polygon],
    faces: list[list[int]],
    tree_edges: list[tuple[int, int]],
    pair_to_edge: dict[frozenset[int], tuple[int, int]],
    vertices: np.ndarray,
) -> list[GlueTab]:
    """展開図の「切れている辺」に、ちょうど片側ずつのりしろを付ける。

    切れている辺とは、面グラフの隣接辺のうち全域木に含まれないもの
    （= 展開図では直接つながっておらず、立体にすると接する辺）。
    各辺について両面の候補を作り、重なりが小さい側を 1 つだけ選ぶので、
    「両方に付く」「どちらにも付かない」は起こらない。

    のりしろは面のポリゴンには含めない。したがって ``net_metrics`` の
    外周長・外接矩形・折る回数は、のりしろを付ける前と同じ値になる
    （仕様では折る回数からのりしろを無視する）。

    のりしろ同士が重なる場合は、重なりが消えるまで奥行きを縮める
    （:func:`_refine_tab_depths`）。面が角で接していて、どちら側にも
    置き場所がない展開図では、重なりを完全には消せない。
    """
    n_faces = len(faces)
    tree_pairs = {frozenset(edge) for edge in tree_edges}
    cut_pairs = [pair for pair in pair_to_edge if pair not in tree_pairs]
    # 面対の昇順に処理する。集合の反復順に依存せず、出力を決定的にする。
    cut_pairs.sort(key=lambda pair: tuple(sorted(pair)))

    coords = [_tab_loop_coords(poly) for poly in polys]
    areas = [_tab_signed_area(loop) for loop in coords]

    candidates: list[list[_TabCandidate]] = []
    for pair in cut_pairs:
        u, v = sorted(pair)
        a, b = pair_to_edge[pair]
        sides = []
        for face_id, neighbor in ((u, v), (v, u)):
            if face_id >= n_faces:
                raise ValueError("面番号が面数以上です。")
            sides.append(
                _make_tab_candidate(
                    coords[face_id],
                    areas[face_id],
                    faces[face_id],
                    face_id,
                    neighbor,
                    a,
                    b,
                    vertices,
                )
            )
        candidates.append(sides)

    chosen, has_overlap = _choose_tab_sides(candidates, polys)
    if has_overlap:
        chosen = _refine_tab_depths(chosen, polys)
    return [
        GlueTab(
            face_id=c.face_id,
            neighbor_face_id=c.neighbor_face_id,
            edge_3d=c.edge_3d,
            edge_2d=c.edge_2d,
            vertices_2d=c.vertices_2d,
        )
        for c in chosen
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

# 色数を数え始めるラウンド。実測（stairs）では 7 ラウンド目までに安定する行は
# 2048 行中 182 行だけで、1〜6 ラウンド目はほぼ全部が残る。その区間の
# `_distinct_counts` はほぼ捨て仕事なので、数えないで進める（署名フェーズ約15%減）。
#
# ⚠️ これは速度だけでなく**正しさのためにも効く**。バッチ版の色は
# ``A*h[i] + B*h[nxt] + C*h[partner] + ...`` という uint64 の線形結合なので、
# 早期のラウンドでは値の線形従属が残っていて衝突が多い。実測（lshape, 8100 行）
# では 3 ラウンド目で 66 行、4 ラウンド目で 28 行が「実際のクラス数より 1 少ない」
# 色数になり、そのせいで「分割が起きたのに色数が変わらない」＝**見かけの安定**が
# 起きる（旧実装では lshape の 9 行がここで早期に凍結し、個別化で拾い直していた）。
# 5 ラウンド目以降は衝突 0 なので、その手前を数えないことで偽の安定を取り込まない。
# 途中で安定した行もここまで回すが、安定後の WL は分割を変えず値だけが決定的に
# 進むので正規形としては同じ。全行が同じ回数以上回るので同型判定も保たれる。
_WL_MIN_ROUNDS = 5


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

    ただし最初の ``_WL_MIN_ROUNDS`` ラウンドは色数を数えない（下の説明を参照）。
    安定した行は以降のラウンドで分割も色も変化しないので、計算対象から外しても
    結果は変わらない。分割が木ごと版と一致することは tools/compare_partitions.py で
    確認している（box / frustum4 で 96 クラス一致）。

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

    nxt = ctx.np_nxt
    n_rows = len(h)
    # 安定した行は以降のラウンドで値が変化しない（分割も値も固定）ので、
    # 計算対象から外しても結果は変わらない。実際の分割は木ごと版と一致することを
    # tools/compare_partitions.py で確認している（box / frustum4 で 96 クラス一致）。
    # 行を間引くことで「安定が早い行を最後まで計算し続けるのを避ける」。
    row_ids = np.arange(n_rows, dtype=np.int64)
    out_colors = np.empty((n_rows, n_nodes), dtype=np.uint64)
    out_counts = np.empty(n_rows, dtype=np.int64)

    # 作業配列は最大行数で確保して使い回す。
    # ⚠️ ラウンドごとに (行, n_nodes) の一時配列を新規確保すると、その実体化
    # （初回書き込みのページフォルト）だけで更新1回あたり数百 us を払う。
    # 実測（2048x48）: 確保あり 1250us → 事前確保 + 平坦添字 gather 469us = 2.7 倍。
    # 出力バッファは 2 枚を交互に使う（入力 h と別でなければ壊れる）。
    t_buf = np.empty((n_rows, n_nodes), dtype=np.uint64)
    t2_buf = np.empty((n_rows, n_nodes), dtype=np.uint64)
    nh_a = np.empty((n_rows, n_nodes), dtype=np.uint64)
    nh_b = np.empty((n_rows, n_nodes), dtype=np.uint64)
    use_a = True
    # 接着相手の色を引く添字。行ごとに変わるので 2 次元 gather
    # （take_along_axis は実測 260us）より、平坦化した 1 次元 gather
    # （実測 52us）のほうが速い。行を間引いたときだけ作り直す。
    row_off = np.arange(n_rows, dtype=np.intp)[:, None] * n_nodes
    flat_partner = partner_or_self + row_off

    # 最初の _WL_MIN_ROUNDS ラウンドは色数を数えない。
    # 実測（tools/probe_stabilize_parts.py、stairs）では 7 ラウンド目までに
    # 安定する行は 2048 行中 182 行だけで、1〜6 ラウンド目はほぼ全部が残る。
    # その区間の `_distinct_counts`（ラウンド費用の約 3 割）はほぼ捨て仕事になる。
    #
    # ⚠️ これは速度だけでなく**正しさのためにも効く**。バッチ版の色は
    # ``A*h[i] + B*h[nxt] + C*h[partner] + ...`` という uint64 の線形結合なので、
    # 早期のラウンドでは値の線形従属が残っていて衝突が多い。実測（lshape, 8100 行）
    # では 3 ラウンド目で 66 行、4 ラウンド目で 28 行が「実際のクラス数より 1 少ない」
    # 色数になり、そのせいで「分割が起きたのに色数が変わらない」＝**見かけの安定**が
    # 起きる（旧実装では lshape の 9 行がここで早期に凍結し、個別化で拾い直していた）。
    # 5 ラウンド目以降は衝突 0 なので、その手前を数えないことで偽の安定を取り込まない。
    # 途中で安定した行もここまで回すが、安定後の WL は分割を変えず値だけが決定的に
    # 進むので正規形としては同じ。全行が同じ回数以上回るので同型判定も保たれる。
    counts: npt.NDArray[np.int64] | None = None
    rnd = 0

    while True:
        rnd += 1
        n = len(h)
        # ⚠️ 入力 h と出力 nh は必ず別の実体にする（2 枚を交互に使う）。
        nh = (nh_a if use_a else nh_b)[:n]
        use_a = not use_a
        t = t_buf[:n]
        t2 = t2_buf[:n]
        # 1 本の式で書くと (rows, n_nodes) の一時配列が 9 個できる。逐次 in-place
        # なら確保は 0 個。値を足す順序は元の式と同一なので**ビット単位で同じ**。
        # （uint64 は mod 2^64 で閉じていて和も積も結合法則・分配法則が成り立つので、
        #   実は足す順序を変えても値は変わらない。tools/check_u64_ring.py。
        #   それでも元の順序のままにしてあるのは、木ごと版との対応が自明だから。）
        np.multiply(h, _WL_A, out=nh)
        np.copyto(t, h[:, nxt])
        np.multiply(t, _WL_B, out=t)
        np.add(nh, t, out=nh)
        # 接着相手の色は「行オフセットを足した平坦添字」で 1 次元 gather する
        # （flat_partner は行を間引いたときだけ作り直す）。
        np.copyto(t2, h.reshape(-1)[flat_partner[:n].reshape(-1)].reshape(n, n_nodes))
        np.multiply(t2, partner_coef[:n], out=t2)
        np.add(nh, t2, out=nh)
        np.add(nh, tag, out=nh)
        if rnd <= _WL_MIN_ROUNDS:
            # 色数を数えずに次へ。この区間では行の間引きもしない（ほぼ空振り）。
            h = nh
            if rnd < _WL_MIN_ROUNDS:
                continue
            counts = _distinct_counts(nh)
            continue
        assert counts is not None
        # 全色が別に達した行（色数 == n_nodes）はもう細分できないので、色数を
        # 数えなくとも次のラウンドで done が確定する（色数は n_nodes のまま）。
        # その行を除いて数えると `_distinct_counts` の並べ替えが減る
        # （実測 stairs で row-round の 12.6%、lshape で 15.8% がこの行）。
        # 色数の値も n_nodes のままなので counts をそのまま流用できる。
        full = counts == n_nodes
        done = full
        new_counts = counts.copy()
        part = ~full
        if part.any():
            part_counts = _distinct_counts(nh[part])
            new_counts[part] = part_counts
            done[part] = part_counts == counts[part]
        # 安定した行は値を確定して以降から外す。h / counts / row_ids は
        # 常に同じ長さで、keep で同時に縮める（片方だけ縮めると壊れる）。
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
        # 平坦添字は行番号に依存するので、間引いたときだけ作り直す。
        np.add(partner_or_self, row_off[: len(h)], out=flat_partner[: len(h)])

    return out_colors, out_counts


def _key_bytes(colors: Iterable[Color]) -> bytes:
    """個別化まで行った行の正規形キーを決定的なバイト列にする。

    キーを bytes に統一するのは、窓内のキーを ``min(自身, 鏡像)`` で比べるため
    （型が混ざると Python の ``min`` が TypeError を投げる）。

    ⚠️ 全色が別の行（8 バイト/色の ``>u8`` 連結）とは**長さで区別できる**
    （こちらは 9 バイト/色）。色数が n_nodes 未満の行なので、そもそも全色別の
    行と同じキーにはなり得ない。**順序に意味は無い**（意味があるのは等しいか
    どうかだけ）が、単射であることは要る。色は int（``hash()`` の結果は負もあり得る）
    なので符号 1 バイト + 絶対値 8 バイトで書く。
    """
    out = bytearray()
    for c in colors:
        if c < 0:
            out += b"\x01" + (-c).to_bytes(8, "big")
        else:
            out += b"\x00" + c.to_bytes(8, "big")
    return bytes(out)


def _trees_to_array(
    trees: list[list[tuple[int, int]]], n_edges: int
) -> npt.NDArray[np.intp]:
    """全域木のリストを ``(n, F-1, 2)`` の辺配列に落とす。

    ``np.asarray(trees, dtype=np.int64)`` は「木のリスト → 辺のタプル」という
    入れ子を汎用の object 経路で走るので遅い（実測 stairs の 2048 本で 2.5ms）。
    平坦な int 列を ``np.fromiter`` に流して整形すると 0.83ms（3.0x）。

    要素の並び（木の順 → 辺の順 → ``u``, ``v`` の順）は `np.asarray` と同一で、
    値も完全に一致する（tools/probe_asarray.py で照合）。戻り値は添字に使うので
    ``intp``。
    """
    n = len(trees)
    return np.fromiter(
        chain.from_iterable(chain.from_iterable(trees)),
        dtype=np.intp,
        count=n * n_edges * 2,
    ).reshape(n, n_edges, 2)


def _context_keys(
    context: RibbonContext,
    trees: list[list[tuple[int, int]]],
    edges_arr: npt.NDArray[np.intp] | None = None,
) -> list[bytes]:
    """1 つの文脈（巻き順）で、窓の全木の正規形キーを返す。

    行（木）ごとに独立な計算なので、``trees`` を**窓の一部だけにしても
    その行の結果は変わらない**。この性質が :func:`iter_unique_nets` の
    「鏡像の正規形は必要な行だけ計算する」高速化の根拠になっている。

    キーは行によらず bytes。全色が別の行は「ソート済みの色列をビッグエンディアンで
    並べた 8 バイト/色の連結」、個別化に回った行は :func:`_key_bytes` による
    9 バイト/色。木ごと版（_net_signature）と違い、ハッシュ関数が違うので
    **キーの中身は一致しない**。一致するのは「同型なら同じキーになる」
    「同じキーになる同型は同値」という性質で、これは batch の挙動が木ごと版と
    同じであることから従う。2 種類のキーは長さが違うので互いに一致しない
    （個別化行の色数は n_nodes 未満で、全色が別の行にはなれない）。
    """
    n = len(trees)
    keys: list[bytes] = cast(list[bytes], [None] * n)
    if n == 0:
        return keys
    if edges_arr is None:
        edges_arr = _trees_to_array(trees, len(trees[0]))
    # 位置の付け直しは**文脈ごと**に行う。build_edge_map は Dict の挿入順で
    # 辺を列挙するので、面の巻き順を反転すると同じ辺でも位置番号が変わる。
    # 片方の位置を他方で流用すると別の辺を指し、鏡像が同一視されずに
    # 一意数が倍増する（実測: cube が 11 -> 30）。
    positions = context.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
    partner, flip = _glue_batch(context, positions)
    colors, counts = _batch_stabilize(context, partner, flip)
    nxt_list = context.nxt
    n_nodes = context.n_nodes
    # 色細分が止まったセルから個別化する。
    for idx in np.flatnonzero(counts != n_nodes).tolist():
        keys[idx] = _key_bytes(
            _individualize_refine(
                nxt_list,
                partner[idx].tolist(),
                flip[idx].tolist(),
                colors[idx].tolist(),
                [0],
            )
        )
    # 全半辺が別色になった行 = 色列そのものが完全な同型判定になる。
    # ソートして並べた色列がそのまま正規形キー。行ごとの sorted() を
    # 呼ばず、軸1の numpy ソートで一括する。
    #
    # ⚠️ キーはタプルではなく**ビッグエンディアンのバイト列**にする。
    # tuple にすると ``.tolist()`` で (2048, 48) ぶんの Python int を
    # 作ることになり、この行が署名フェーズの1割を占めていた（実測 0.44s）。
    # ビッグエンディアンならバイト列の辞書順は uint64 の数値順と一致し、
    # 固定幅の連結はタプルの辞書順と一致するので、``min(自身, 鏡像)`` の
    # 選び方まで tuple 版と**完全に同じ**（= 代表として選ばれる木が変わらない）。
    rows_distinct = np.flatnonzero(counts == n_nodes)
    if rows_distinct.size:
        flat = np.sort(colors, axis=1).astype(">u8").tobytes()
        stride = n_nodes * 8
        for idx in rows_distinct.tolist():
            keys[idx] = flat[idx * stride : (idx + 1) * stride]
    return keys


def _batch_canonical_keys(
    ctx: RibbonContext,
    mirror_ctx: RibbonContext,
    trees: list[list[tuple[int, int]]],
) -> list[tuple[Any, Any]]:
    """窓内の全木について (自身の正規形, 鏡像の正規形) を返す。

    項目の並びは ``trees`` の順。**全行を計算する参照実装**で、
    :func:`iter_unique_nets` はこの結果のうち必要な行だけを :func:`_context_keys`
    で直接計算する（鏡像の正規形が要る行は「窓で最初に現れる新しいキー」だけ）。

    キーの作り方と同型判定の性質は :func:`_context_keys` の docstring 参照。
    """
    if not trees:
        return []
    edges_arr = _trees_to_array(trees, len(trees[0]))
    return list(
        zip(
            _context_keys(ctx, trees, edges_arr),
            _context_keys(mirror_ctx, trees, edges_arr),
        )
    )


def iter_unique_nets(
    vertices: np.ndarray,
    faces: list[list[int]],
    all_trees: Iterable[list[tuple[int, int]]],
    normals: list[np.ndarray],
) -> Iterator[tuple[list[tuple[int, int]], list[Polygon]]]:
    """一意な展開図を (全域木, 2Dポリゴン) の組として順に返す。

    重複は「正規形（同型なら必ず一致するキー）」で潰す。
    鏡像も同一視するので、自身と鏡像の正規形の小さい方をキーにする。

    シグネ計算は窓単位でバッチ化する（:_context_keys 参照）。
    窓内では木の順序に従って重複判定・展開・重なり判定をするので、
    返り値の順序は木ごとの列挙順と一致する。

    **鏡像の正規形は、要る行だけ計算する**（全体の 1/2 近くがこれで消える）。
    「自身のキー sig が既出 ⟹ その木は既に登録済みのクラス」なので、
    sig が新しい行だけ鏡像を計算すればよい。根拠:

    * 正規形が等しい2つの木は同型（正規形は完全な同型判定。
      ``test_batched_and_per_tree_signatures_agree_on_uniqueness``）。
    * 鏡像のキーは「鏡像文脈で測った正規形」で、鏡像の鏡像は元に戻る。
      よって ``sig(T) == mirror(T')`` なら ``T`` は ``T'`` の鏡像と同型。
    * クラスのキー ``min(sig, mirror)`` は同型類の関数（鏡像は同一視）なので、
      どの経路で見つけても同じキーになり、``exact_seen`` の判定は変わらない。

    ⚠️ このループを ``except ValueError`` で包まないこと。
    そうするとループ中のあらゆる ValueError（予算超過を含む）が
    「この木は捨てる」と解釈され、**静かに誤った一意数**を返す。
    前条件を破った呼び出しは、投げられたまま呼び出し元へ届くのが正しい。
    """
    ctx = RibbonContext(vertices, faces)
    mirror_ctx = RibbonContext(vertices, [list(reversed(f)) for f in faces])
    layout_prep = prepare_layout(vertices, faces, normals)

    exact_seen: set[bytes] = set()
    # 自身/鏡像の正規形として既に見た値。「ここに入っている sig の木は
    # 登録済みクラス」という不変条件を保つ（キーが既出だった場合も含めて
    # 登録時に必ず入れる。クラスのキーは変わらないので過剰な除外にはならない）。
    seen_side: set[bytes] = set()

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
        edges_arr = _trees_to_array(window, len(faces) - 1)
        sigs = _context_keys(ctx, window, edges_arr)

        # 鏡像の正規形が要るのは「窓の中で最初に現れる、まだ見ていない sig」の行。
        # 窓内の重複（同じ sig の2本目以降）はこの時点で落ちる。
        fresh: list[int] = []
        for i, sig in enumerate(sigs):
            if sig in seen_side:
                continue
            seen_side.add(sig)
            fresh.append(i)
        mirrors = _context_keys(mirror_ctx, [window[i] for i in fresh])
        mirror_of = dict(zip(fresh, mirrors))

        # 窓内で一意な木を集めてから、展開をまとめて行う。
        # 1 本ずつ layout_tree を呼ぶと numpy 呼び出しの固定費が支配的なので、
        # 窓単位でベクトル化する（_layout_batch、結果はビット単位で同じ）。
        # 重複判定は木の順序どおりに済ませてから展開するので、返り値の順序は
        # 従来と完全に一致する（重複判定は展開結果に依存しない）。
        pending: list[list[tuple[int, int]]] = []
        for i, tree in enumerate(window):
            mirror_sig = mirror_of.get(i)
            if mirror_sig is None:
                # sig が既出 = 既に登録済みのクラス（上の docstring の根拠）。
                continue
            sig = sigs[i]
            # キーを先に登録してあるので、同型な木は展開せずに弾ける。
            key = min(sig, mirror_sig)
            # キーが既出でも sig / 鏡像は「処理済みクラスの正規形」なので登録する。
            seen_side.add(mirror_sig)
            if key in exact_seen:
                continue
            exact_seen.add(key)
            pending.append(tree)

        if not pending:
            continue
        batch = _layout_batch(len(faces), pending, layout_prep)

        # 重なり判定も窓単位でベクトル化する（外接矩形の絞り込み + GEOS を
        # 呼ばない証明）。判定は is_valid_layout を順に呼んだ場合と常に一致する
        # （証明できた候補は本当に重なっているので、early-exit の結果は変わらない。
        # ただし intersection の**呼び出し回数**は減る）。
        # Polygon は判定・出力に要るネットだけ _valid_batch が作って返す。
        ok_flags, polys_by_net = _valid_batch(
            batch, _trees_to_array(pending, len(faces) - 1), layout_prep
        )
        for tree, polys, ok in zip(pending, polys_by_net, ok_flags):
            if not ok:
                continue
            assert polys is not None
            yield tree, polys
