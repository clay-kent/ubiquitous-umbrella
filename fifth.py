import io
import math
import sys
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import networkx as nx
import numpy as np
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

    scale = float(np.max(np.abs(vertices))) or 1.0
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

        def remap(x: int) -> int:
            if x == pu or x == pv:
                return merged
            # pu・pv を除いた中の順位に詰め直す（pu と pv の大小は問わない）
            return x - (pu < x) - (pv < x)

        contracted = [(i, remap(a), remap(b)) for i, a, b in rest]
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
    非可向きの立体は ValueError。
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
        if len(holders) != 2:
            continue
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


def prepare_layout(
    vertices: np.ndarray, faces: list[list[int]], normals: list[np.ndarray]
) -> dict:
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
        # 全面の頂点数が同じなら、まとめて Polygon を作れる
        "uniform_ring": len({len(f) for f in faces}) == 1,
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
    precomputed: dict | None = None,
) -> list[Polygon]:
    """全域木に基づいて面を展開し、ShapelyのPolygonリスト（2D座標系）を返す。

    平行移動・回転は 4x4 行列ではなく (回転行列 R, 移動ベクトル t) で持ち、
    最終的な 2D への投影もまとめてベクトル演算する。
    全域木の数だけ呼び出されるため、ここは定数倍でも効く。
    """
    if precomputed is None:
        precomputed = prepare_layout(vertices, faces, normals)
    shared_of = precomputed["shared_of"]
    n_faces = len(faces)

    _parents, pairs = _tree_bfs(n_faces, tree_edges)

    # 親が子より先に現れる幅優先順で、回転と平行移動を伝播させる。
    # placed は到達した面だけ True にする（rot の型を Optional にしないため）。
    eye = np.eye(3)
    zero = np.zeros(3)
    rot = [eye] * n_faces
    off = [zero] * n_faces
    placed = [False] * n_faces
    placed[0] = True

    for u, v in pairs:
        # tree_edges は build_adjacency_graph 由来の全域木なので共有辺は必ず存在する。
        # そうでない呼び出し（利用者が前条件を破った場合）はここで KeyError になる。
        edge = shared_of[(u, v)]

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

        rot[v] = rot[u] @ r
        off[v] = rot[u] @ (p_a - r @ p_a) + off[u]
        placed[v] = True

    origin = precomputed["origin"]
    u_x = precomputed["u_x"]
    u_y = precomputed["u_y"]
    face_pts = precomputed["face_pts"]

    coords = []
    for i in range(n_faces):
        if not placed[i]:
            raise ValueError(f"面{i} へ到達できませんでした（全域木が不正です）。")
        face_rot, t = rot[i], off[i]
        # p_trans = r @ p + t を、2D 投影に代入してまとめて計算する
        axis_x = face_rot.T @ u_x
        axis_y = face_rot.T @ u_y
        shift = t - origin
        pts = face_pts[i]
        coords_x = pts @ axis_x + float(shift @ u_x)
        coords_y = pts @ axis_y + float(shift @ u_y)
        coords.append(np.column_stack([coords_x, coords_y]))

    if precomputed["uniform_ring"]:
        # 頂点数が揃っていればまとめて作ると Polygon() を1つずつ呼ぶより速い
        rings = shapely.polygons(shapely.linearrings(np.stack(coords)))
        if isinstance(rings, Polygon):
            return [rings]
        return list(rings)
    return [Polygon(c) for c in coords]


def is_valid_layout(polygons) -> bool:
    """展開図のポリゴン間に重なり（衝突）がないかを検証する。

    外面の数が少ないので、空間索引（STRtree）よりも外接矩形で絞り込んだほうが
    intersection を呼ぶ回数が減って速い。
    """
    n_polys = len(polygons)
    bounds = [p.bounds for p in polygons]
    for i in range(n_polys):
        ax0, ay0, ax1, ay1 = bounds[i]
        for j in range(i + 1, n_polys):
            bx0, by0, bx1, by1 = bounds[j]
            # 外接矩形が重ならなければ交差面積も 0 なので交差計算を省略できる
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            # 閾値を面積の絶対値にすると、立体が小さいと実重なりを見落とす。
            # 2つの面の和集合の外接矩形の対角の長さ^2 を基準にした比で判定する。
            # 和集合を使うのは「接しているだけ」のとき交差側の幅が0になり、
            # 閾値も0になって浮動ノイズを実重なりと誤判定するため。
            dx = max(ax1, bx1) - min(ax0, bx0)
            dy = max(ay1, by1) - min(ay0, by0)
            if (
                polygons[i].intersection(polygons[j]).area
                > OVERLAP_AREA_REL_TOL * (dx * dx + dy * dy)
            ):
                return False
    return True


class NetMetrics(NamedTuple):
    """展開図 1 枚分の評価指標。"""

    bounding_box_width: float
    bounding_box_height: float
    bounding_box_area: float
    perimeter: float


def net_metrics(
    polys: list[Polygon],
    tree_edges: list[tuple[int, int]],
    vertices: np.ndarray,
    shared_of: dict[tuple[int, int], tuple[int, int]],
) -> NetMetrics:
    """展開図の評価指標を求める。

    **外周長 = 全面の周長合計 − 2 × 折り曲げ軸の長さ合計**。

    展開図の外周は「折ったときに消える内側の棱」だけ短い。
    したがって外接矩形の周長 ``2*(w+h)`` は外周長ではない
    （L 字型の展開図では外接矩形の空这部分だけ周を含む）。

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
        edge = shared_of.get((u, v))
        if edge is not None:
            a, b = edge
            glued += float(np.linalg.norm(vertices[a] - vertices[b]))

    return NetMetrics(
        bounding_box_width=float(bbox_w),
        bounding_box_height=float(bbox_h),
        bounding_box_area=float(bbox_w * bbox_h),
        perimeter=float(sum(p.length for p in polys) - 2.0 * glued),
    )


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
    norm_t = float(np.linalg.norm(t))
    if norm_t == 0.0:
        return 0.0
    t = t / norm_t

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
            edge = pair_to_edge.get(frozenset((face_id, parent_id)))
            if edge is not None:
                a, b = edge
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


def build_ribbon_graph(
    vertices: np.ndarray,
    faces: list[list[int]],
    tree_edges: list[tuple[int, int]],
) -> nx.DiGraph:
    """展開図の木構造から半辺グラフ (Ribbon Graph) を構築する。

    ノード: 各面の各辺に対応する半辺 (face_index, edge_index)
      - 属性 `length`: 辺の長さ (丸め処理)
    エッジ:
      - 'next' 型の有向エッジ: 同一面上での巡回順序 (CCW)
      - 'glue' 型の無向/双方向有向エッジ: 隣接面間の接着
        - 属性 `flipped`: 接着される向き
    """
    g: nx.DiGraph = nx.DiGraph()

    # 1. ハーフエッジのノードを作成し、面内の「next」エッジで巡回的に結ぶ
    for f_idx, face in enumerate(faces):
        n_v = len(face)
        for i in range(n_v):
            p1 = vertices[face[i]]
            p2 = vertices[face[(i + 1) % n_v]]
            length = np.linalg.norm(p2 - p1)

            node_id = (f_idx, i)
            g.add_node(node_id, length=round(length, 5))

        # nextエッジを追加 (面をCCWに一周するサイクル)
        for i in range(n_v):
            g.add_edge((f_idx, i), (f_idx, (i + 1) % n_v), type="next")

    # 2. 面間の接着 (glue) エッジを追加
    for u, v in tree_edges:
        # 面uと面vの共有頂点を3Dで探す
        shared = set(faces[u]) & set(faces[v])
        if len(shared) != 2:
            continue

        # 面uにおける共有辺のインデックスを探す
        idx_u = None
        n_u = len(faces[u])
        for i in range(n_u):
            a, b = faces[u][i], faces[u][(i + 1) % n_u]
            if a in shared and b in shared:
                idx_u = i
                break

        # 面vにおける共有辺のインデックスを探す
        idx_v = None
        n_v = len(faces[v])
        for i in range(n_v):
            a, b = faces[v][i], faces[v][(i + 1) % n_v]
            if a in shared and b in shared:
                idx_v = i
                break

        if idx_u is not None and idx_v is not None:
            # glueエッジを双方向で接続
            u_start, u_end = faces[u][idx_u], faces[u][(idx_u + 1) % n_u]
            v_start, v_end = faces[v][idx_v], faces[v][(idx_v + 1) % n_v]

            flipped = (u_start == v_end) and (u_end == v_start)

            g.add_edge((u, idx_u), (v, idx_v), type="glue", flipped=flipped)
            g.add_edge((v, idx_v), (u, idx_u), type="glue", flipped=flipped)

    return g


def is_isomorphic_net(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
    """2つの半辺グラフが同型であるかを判定する。"""
    def node_match(n1, n2):
        # build_ribbon_graph は round(length, 5) で丸めた値を格納している。
        # _ribbon_invariant も同じ丸めを使うので、両者を完全一致で比較することで
        # 不変量だけが別になるケースを防げるよう、両者の丸めを揃える。
        return n1["length"] == n2["length"]

    def edge_match(e1, e2):
        # エッジのタイプ (next or glue) と flipped 属性が一致するかチェック
        if e1["type"] != e2["type"]:
            return False
        if e1["type"] == "glue":
            return e1.get("flipped") == e2.get("flipped")
        return True

    GM = nx.algorithms.isomorphism.DiGraphMatcher(
        g1, g2, node_match=node_match, edge_match=edge_match
    )
    return GM.is_isomorphic()


class RibbonContext:
    """半辺構造の前計算結果。全域木ごとに変わらないので一度だけ作る。"""

    __slots__ = (
        "edge_len",
        "faces",
        "idx_map",
        "n_faces",
        "n_nodes",
        "node_id",
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

    glue は build_ribbon_graph が双方向に張るので、is_isomorphic_net から見れば
    無向辺になっている。ここでは向きを区別せず「接着相手」として扱う。
    """
    faces = ctx.faces
    n_nodes = ctx.n_nodes
    partner = [-1] * n_nodes
    flip = [False] * n_nodes

    for u, v in tree_edges:
        edge = ctx.shared.get((u, v))
        if edge is None:
            continue
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


def _wl_refine(nxt: list[int], partner: list[int], flip: list[bool], color: list) -> list:
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
    color: list,
    budget: list[int],
) -> tuple | None:
    """個別化-細分による正規形。予算超過なら None（呼び出し側は VF2 にフォールバック）。"""
    n = len(nxt)
    if budget[0] > 200_000:
        return None
    budget[0] += n

    color = _wl_refine(nxt, partner, flip, color)
    if len(set(color)) == n:
        # 全ての半辺が別々の色になった = 色列そのものが完全な同型判定になる
        return tuple(sorted(color))

    cells: dict = {}
    for i, c in enumerate(color):
        cells.setdefault(c, []).append(i)
    multi = {c: members for c, members in cells.items() if len(members) > 1}
    if not multi:
        return tuple(sorted(color))

    # 色の値が最小のセルを1つ選んで個別化する（この選択は同型で変わらない）
    target = min(multi)
    best: tuple | None = None
    for v in multi[target]:
        branch = list(color)
        branch[v] = min(color) - 1  # 既存の色と衝突しない新しい色
        got = _individualize_refine(nxt, partner, flip, branch, budget)
        if got is None:
            return None
        if best is None or got < best:
            best = got
    return best


def _ribbon_invariant(ctx: RibbonContext, tree_edges: list[tuple[int, int]]) -> tuple:
    """半辺構造の同型不変量（Weisfeiler-Lehman 色の細分のみ）。

    同型な半辺グラフは必ず同じ不変量を持つ（分離は起きない）。
    ただし別々の展開図が同じ不変量になることもあるため、
    この値だけで同一とは判定せず is_isomorphic_net を併用する。
    """
    partner, flip = _glue_arrays(ctx, tree_edges)
    color = [(ctx.edge_len[i], partner[i] >= 0, flip[i]) for i in range(ctx.n_nodes)]
    return tuple(sorted(_wl_refine(ctx.nxt, partner, flip, color)))


def _net_signature(
    ctx: RibbonContext, tree_edges: list[tuple[int, int]]
) -> tuple | None:
    """展開図の正規形（同型なら必ず一致するキー）。

    個別化-細分が全ての半辺を区別できた場合、その色は完全な同型判定になる。
    予算超過で失敗したときだけ None を返し、VF2 に任せる。
    """
    partner, flip = _glue_arrays(ctx, tree_edges)
    color = [(ctx.edge_len[i], partner[i] >= 0, flip[i]) for i in range(ctx.n_nodes)]
    return _individualize_refine(ctx.nxt, partner, flip, color, [0])



def reverse_ribbon_graph(g: nx.DiGraph) -> nx.DiGraph:
    """ハーフエッジグラフの面内有向エッジ(next)をすべて反転させた、鏡像（裏返し）グラフを生成する。"""
    rev_g: nx.DiGraph = nx.DiGraph()
    for node, data in g.nodes(data=True):
        rev_g.add_node(node, **data)
    for u, v, data in g.edges(data=True):
        if data["type"] == "next":
            rev_g.add_edge(v, u, **data)
        else:
            rev_g.add_edge(u, v, **data)
    return rev_g


def iter_unique_nets(
    vertices: np.ndarray,
    faces: list[list[int]],
    all_trees: Iterable[list[tuple[int, int]]],
    normals: list[np.ndarray],
) -> Iterator[tuple[list[tuple[int, int]], list[Polygon]]]:
    """一意な展開図を (全域木, 2Dポリゴン) の組として順に返す。

    2段構えで重複を潰す:

    1. _net_signature（個別化-細分の正規形）だけで一意に決まるものはハッシュで潰す。
       正規形は同型なら必ず一致し、通常の展開図ではこれだけで完毕する。
    2. 正規形を予算超過で出せなかった場合だけ、Weisfeiler-Lehman 不変量で
       バケット分割し、その中でのみ実際の VF2 同型判定（is_isomorphic_net）を行う。

    どちらの段も「同型なら必ず同じバケット」という保証しか持たないため、
    最終的な正しさは is_isomorphic_net に担保させている。
    """
    ctx = RibbonContext(vertices, faces)
    mirror_ctx = RibbonContext(vertices, [list(reversed(f)) for f in faces])
    layout_prep = prepare_layout(vertices, faces, normals)

    exact_seen: set[tuple] = set()
    buckets: dict[tuple, list[tuple]] = {}
    graph_cache: dict[tuple, tuple[nx.DiGraph, nx.DiGraph]] = {}

    def graph_pair(tree: list[tuple[int, int]]) -> tuple[nx.DiGraph, nx.DiGraph]:
        """半辺グラフとその鏡像を作る（同一木の再計算を避けるためキャッシュ）。"""
        key = tuple(tree)
        cached = graph_cache.get(key)
        if cached is None:
            rg = build_ribbon_graph(vertices, faces, tree)
            cached = (rg, reverse_ribbon_graph(rg))
            graph_cache[key] = cached
        return cached

    for tree in all_trees:
        try:
            # レイアウトより先に重複を判定する。
            # 全域木の多くは別の木に既に含まれる重複なので、ここで弾けるほど
            # 展開と重なり判定の回数が減る（実測で全体の約7割を占めていた）。
            # 同じ正規形の木は展開図が合同であり重なり判定の結果まで一致する
            # （test_unique_nets_geometry.py で独立に検証済み）ため、順序を入れ替えても
            # 返り値は変わらない。
            sig = _net_signature(ctx, tree)
            mirror_sig = _net_signature(mirror_ctx, tree)

            if sig is not None and mirror_sig is not None:
                # 鏡像は同一視するので、自身と鏡像の正規形の小さい方をキーにする。
                # キーを先に登録してあるので、同型な木は展開せずに弾ける。
                key = min(sig, mirror_sig)
                if key in exact_seen:
                    continue
                exact_seen.add(key)
                polys = layout_tree(vertices, faces, normals, tree, layout_prep)
                if not is_valid_layout(polys):
                    continue
                yield tree, polys
                continue

            inv = _ribbon_invariant(ctx, tree)
            mirror_inv = _ribbon_invariant(mirror_ctx, tree)
            bucket = buckets.setdefault(min(inv, mirror_inv), [])

            if bucket:
                rg, _ = graph_pair(tree)
                is_duplicate = any(
                    is_isomorphic_net(rg, other) or is_isomorphic_net(rg, other_rev)
                    for other, other_rev in (graph_pair(list(t)) for t in bucket)
                )
                if is_duplicate:
                    continue

            polys = layout_tree(vertices, faces, normals, tree, layout_prep)
            if not is_valid_layout(polys):
                continue
            bucket.append(tuple(tree))
            yield tree, polys
        except ValueError:
            # 展開できない木（退化した面など）は静かに捨てる
            continue


def get_unique_nets_by_ribbon_graph(
    vertices: np.ndarray,
    faces: list[list[int]],
    all_trees: Iterable[list[tuple[int, int]]],
    normals: list[np.ndarray],
) -> list[list[tuple[int, int]]]:
    """一意な展開図に対応する全域木を抽出する（iter_unique_nets の簡易ラッパ）。"""
    return [tree for tree, _ in iter_unique_nets(vertices, faces, all_trees, normals)]


def generate_unfolding_nets(
    vertices: np.ndarray | list[list[float]], faces: list[list[int]]
) -> list[list[Polygon]]:
    """半辺グラフ同型判定を用いて一意な展開図を生成する。"""
    vertices = np.array(vertices, dtype=float)
    n_faces = len(faces)

    if n_faces < 4:
        raise SolidError("too_few_faces", "多面体は少なくとも4つ以上の面を持つ必要があります。")

    edges = build_adjacency_graph(faces)
    all_trees = enumerate_spanning_trees(edges, n_faces)
    normals = compute_normals(vertices, faces)

    return [polys for _, polys in iter_unique_nets(vertices, faces, all_trees, normals)]


if __name__ == "__main__":
    # 出力をUTF-8に変更 (Windows/CP932環境での文字化け対策)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    # --- 1. 立方体 ---
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    cube_faces = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 上面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 後面
        [3, 0, 4, 7],  # 左面
    ]

    # --- 2. 正四面体 ---
    tetra_vertices = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=float,
    )
    tetra_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]

    # --- 3. 四角錐 ---
    pyramid_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=float,
    )
    pyramid_faces = [
        [0, 3, 2, 1],  # 底面
        [0, 1, 4],  # 側面1
        [1, 2, 4],  # 側面2
        [2, 3, 4],  # 側面3
        [3, 0, 4],  # 側面4
    ]

    shapes = {
        "立方体": (cube_vertices, cube_faces),
        "正四面体": (tetra_vertices, tetra_faces),
        "四角錐": (pyramid_vertices, pyramid_faces),
    }

    for name, (verts, fcs) in shapes.items():
        nets = generate_unfolding_nets(verts, fcs)
        print(f"--- {name} ---")
        print(f"面の数: {len(fcs)}")
        print(f"一意な展開図の数: {len(nets)}\n")
