"""任意の多面体を受け取り、評価指標の上位 n 個の展開図を JSON で返す API。"""

import math
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import fifth

app = FastAPI(title="Polyhedron Net Generator API")

# 開発環境での既定値。公開するときは CORS_ORIGINS にフロントエンドの Origin を指定する。
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "null,http://localhost,http://127.0.0.1,http://localhost:3000,"
        "http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 展開図の候補数は指数的に増えるため、列挙前に全域木の個数を数えて
# 上限を超えていれば計算せずに 422 で返す。
# 値の根拠は README.md「既知の制限」の実測表を参照（決めるのは高速化後の再計測）。
MAX_SPANNING_TREES = 10_000

# 返す展開図の枚数の上限。全域木の数とは別概念なので定数も分ける。
MAX_TOP_N = 10_000

# 422 のレスポンススキーマ。Pydantic のフィールド検証エラーと本 API のエラーは
# どちらも {"type", "loc", "msg"} の配列なので、1つにまとめられる。
# pydantic 側は input / ctx を、本 API 側は spanning_trees / limit を付けるため
# additionalProperties は true。
ERROR_RESPONSE = {
    "description": (
        "入力エラー、または計算量の上限超過（全域木が "
        "MAX_SPANNING_TREES を超える）。detail は常にオブジェクトの配列。"
    ),
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": (
                                        "機械可読な分類。例: non_planar_face, "
                                        "too_many_spanning_trees, missing"
                                    ),
                                },
                                "loc": {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [{"type": "string"}, {"type": "integer"}]
                                    },
                                    "description": "発生箇所",
                                },
                                "msg": {
                                    "type": "string",
                                    "description": "人間向けの説明（表示に回す）",
                                },
                                "spanning_trees": {
                                    "type": "integer",
                                    "description": "type が too_many_spanning_trees のときの全域木数",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "type が too_many_spanning_trees のときの上限",
                                },
                            },
                            "required": ["type", "loc", "msg"],
                            "additionalProperties": True,
                        },
                    }
                },
                "required": ["detail"],
            }
        }
    },
}


class SolidRequest(BaseModel):
    """任意の多面体の定義。

    未知のフィールドは拒否する。
    """

    model_config = ConfigDict(extra="forbid")

    vertices: list[list[float]] = Field(
        ...,
        description="頂点座標の配列 [[x, y, z], ...]",
        examples=[[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]],
    )
    faces: list[list[int]] = Field(
        ...,
        description=(
            "各面の頂点インデックス（3角形/4角形/N角形すべて可）。"
            "巻き順は任意で、外向きに正規化される。"
        ),
        examples=[[[0, 1, 2], [0, 2, 3]]],
    )
    top_n: int = Field(
        1, ge=1, le=MAX_TOP_N, description="返す展開図の数。評価指標の良い順に n 個。"
    )
    sort_by: Literal["area", "perimeter"] = Field("area", description="並べ替えの基準")


def get_layout_parent_relation(
    faces: list[list[int]], tree_edges: list[tuple[int, int]]
) -> list[int | None]:
    """全域木から面の親子関係を求める。面0をルートとする。"""
    n_faces = len(faces)
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    parent_map: list[int | None] = [None] * n_faces
    queue = [0]
    visited = {0}
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                parent_map[v] = u
                visited.add(v)
                queue.append(v)
    return parent_map


def _face_area_3d(vertices: np.ndarray, face: list[int]) -> float:
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
        frozenset((u, v)): edge for u, v, edge in fifth.build_adjacency_with_edges(faces)
    }


def _fold_axis_2d(
    polys: list, face_id: int, parent_id: int
) -> list[list[float]] | None:
    """展開図平面上での折り曲げ軸（2D）を、親面と共有する2頂点から求める。"""
    def key(pt: tuple[float, float]) -> tuple[float, float]:
        return (round(pt[0], 9), round(pt[1], 9))

    parent_pts = {key(c) for c in polys[parent_id].exterior.coords}
    shared = [c for c in polys[face_id].exterior.coords if key(c) in parent_pts]
    if len(shared) < 2:
        return None
    return [[float(shared[0][0]), float(shared[0][1])],
            [float(shared[1][0]), float(shared[1][1])]]


def _face_geometry(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    tree_edges: list[tuple[int, int]],
    polys: list,
    pair_to_edge: dict[frozenset[int], tuple[int, int]],
) -> list[dict]:
    """各面の2D形状・親子関係・折り曲げ軸/角を JSON 化可能な形にする。"""
    parent_map = get_layout_parent_relation(faces, tree_edges)

    out = []
    for face_id, poly in enumerate(polys):
        x, y = poly.exterior.xy
        # shapely は末尾に開始点を重複して持つので取り除く
        coords = [[float(x[i]), float(y[i])] for i in range(len(x) - 1)]

        parent_id = parent_map[face_id]
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
            {
                "face_id": face_id,
                "name": f"面{face_id}",
                "vertices_2d": coords,
                "parent_face_id": parent_id,
                "dihedral_angle": dihedral,
                "fold_angle": fold_angle,
                "fold_axis_3d": axis_3d,
                "fold_axis_2d": axis_2d,
            }
        )
    return out


def _error(
    type_: str,
    msg: str,
    *,
    loc: tuple[str, ...] = ("body",),
    **extra: object,
) -> HTTPException:
    """422 を ``detail`` の配列で返す例外を作る。

    FastAPI のフィールド検証エラー（RequestValidationError）が
    ``detail`` を ``{"type", "loc", "msg"}`` の配列で返すのと同じ形にしておくため、
    422 は常に「オブジェクトの配列」になる。クライアントは detail を
    一律に ``for e in detail:`` で辿れ、``e["type"]`` で機械可読に分岐し、
    ``e["msg"]`` を表示に回せる。

    422 が二つの形に分かれていた頃は、クライアントが本文を読まないと
    「入力ミス」と「計算量の上限超過」を区別できなかった。
    """
    detail: dict[str, object] = {"type": type_, "loc": list(loc), "msg": msg}
    detail.update(extra)
    return HTTPException(status_code=422, detail=[detail])


def _build_solid(req: SolidRequest) -> tuple[np.ndarray, list[list[int]]]:
    """入力を検証し、正規化済みの (vertices, faces) を返す。"""
    try:
        vertices = np.array(req.vertices, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _error("vertices_invalid", f"vertices を解釈できません: {exc}") from exc

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise _error(
            "vertices_shape", "vertices は [[x, y, z], ...] の形式である必要があります。"
        )
    if not np.all(np.isfinite(vertices)):
        raise _error("non_finite_vertex", "vertices に有限でない値があります。")

    faces = [[int(i) for i in f] for f in req.faces]
    try:
        fifth.validate_solid(vertices, faces)
        faces = fifth.orient_faces_outward(vertices, faces)
    except fifth.SolidError as exc:
        raise _error(exc.code, str(exc)) from exc
    except ValueError as exc:
        raise _error("invalid_solid", str(exc)) from exc
    return vertices, faces


@app.post(
    "/api/net/calculate",
    summary="任意多面体の展開図を上位n個返す",
    responses={422: ERROR_RESPONSE},
)
def calculate_nets(req: SolidRequest):
    """任意の多面体を受け取り、重なりのない展開図を評価指標の良い順に n 個返す。"""
    vertices, faces = _build_solid(req)
    n_faces = len(faces)

    try:
        edges = fifth.build_adjacency_graph(faces)
        n_trees = fifth.count_spanning_trees(edges, n_faces)
        if n_trees == 0:
            raise _error(
                "not_connected",
                "面の隣接グラフが連結でないため、展開図を構成できません。",
            )
        if n_trees > MAX_SPANNING_TREES:
            # 入力は正当。応答時間の予算を超えたというだけなので、
            # 入力エラー（type が異なる）と区別できるよう数値も添える。
            raise _error(
                "too_many_spanning_trees",
                (
                    f"この立体の展開図候補は {n_trees:,} 通りで上限 "
                    f"{MAX_SPANNING_TREES:,} を超えています。面の数を減らしてください。"
                ),
                spanning_trees=n_trees,
                limit=MAX_SPANNING_TREES,
            )

        all_trees = fifth.enumerate_spanning_trees(edges, n_faces)
        normals = fifth.compute_normals(vertices, faces)
        unique = list(fifth.iter_unique_nets(vertices, faces, all_trees, normals))
    except HTTPException:
        raise
    except fifth.SolidError as exc:
        raise _error(exc.code, str(exc)) from exc
    except ValueError as exc:
        raise _error("invalid_solid", str(exc)) from exc

    edge_lengths = {
        edge: float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
        for edge in fifth.build_edge_map(faces)
    }
    pair_to_edge = _pair_to_edge(faces)

    def metrics_of(polys: list, tree: list[tuple[int, int]]) -> dict:
        # 全頂点を Python で回ると一意な展開図の数だけ遅くなるので、
        # 外接矩形（bounds）だけから求める。
        boxes = [p.bounds for p in polys]
        bbox_w = max(b[2] for b in boxes) - min(b[0] for b in boxes)
        bbox_h = max(b[3] for b in boxes) - min(b[1] for b in boxes)

        # 外周長 = 全面の周長合計 - 2 * 折り曲げ軸の長さ
        glued = 0.0
        for u, v in tree:
            edge = pair_to_edge.get(frozenset((u, v)))
            if edge is not None:
                glued += edge_lengths[edge]

        return {
            "bounding_box": {"width": float(bbox_w), "height": float(bbox_h)},
            "bounding_box_area": float(bbox_w * bbox_h),
            "perimeter": float(sum(p.length for p in polys) - 2.0 * glued),
        }

    # 指標は軽いので全部つないでもきる。重い JSON 化は上位 n 個だけで行う。
    scored = [(metrics_of(polys, tree), tree, polys) for tree, polys in unique]
    key = (
        (lambda s: s[0]["bounding_box_area"])
        if req.sort_by == "area"
        else (lambda s: s[0]["perimeter"])
    )
    scored.sort(key=key)

    selected = []
    for i, (metrics, tree, polys) in enumerate(scored[: req.top_n]):
        selected.append(
            {
                "net_id": i + 1,
                "metrics": metrics,
                "faces": _face_geometry(
                    vertices, faces, normals, tree, polys, pair_to_edge
                ),
            }
        )

    return {
        "solid": {
            "vertex_count": len(vertices),
            "face_count": int(n_faces),
            "face_vertex_counts": [len(f) for f in faces],
            "bounding_box": {
                "width": float(np.ptp(vertices[:, 0])),
                "height": float(np.ptp(vertices[:, 1])),
                "depth": float(np.ptp(vertices[:, 2])),
            },
            "total_area": float(sum(_face_area_3d(vertices, f) for f in faces)),
        },
        "sort_by": req.sort_by,
        "total_unique_nets": len(scored),
        "returned": len(selected),
        "nets": selected,
    }
