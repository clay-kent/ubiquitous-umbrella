"""任意の多面体を受け取り、評価指標の上位 n 個の展開図を JSON で返す API。"""

import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon

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


def _face_geometry(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: list[np.ndarray],
    tree_edges: list[tuple[int, int]],
    polys: list[Polygon],
    pair_to_edge: dict[frozenset[int], tuple[int, int]],
) -> list[dict[str, object]]:
    """各面の2D形状・親子関係・折り曲げ軸/角を JSON 化可能な形にする。

    幾何の計算そのものは ``fifth.face_geometries`` にあり、ここは
    キー名を付けるだけ（JSON の形は応答仕様なのでこの側の都合で決めてよい）。
    """
    return [
        {
            "face_id": g.face_id,
            "name": f"面{g.face_id}",
            "vertices_2d": g.coords,
            "parent_face_id": g.parent_id,
            "dihedral_angle": g.dihedral_angle,
            "fold_angle": g.fold_angle,
            "fold_axis_3d": g.fold_axis_3d,
            "fold_axis_2d": g.fold_axis_2d,
        }
        for g in fifth.face_geometries(
            vertices, faces, normals, tree_edges, polys, pair_to_edge
        )
    ]


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
        vertices = fifth.parse_vertices(req.vertices)
        faces = [[int(i) for i in f] for f in req.faces]
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
def calculate_nets(req: SolidRequest) -> dict[str, object]:
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
    except fifth.SolidError as exc:
        # 上の _error(...) は HTTPException で、ValueError ではないので
        # 下の except ValueError に飲まれない。再raise句は不要。
        raise _error(exc.code, str(exc)) from exc
    except ValueError as exc:
        raise _error("invalid_solid", str(exc)) from exc

    shared_of = fifth.build_shared_of(faces)
    pair_to_edge = fifth._pair_to_edge(faces)

    def metrics_json(m: fifth.NetMetrics) -> dict[str, object]:
        """NamedTuple はそのままだと JSON の配列になるので、応答の形に戻す。

        指標の**計算**は fifth.net_metrics に一本化し、
        **JSON の形**はここ（応答仕様）側で決める。
        """
        return {
            "bounding_box": {"width": m.bounding_box_width, "height": m.bounding_box_height},
            "bounding_box_area": m.bounding_box_area,
            "perimeter": m.perimeter,
        }

    # 指標の計算は fifth.net_metrics に一本化した。ここは JSON の形にするだけ。
    scored = [
        (
            fifth.net_metrics(polys, tree, vertices, shared_of),
            tree,
            polys,
        )
        for tree, polys in unique
    ]
    key = (
        (lambda s: s[0].bounding_box_area)
        if req.sort_by == "area"
        else (lambda s: s[0].perimeter)
    )
    scored.sort(key=key)

    selected = []
    for i, (metrics, tree, polys) in enumerate(scored[: req.top_n]):
        selected.append(
            {
                "net_id": i + 1,
                "metrics": metrics_json(metrics),
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
            "total_area": float(sum(fifth._face_area_3d(vertices, f) for f in faces)),
        },
        "sort_by": req.sort_by,
        "total_unique_nets": len(scored),
        "returned": len(selected),
        "nets": selected,
    }
