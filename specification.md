# 要求仕様

## 1. api_server の要求仕様

任意の多面体（凸・非凸を問わず）を受け取り、評価指標が上位 n 個の展開図を JSON で返す。

*   **入力 (POST リクエストボディ)**:
    *   `vertices` (必須): 頂点座標の配列 `[[x, y, z], ...]`
    *   `faces` (必須): 各面の頂点インデックスの配列 `[[i, j, k, ...], ...]`。三角形・四角形・N 角形の混在可。頂点インデックスの順序（巻き順）は呼び出し側が意識しなくてよい（サーバ側で外向きに正規化する）。
    *   `top_n` (任意, 既定 1): 返す展開図の数。評価指標の良い順に n 個。
    *   `sort_by` (任意, 既定 `area`): 並べ替えの基準。`area`（外接矩形の面積）または `perimeter`（外周長）。
    *   未知のフィールドは拒否する（422）。typo や旧仕様のフィールドが黙って捨てられないため。
*   **出力 (レスポンス)**:
    *   **展開図リスト**: 重なり（自己交差）のないユニークな展開図のうち、上位 n 個。各パターンには以下のデータが含まれる。
        *   **評価指標 (Metrics)**: 外接矩形（バウンディングボックス）の幅・高さ・面積、外周長
        *   **展開図構成データ**: 各面のローカル 2D 頂点座標、面の親子関係（ツリー構造）、折り曲げ角・折り曲げ軸（3D 組み立てシミュレーション用）
*   **一意性の定義**: 面をノード、共有辺をエッジとした面グラフの全域木を列挙し、その木から作られる半辺構造（リボングラフ）の**正規形（個別化-細分）**が一致するものを「同じ展開図」とみなす。**回転・並進・鏡映はすべて同一視する。** したがって立方体は 11 種であり、112 ではない。
*   **計算量の制限**: 全域木の数は指数的に増えるため、列挙に着手する**前**に Matrix-Tree 定理で全域木の数を数え、上限 `MAX_SPANNING_TREES`（10,000）を超えていれば計算せずに 422 を返す。
    *   これは「応答時間の予算」であって入力の正しさとは無関係。1枚あたり約 0.3ms で処理できるため 10,000 本 ≒ 3秒に相当する。**上限に達しても入力は正当**（面数不足・非平面・境界・非連結といった入力エラーとは区別する）。
    *   上限と実測の対応は README「既知の制限」に記載する。
    *   返す枚数の上限 `MAX_TOP_N` は全域木の数とは別の定数である。

> 展開図の全列挙は第五章のアルゴリズムによる（既存プログラム `fifth.py` をライブラリとして呼び出す）。

## 2. 展開図生成アルゴリズムの設計

既存のプログラム `fifth.py` をライブラリとして呼び出すため省略。

## 3. データ形式の定義

既存のプログラム `fifth.py` に実装済みのため省略。

要約すると、面の形状をグラフとして表現する。

そのグラフをノード、辺をエッジとしたグラフで立体を表現する。

### 3.1 グラフのデータ構造

頂点 `vertices` と面 `faces` の対から以下を構築する。

*   `build_edge_map(faces)` — 辺ごとの所有面。ちょうど 2 面に共有される辺のみが内部棱となり、それ以外（1 面のみ・3 面以上）は 422 とする。
*   `build_adjacency_graph(faces)` — 面グラフの辺リスト `[(u, v), ...]`。共有辺 1 本につき 1 エントリ。
*   `orient_faces_outward(vertices, faces)` — **parity union-find** で「各辺を逆向きに 2 面が通る」制約を解き、整合する面だけ裏返す。その後、符号付き体積が負なら全体を裏返して外向きにする。呼び出し側の巻き順は問わない。整合化できない（非可向き）場合は 422。
    *   面の巻き順が正しければ何もしない。混在していれば矛盾する面だけを裏返す。
    *   **前条件**: `validate_solid` を先に通していること。辺がちょうど 2 面に共有される保証はそこに閉じている。前条件を破った入力はタプル展開が `ValueError` になる（黙って無視しない）。

### 3.2 リクエストスキーマ

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "additionalProperties": false,
  "required": ["vertices", "faces"],
  "properties": {
    "vertices": {
      "type": "array",
      "minItems": 4,
      "items": {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": { "type": "number" }
      }
    },
    "faces": {
      "type": "array",
      "minItems": 4,
      "items": {
        "type": "array",
        "minItems": 3,
        "items": { "type": "integer", "minimum": 0 }
      }
    },
    "top_n": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10000,
      "default": 1
    },
    "sort_by": {
      "type": "string",
      "enum": ["area", "perimeter"],
      "default": "area"
    }
  }
}
```

`faces` の各面は頂点数 3 以上。面ごとに三角形・四角形・N 角形が混在してよい。

### 3.3 レスポンススキーマ

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["solid", "sort_by", "total_unique_nets", "returned", "nets"],
  "properties": {
    "solid": {
      "type": "object",
      "required": ["vertex_count", "face_count", "face_vertex_counts", "bounding_box", "total_area"],
      "properties": {
        "vertex_count": { "type": "integer" },
        "face_count": { "type": "integer" },
        "face_vertex_counts": {
          "type": "array",
          "items": { "type": "integer", "minimum": 3 }
        },
        "bounding_box": {
          "type": "object",
          "required": ["width", "height", "depth"],
          "properties": {
            "width": { "type": "number" },
            "height": { "type": "number" },
            "depth": { "type": "number" }
          }
        },
        "total_area": { "type": "number" }
      }
    },
    "sort_by": { "type": "string", "enum": ["area", "perimeter"] },
    "total_unique_nets": { "type": "integer" },
    "returned": { "type": "integer" },
    "nets": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["net_id", "metrics", "faces"],
        "properties": {
          "net_id": { "type": "integer", "minimum": 1 },
          "metrics": {
            "type": "object",
            "required": ["bounding_box", "bounding_box_area", "perimeter"],
            "properties": {
              "bounding_box": {
                "type": "object",
                "required": ["width", "height"],
                "properties": {
                  "width": { "type": "number" },
                  "height": { "type": "number" }
                }
              },
              "bounding_box_area": { "type": "number" },
              "perimeter": { "type": "number" }
            }
          },
          "faces": {
            "type": "array",
            "items": {
              "type": "object",
              "required": [
                "face_id", "name", "vertices_2d", "parent_face_id",
                "dihedral_angle", "fold_angle", "fold_axis_3d", "fold_axis_2d"
              ],
              "properties": {
                "face_id": {
                  "type": "integer",
                  "minimum": 0,
                  "description": "入力の faces 配列におけるインデックス"
                },
                "name": { "type": "string" },
                "vertices_2d": {
                  "type": "array",
                  "minItems": 3,
                  "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": { "type": "number" }
                  },
                  "description": "展開図上の 2D 座標。頂点数はその面の頂点数と一致する"
                },
                "parent_face_id": {
                  "type": ["integer", "null"],
                  "description": "展開図の木構造における親の面。根は null"
                },
                "dihedral_angle": {
                  "type": ["number", "null"],
                  "description": "親の面との内側の二面角（度）。凸角は 0〜180、凹角は 180〜360。根は null"
                },
                "fold_angle": {
                  "type": ["number", "null"],
                  "description": "平らな展開図を立体に畳み込むのに必要な回転角 = 180 − dihedral_angle。凹角では負になる。根は null"
                },
                "fold_axis_3d": {
                  "type": ["array", "null"],
                  "minItems": 2,
                  "maxItems": 2,
                  "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": { "type": "number" }
                  },
                  "description": "立体上での折り曲げ軸（共有辺の 3D 座標 2 点）。根は null"
                },
                "fold_axis_2d": {
                  "type": ["array", "null"],
                  "minItems": 2,
                  "maxItems": 2,
                  "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": { "type": "number" }
                  },
                  "description": "展開図上の折り曲げ軸（この面の vertices_2d に含まれる 2 点）。根は null"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 3.4 二面角の求め方

外向き法線 `n_u`, `n_v` だけから `180 − 角度(n_u, n_v)` とすると、L 字柱のような**非凸**立体の凹角を 90 度と誤って求めてしまう。正しくは、棱に垂直な断面で見たときの「立体の内側が張る角」を求める。

1.  面 `u` の周回順に沿った棱方向 `t` を得る。
2.  `in_u = n_u × t`, `in_v = −n_v × t`（面 `v` では棱を逆向きに歩く）。どちらも「その面の内側（立体側）」を指す。
3.  `phi = 角度(in_u, in_v)`（0〜180 度）。
4.  ベクトル `in_u + in_v` が両法線の負の側（立体の内側）を向いていれば内角は `phi`、外側（凹角）を向いていれば `360 − phi`。

検算例: 立方体 = 12 棱すべて 90 度、正四面体 = 6 棱すべて `arccos(1/3) ≒ 70.53` 度、L 字柱 = 17 棱 90 度 + 凹角 1 棱 270 度。

---

## 4. API エンドポイント (FastAPI)

**POST のみ。GET は 405、以前の `GET /api/net/stl` は 404 となる。**

*   `POST /api/net/calculate`
    *   **リクエストボディ**: 3.2 のリクエストスキーマに従う。
        *   `vertices` (必須, `number[][]`): 頂点座標 `[[x, y, z], ...]`
        *   `faces` (必須, `integer[][]`): 各面の頂点インデックス。巻き順は任意で外向きに正規化される
        *   `top_n` (optional, `integer`, 1〜`MAX_TOP_N`, 既定 1): 返す展開図の数
        *   `sort_by` (optional, `string`, `area` / `perimeter`, 既定 `area`): 並べ替えの基準
    *   **レスポンス**: 3.3 のレスポンススキーマに従う展開図の JSON。
    *   **エラー**:
        *   `405` — GET など POST 以外のメソッド
        *   `422` — (a) 入力エラー: 形状不正（面数不足・頂点範囲外・非平面・境界あり・非連結・頂点の重複・有限でない座標）、`top_n` / `sort_by` の範囲外、未知のフィールド
        *   `422` — (b) 計算量の上限超過: 全域木が `MAX_SPANNING_TREES` を超える。**入力は正当**。詳細は 4.2

### 4.1 422 となる入力の例

`type` は 4.2 で定義する機械可読な分類。

| 入力 | `type` | 理由 |
| --- | --- | --- |
| 頂点が 4 未満 | `too_few_vertices` | 閉じた多面体を作れない |
| 面が 4 未満 | `too_few_faces` | 閉じた多面体を作れない |
| 頂点インデックスが `vertices` の長さを超える | `vertex_index_out_of_range` | 参照範囲外 |
| 同じ面に同じ頂点が 2 回以上現れる | `duplicate_face_vertex` | エッジにならない |
| **異なる頂点が同じ座標を持つ** | `duplicate_vertex` | 面はインデックスで頂点を指すので、同じ点に2枚の状態が被り、隣接グラフが定義と食い違う |
| 1つの面の頂点数が 3 未満 | `face_too_few_vertices` | 面が定義できない（多角形として閉じていない） |
| 面の中に長さ 0 の辺がある | `zero_length_edge` | 法線が定義できない |
| 面の頂点が同一平面上にない | `non_planar_face` | 面が定義できない |
| ある辺を 1 面しか共有しない | `boundary_or_non_manifold` | 表面が閉じておらず（境界がある）、展開図を構成できない |
| ある辺を 3 面以上が共有する | `boundary_or_non_manifold` | 非多様体 |
| 頂点を含む面が 3 未満 | `vertex_underused` | 頂点まわりの面が1つのサイクルにならない |
| 頂点まわりの面が連結でない | `non_manifold_vertex` | 面の飛び（非多様体） |
| 面の面積が 0（退化） | `degenerate_face` | 法線が定義できない |
| 面の向きを外向きに揃えられない | `non_orientable` | 非可向きの立体 |
| 面の連結成分が 2 つ以上ある | `not_connected` | 離れた立体は 1 枚の展開図にできない |
| 座標に NaN / Infinity | `non_finite_vertex` | 幾何計算が成立しない |
| `vertices` が `[[x, y, z], ...]` の形式でない（幅が3でない矩形配列など） | `vertices_shape` | 頂点として解釈できない |
| `vertices` が数値の配列として解釈できない（要素数が揃っていないなど） | `vertices_invalid` | 座標の型が不正 |
| 上のどれにも当てはまらない `ValueError` | `invalid_solid` | 最終受け皿。具体的な到達経路は未確認（v3 PR4/5 の削除後にほぼ到達しない） |
| 同型判定が計算量の上限（`SEARCH_BUDGET`）を超えた | `search_budget_exceeded` | **入力は正当**。面の数が多すぎるため計算を打ち切る（4.2） |
| 全域木が 10,000 本超 | `too_many_spanning_trees` | 列挙前に上限を検出して 422（例: 7 角柱）。**入力は正当**（4.2） |
| `top_n` / `sort_by` が範囲外 | Pydantic の型名（例 `greater_than_equal`） | 仕様違反 |
| 未知のフィールド名 | `extra_forbidden` | typo や旧仕様フィールドを黙って捨てないため |

**凹角（非凸）は 422 にしない。** L 字柱や貫通穴つきの立体（genus > 0）は正当な入力として受理する。

**NaN / Infinity は JSON として不正なので、`Content-Type: application/json` で送ると通信层面上弾かれる。** `non_finite_vertex` を確認したい場合は、ヘッダを `text/plain` に変えて本文だけを送ること（`httpx2` では `content=` で送る）。JSON として妥当な値を送った場合は、この型は出ない。

### 4.2 エラーの構造

`422` の `detail` は**常にオブジェクトの配列**である（本 API のエラーも FastAPI/Pydantic のフィールド検証エラーも同じ形）。

```json
{
  "detail": [
    {
      "type": "too_many_spanning_trees",
      "loc": ["body"],
      "msg": "この立体の展開図候補は 35,287 通りで上限 10,000 を超えています。面の数を減らしてください。",
      "spanning_trees": 35287,
      "limit": 10000
    }
  ]
}
```

| キー | 意味 |
| --- | --- |
| `type` | 機械可読な分類。クライアントはこれで分岐する |
| `loc` | 発生箇所。本 API のエラーはすべて `["body"]`。フィールド検証エラーは `["body", "top_n"]` のようにフィールド名まで入る |
| `msg` | 人間向けの説明（日本語）。表示に回す |
| `spanning_trees` / `limit` | `type` が `too_many_spanning_trees` のときだけ付く**整数** |

*   **クライアントは 422 の `detail` を一律に `for e in detail:` で処理できる。** `e["type"]` で分岐し、`e["msg"]` を表示に回す。
*   **`type == "too_many_spanning_trees"` と `search_budget_exceeded` は入力が正当**であるということ。面の数を減らせば成功する。それ以外の `type` は入力の修正が必要。
*   **頂点の重複（`duplicate_vertex`）と面の非連結（`not_connected`）は別物**。前者は座標を1つにまとめる修正が必要で、面を削る方法では直らない。
*   `detail` が文字列（本 API）と配列（Pydantic）に分かれていた頃は、本文を読まないと両者を区別できなかった。統一したのはこのため。

---

## 5. 技術スタック

*   **言語**: Python 3.13
*   **フレームワーク**: FastAPI, Uvicorn
*   **幾何計算ライブラリ**:
    *   `numpy` (ベクトル計算・法線・法線行列)
    *   `shapely` (ポリゴンの交差・重なり判定)
    *   `networkx` は本体ではなく**テスト専用**（全域木列挙の参照実装 `_brute_force_trees` のみ）
*   **テスト**: `pytest`, `httpx2`（`fastapi.testclient.TestClient` の実装に必要）
