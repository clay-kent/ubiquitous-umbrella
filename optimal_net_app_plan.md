# 直方体展開図生成API 設計プラン

本プロジェクトでは、指定された寸法（幅・高さ・奥行き）の直方体に対して、自己交差（重なり）のないすべての有効な展開図パターンを計算し、各パターンの指標（外接面積、外周長さなど）とともにJSONおよびSTLデータを提供するAPIを構築します。

## 1. 要求仕様

*   **入力パラメータ**:
    *   直方体の3辺長さ: $W$ (Width), $H$ (Height), $D$ (Depth)
*   **出力データ**:
    *   **立体モデル**: 直方体の3Dモデル (STL形式のバイナリまたはテキストデータ)
    *   **展開図リスト**: 重なり（自己交差）のないすべての有効な展開図パターンのJSONリスト。各パターンには以下のデータが含まれます。
        *   **評価指標 (Metrics)**: 外接矩形（バウンディングボックス）の面積、外周長、縦横比
        *   **展開図構成データ (JSON)**:
            *   各面（計6面）のローカル/グローバル頂点座標
            *   面の隣接関係・接続関係のツリー構造
            *   各面の折り曲げ軸と初期折り曲げ角度（3D組み立てシミュレーション用）

---

#### レスポンス JSON Schema:


## 2. 展開図生成アルゴリズムの設計
既存のアルゴリズムを使う
---

## 3. データフォーマット定義

### 3.1 展開図JSONデータ構造例

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "dimensions": {
      "type": "object",
      "properties": {
        "width": { "type": "number" },
        "height": { "type": "number" },
        "depth": { "type": "number" }
      },
      "required": ["width", "height", "depth"]
    },
    "nets": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "net_id": { "type": "integer" },
          "metrics": {
            "type": "object",
            "properties": {
              "bounding_box": {
                "type": "object",
                "properties": {
                  "width": { "type": "number" },
                  "height": { "type": "number" }
                },
                "required": ["width", "height"]
              },
              "bounding_box_area": { "type": "number" },
              "perimeter": { "type": "number" }
            },
            "required": ["bounding_box", "bounding_box_area", "perimeter"]
          },
          "faces": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "face_id": { "type": "integer", "minimum": 0, "maximum": 5 },
                "name": { "type": "string" },
                "vertices_2d": {
                  "type": "array",
                  "minItems": 4,
                  "maxItems": 4,
                  "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": { "type": "number" }
                  }
                },
                "parent_face_id": { "type": ["integer", "null"] },
                "fold_axis": {
                  "type": ["array", "null"],
                  "minItems": 2,
                  "maxItems": 2,
                  "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": { "type": "number" }
                  }
                },
                "fold_angle": { "type": "number" }
              },
              "required": ["face_id", "name", "vertices_2d", "parent_face_id", "fold_axis", "fold_angle"]
            }
          }
        },
        "required": ["net_id", "metrics", "faces"]
      }
    }
  },
  "required": ["dimensions", "nets"]
}
```

---

## 4. APIエンドポイント (FastAPI)

*   `GET /api/net/calculate`
    *   **クエリパラメータ**:
        *   `w` (float): 幅
        *   `h` (float): 高さ
        *   `d` (float): 奥行き
        *   `sort_by` (string, optional): ソート基準 (`area`, `perimeter`)。デフォルトは `area`。将来的に評価指標が増えた場合に備えて任意の文字列とし、想定外の入力の場合はエラーを返す
    *   **レスポンス**: 展開図のJSONデータ一覧。
*   `GET /api/net/stl`
    *   **クエリパラメータ**:
        *   `w` (float), `h` (float), `d` (float)
    *   **レスポンス**: STL形式の3Dモデルファイル (バイナリ/テキスト)。

---

## 5. 技術スタック

*   **言語**: Python 3.9+
*   **フレームワーク**: FastAPI, Uvicorn
*   **幾何計算ライブラリ**:
    *   `numpy` (ベクトル計算用)
    *   `shapely` (ポリゴンの交差・重なり判定用)
    *   `numpy-stl` または `trimesh` (STLファイルの生成用)
