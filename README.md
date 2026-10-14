任意の多面体（凸・非凸・貫通穴つきなど）から、重なりのないユニークな展開図を列挙・計算・視覚化し、API サーバーの JSON 配信や PNG 画像保存を行う Python スクリプト群。

## 主な機能
- **任意多面体に対応**: 直方体だけでなく、四面体・角柱・角錐・L 字柱（凹角あり）・貫通穴つき（genus=1）などの頂点と面のリストを受け付けます。
- **展開図（全域木）の列挙と一意性判定**: 面をグラフノードとし、隣接関係から全域木を列挙。半辺構造の正規形（個別化-細分）で重複を潰し、ユニークな展開図のみを抽出します。**回転・並進・鏡映は同一視**するため、立方体は 11 種です（112 ではありません）。
- **2D 自動レイアウト**: 各面の 3D 幾何構造を展開し、Shapely を使用して 2D ポリゴン座標に変換。自己交差（重なり）のない正しい展開図のみを返します。
- **折り組み立て情報の出力**: 面ごとの 2D 座標、親子関係、内側の二面角、折り曲げ角（= 180 − 二面角）、3D / 2D の折り曲げ軸。
- **画像出力**: 展開図の 2D レイアウトをプロットし、PNG ファイルとしてローカルディレクトリに保存します。
- **FastAPI による Web API サーバー**: `POST /api/net/calculate` 1 本のみ。評価指標の上位 n 個を JSON で返します。
- **ユニット/統合テスト**: pytest による網羅的なテスト。

## ディレクトリ構造

```text
├── fifth.py                     # 展開図作成のコア。任意の立体の展開図を作れる
├── api_server.py                # API サーバー (POST /api/net/calculate)
├── make_nets.py                 # 任意の立体の展開図を計算し、PNG 画像を生成するお試し用スクリプト
├── test_fifth.py                # コア部分のユニットテスト
├── test_api_server.py           # API のユニット・結合テスト
├── test_unique_nets_geometry.py # 一意性の判定を平面図形の合同で独立に検算するテスト
├── specification.md             # 要求仕様・JSON スキーマ
├── pyproject.toml               # 依存関係定義
├── requirements.txt             # pip 向けの依存関係一覧
├── uv.lock                      # uv のロックファイル
└── README.md                    # 本ドキュメント
```

---

## セットアップ

### 必須環境
- Python 3.13 以上

### 依存パッケージのインストール
本プロジェクトは `fastapi`, `uvicorn`, `shapely`, `numpy`, `matplotlib`, `pytest`, `httpx2` を使用します。`networkx` はテスト専用（全域木列挙の参照実装のみ）です。

#### uv を使用する場合:
```powershell
uv sync
```

#### pip を使用する場合:
```powershell
pip install fastapi uvicorn shapely numpy matplotlib pytest httpx
```

---

## 使い方

### 展開図画像の生成
プリセットした立体の一意な展開図をすべて計算し、PNG 画像として保存します。

```powershell
python make_nets.py                              # 既定の直方体 (1.0 x 1.5 x 2.0)
python make_nets.py --solid prism5               # 5 角柱
python make_nets.py --solid lshape --top-n 6 --sort-by perimeter
python make_nets.py --solid pyr4 --outdir images
```

- `--solid`: `box` / `tetra` / `prism3` / `prism4` / `prism5` / `pyr4` / `pyr5` / `lshape` / `cutout`
- `--top-n`: 保存する枚数（0 で全件、既定 0）
- `--sort-by`: `area`（外接矩形の面積）/ `perimeter`（外周長）
- `--outdir`: 保存先ディレクトリ（既定 `nets`）

> **`--sort-by perimeter` の定義変更（v3 PR3）**
> 以前は外接矩形の周長 `2*(w+h)` で並べていましたが、現在は API と同じ
> **外周長（全面の周長合計 − 2 × 折り曲げ軸の長さ合計）** で並べます。
> 表示される値だけでなく、`--top-n` で**選ばれる展開図も変わります**
> （L 字型の展開図のように外接矩形の空部分を含む形状では差が出るため）。
> たとえば 5角柱は上位3枚すべてが外周長 19.4046 になります。

独自の立体を描きたい場合は `make_nets.py` に頂点と面を定義した関数を追加し、`SOLIDS` に登録してください。API 経由で計算したい場合は下の `POST /api/net/calculate` を使えば同じ結果が得られます。

### API サーバーの起動
API サーバーをローカルで起動します。顧客の端末上のフロントエンドアプリと連携して動作させるためのもの。

```powershell
uvicorn api_server:app --reload
```
サーバー起動後、ブラウザで **`http://127.0.0.1:8000/docs`** にアクセスすると、インタラクティブな API仕様書（Swagger UI）から動作確認が可能です。

ブラウザからのアクセスは、既定で `http://localhost` / `http://127.0.0.1` とポート `3000`、`5173`、およびローカルの `file://` ページ（Origin が `null`）を許可しています。本番や別ポートのフロントエンドを許可する場合は、起動前に Origin をカンマ区切りで指定してください。

```powershell
$env:CORS_ORIGINS = "http://localhost:3000,https://example.com"
uvicorn api_server:app --reload
```

#### 提供 API

- **`POST /api/net/calculate`**: 任意の多面体から評価指標の上位 n 個の展開図を計算する
  - **リクエストボディ**
    - `vertices` (必須, `number[][]`): 頂点座標 `[[x, y, z], ...]`
    - `faces` (必須, `integer[][]`): 各面の頂点インデックス。三角形・四角形・N 角形の混在可。**巻き順は任意**（外向きに正規化される）
    - `top_n` (任意, `1〜10000`, 既定 `1`): 返す展開図の数
    - `sort_by` (任意, `area` / `perimeter`, 既定 `area`): 並べ替えの基準
  - **レスポンス**: 展開図の JSON。スキーマは [specification.md] を参照。
  - **422 で返る場合**: 入力エラー（面数不足・頂点インデックス範囲外・非平面な面・境界あり・非連結・有限でない座標・未知のフィールド）と、計算量の上限超過（全域木が 10,000 本超）。**後者は入力が間違っているわけではなく**、詳細は「既知の制限」を参照。
    なお**凹角（非凸）や貫通穴は 422 にしません**。正当な入力として受理します。

#### リクエスト例

```json
{
  "vertices": [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
  ],
  "faces": [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
  ],
  "top_n": 2,
  "sort_by": "area"
}
```

#### レスポンス例

```json
{
  "solid": {
    "vertex_count": 8,
    "face_count": 6,
    "face_vertex_counts": [4, 4, 4, 4, 4, 4],
    "bounding_box": { "width": 1.0, "height": 1.0, "depth": 1.0 },
    "total_area": 6.0
  },
  "sort_by": "area",
  "total_unique_nets": 11,
  "returned": 2,
  "nets": [
    {
      "net_id": 1,
      "metrics": {
        "bounding_box": { "width": 5.0, "height": 2.0 },
        "bounding_box_area": 10.0,
        "perimeter": 14.0
      },
      "faces": [
        {
          "face_id": 0,
          "name": "面0",
          "vertices_2d": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
          "parent_face_id": null,
          "dihedral_angle": null,
          "fold_angle": null,
          "fold_axis_3d": null,
          "fold_axis_2d": null
        },
        {
          "face_id": 2,
          "name": "面2",
          "vertices_2d": [[0.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]],
          "parent_face_id": 0,
          "dihedral_angle": 90.0,
          "fold_angle": 90.0,
          "fold_axis_3d": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
          "fold_axis_2d": [[0.0, 0.0], [0.0, 1.0]]
        }
      ]
    }
  ]
}
```

各フィールドの意味:

| フィールド | 内容 |
| --- | --- |
| `solid.*` | 入力立体の概要（頂点数・面数・面ごとの頂点数・外接寸法・表面積合計） |
| `total_unique_nets` | ユニークな展開図の総数（鏡映を含めて同一視した数） |
| `returned` | 実際に返した枚数（`top_n` と総数の小さい方） |
| `net_id` | 1 からの連番。`sort_by` の良い順 |
| `metrics.bounding_box` | 展開図の外接矩形の幅・高さ |
| `metrics.bounding_box_area` | 外接矩形の面積 |
| `metrics.perimeter` | 展開図の外周長 = 全面の周長合計 − 2 × 折り曲げ軸の長さ合計。外接矩形の周長ではない |
| `vertices_2d` | 展開図上のその面の頂点座標（頂点数は元の面と一致） |
| `parent_face_id` | 展開図の木構造における親の面。根（面0）は `null` |
| `dihedral_angle` | 親の面との**内側の二面角**（度）。凸角は 0〜180、凹角は 180〜360。根は `null` |
| `fold_angle` | 平らな展開図を立体に畳み込むのに必要な回転角 = `180 − dihedral_angle`。凹角では負になる。根は `null` |
| `fold_axis_3d` | 立体上での折り曲げ軸（共有辺の 3D 座標 2 点）。根は `null` |
| `fold_axis_2d` | 展開図上の折り曲げ軸（この面の `vertices_2d` に含まれる 2 点）。根は `null` |

非凸の立体では `dihedral_angle` が 180 度を超えます。例えば凹角のある L 字柱では 270 度、対応する `fold_angle` は −90 度です。

#### 一意性の検算

`test_unique_nets_geometry.py` では、平面図形の合同（全頂点対の距離の多重集合）で一意数を数え直し、両者が一致することを確認しています。

```powershell
pytest -q
```

---

## 既知の制限

### 展開図の候補数（全域木）に上限がある

展開図は面グラフの全域木と1対1で対応し、その数は面数に対して**指数的に**増えます。面を1枚減らすだけで大きく変わります（7角柱 35,287 → 6角柱 8,100）。

そのためサーバーは列挙に着手する**前**に Matrix-Tree 定理で全域木の数を数え、`MAX_SPANNING_TREES`（既定 **10,000**）を超えていれば列挙せず、`type: "too_many_spanning_trees"` の 422 で返します。数え上げ自体は `O(面数³)` で軽く、この判定に時間がかかりません。

**上限の根拠は「応答時間の予算」であって、正しさや入力の品質の話ではありません。** 1枚あたり約 0.3ms で処理できるため、10,000 本 ≒ 3秒です。同期 HTTP として現実的な範囲に収めるための値です。

| 立体 | 面 | 全域木 | 一意な展開図 | 所要時間 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 5角柱 | 7 | 1,805 | 99 | 0.51s | 通る |
| 6角柱 | 8 | 8,100 | 354 | 2.82s | 通る（既定上限の手前） |
| 7角柱 | 9 | 35,287 | 1,290 | 10.03s | 422 |
| 8角柱 | 10 | 150,528 | 4,762 | 48.74s | 422 |

この表は `iter_unique_nets` で重複判定を展開より先に回した後の実測です。それ以前は 6角柱 9.81s / 7角柱 54.01s / 8角柱 216.25s でした。

**422 になっても入力は間違っていません。** 面数不足・非平面・境界・非連結といった入力エラーとは別物なので、クライアントは `detail[0]["type"]` で区別してください（→「エラーの構造」）。

**回避策**: 面を1枚減らす（7角柱→6角柱）か角数を減らすと全域木の数が激減します。

なお `top_n` の上限は `MAX_TOP_N` という**別の定数**です。全域木の数とは無関係なので、値を変えても互いに影響しません。

---

## エラーの構造

`422` の `detail` は**常にオブジェクトの配列**です。FastAPI / Pydantic のフィールド検証エラーも本 API 自身のエラーも同じ形なので、クライアントは一律に処理できます。

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
| `type` | 機械可読な分類。**クライアントはこれで分岐する** |
| `loc` | 発生箇所。本 API のエラーは `["body"]`。フィールド検証は `["body", "top_n"]` のようにフィールド名まで入る |
| `msg` | 人間向けの説明（日本語）。表示に回す |
| `spanning_trees` / `limit` | `type` が `too_many_spanning_trees` のときだけ付く整数 |

```python
err = r.json()["detail"][0]
if err["type"] == "too_many_spanning_trees":
    # 入力は正当。面を減らせば通る
    ...
else:
    # 入力の修正が必要。err["msg"] を表示する
    ...
```

`detail` が文字列と配列に分かれていた頃は、本文を読まないと「入力ミス」と「計算量の上限超過」を区別できませんでした。統一したのはこのためです。

詳細な `type` 一覧は `specification.md` の 4.1 / 4.2 を参照してください。422 の形は `/docs`（OpenAPI）にも明記してあります。
