直方体の寸法（幅・高さ・奥行き）から、重なりのないユニークな展開図（全域木）をすべて列挙・計算・視覚化し、APIサーバーや画像保存、3D STLモデルの出力を行う Python スクリプト群。

## 主な機能
- **展開図（全域木）の列挙とグラフ同型判定**: 平面で構成される任意の立体の面をグラフノードとし、隣接関係から全域木を列挙。回転対称性などを考慮したリボングラフ同型判定により、ユニークな展開図のみを抽出します。
- **2D自動レイアウト**: 各面の3D幾何構造を展開し、Shapely を使用して2Dポリゴン座標に変換。自己交差（重なり）のない正しい展開図をフィルタリングします。
- **画像出力**: 展開図の2Dレイアウトをプロットし、PNGファイルとしてローカルディレクトリに保存します。
- **FastAPIによるWeb APIサーバー**: 展開図の計算結果（面ごとの2D座標、親子関係、折り曲げ軸等）のJSON配信、および3Dプリント用STLファイルのダウンロード機能を提供します。
- **ユニット/統合テスト**: pytestによる網羅的なテスト。

## ディレクトリ構造

```text
├── cuboids/               # 生成された展開図画像の保存先（自動作成）
├── fifth.py               # 展開図作成のコア機能。さまざまな立体の展開図を作成できる。
├── make_cuboid.py         # 指定した寸法の直方体展開図を計算し、PNG画像を生成する**お試し用**スクリプト
├── api_server.py          # APIサーバー
├── test_api_server.py     # ユニット・結合テスト
├── pyproject.toml         # 依存関係定義
└── README.md              # 本ドキュメント
```

---

## セットアップ

### 必須環境
- Python 3.10 以上

### 依存パッケージのインストール
本プロジェクトは、`fastapi`, `uvicorn`, `shapely`, `numpy`, `matplotlib`, `networkx`, `pytest` を使用します。

#### uv を使用する場合:
```powershell
uv pip install -r pyproject.toml
```

#### pip を使用する場合:
```powershell
pip install fastapi uvicorn shapely numpy matplotlib networkx pytest
```

---

## 使い方

### 展開図画像の生成
任意の直方体から得られる展開図パターンをすべて計算し、`cuboids/` フォルダにPNG画像として保存します。

```powershell
python make_cuboid.py
```
* 直方体の寸法を変更したい場合は、[make_cuboid.py]内の `main()` 関数にある `w, h, d` の値を直接変更してください。

### APIサーバーの起動
APIサーバーをローカルで起動します。顧客の端末上のフロントエンドアプリと連携して動作させるためのもの。

```powershell
uvicorn api_server:app --reload
```
サーバー起動後、ブラウザで **`http://127.0.0.1:8000/docs`** にアクセスすると、インタラクティブなAPI仕様書（Swagger UI）から動作確認が可能です。

#### 提供API
- **`GET /api/net/calculate`**: 展開図計算API
  - パラメータ: `w` (幅), `h` (高さ), `d` (奥行き), `sort_by` (`area` / `perimeter` - 外枠面積、外周長といった評価指標でソート)
- **`GET /api/net/stl`**: STLダウンロードAPI
  - パラメータ: `w` (幅), `h` (高さ), `d` (奥行き)

/api/net/caluculateの返り値はJSONデータです。以下に例を示します。
JSONスキーマ形式は[specification.md]を参照してください。
```
{
"dimensions": {
  "width": 1.0,
  "height": 1.5,
  "depth": 2.0
},
"nets": [
  {
    "net_id": 1,
    "metrics": {
      "bounding_box": {
        "width": 5.0,
        "height": 4.5
      },
      "bounding_box_area": 22.5,
      "perimeter": 19.0
    },
    "faces": [
      {
        "face_id": 0,
        "name": "底面",
        "vertices_2d": [
          [0.0, 0.0],
          [1.0, 0.0],
          [1.0, 1.5],
          [0.0, 1.5]
        ],
        "parent_face_id": null,
        "fold_axis": null,
        "fold_angle": 0.0
      },
      {
        "face_id": 2,
        "name": "前面",
        "vertices_2d": [
          [0.0, 0.0],
          [1.0, 0.0],
          [1.0, -2.0],
          [0.0, -2.0]
        ],
        "parent_face_id": 0,
        "fold_axis": [
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0]
        ],
        "fold_angle": 90.0
      }
    ]
  }
]
}
```
