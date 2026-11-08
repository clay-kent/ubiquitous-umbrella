任意の多面体（凸・非凸・貫通穴つきなど）から、重なりのないユニークな展開図を列挙・計算・視覚化し、API サーバーの JSON 配信や PNG 画像保存を行う Python スクリプト群。

## 主な機能
- **任意多面体に対応**: 直方体だけでなく、四面体・角柱・角錐・L 字柱（凹角あり）・貫通穴つき（genus=1）などの頂点と面のリストを受け付けます。
- **展開図（全域木）の列挙と一意性判定**: 面をグラフノードとし、隣接関係から全域木を列挙。半辺構造の正規形（個別化-細分）で重複を潰し、ユニークな展開図のみを抽出します。**回転・並進・鏡映は同一視**するため、立方体は 11 種です（112 ではありません）。
- **2D 自動レイアウト**: 各面の 3D 幾何構造を展開し、Shapely を使用して 2D ポリゴン座標に変換。自己交差（重なり）のない正しい展開図のみを返します。
- **折り組み立て情報の出力**: 面ごとの 2D 座標、親子関係、内側の二面角、折り曲げ角（= 180 − 二面角）、3D / 2D の折り曲げ軸。
- **のりしろ（glue tab）の自動配置**: 展開図で切れている辺（立体にすると接する辺）ごとに、両面のうち**ちょうど片側**へ台形ののりしろを付ける。のりしろ同士が重なる場合は奥行きを縮めて重なりを減らす（下限あり）。のりしろは指標（外接矩形・外周長・折る回数）には含めない。
- **評価指標**: 外接矩形の面積、外周長、折る回数（同じ直線上で同じ折り方の折り曲げ軸を 1 回にまとめた本数）。
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
├── net_test_cache.py            # テスト専用。重い展開図計算をファイル間で共有する
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
pip install fastapi uvicorn shapely numpy matplotlib pytest httpx2
```

---

## 使い方

### 展開図画像の生成
プリセットした立体の一意な展開図をすべて計算し、PNG 画像として保存します。

```powershell
python make_nets.py                              # 既定の直方体 (1.0 x 1.5 x 2.0)
python make_nets.py --solid prism5               # 5 角柱
python make_nets.py --solid lshape --top-n 6 --sort-by perimeter
python make_nets.py --solid lshape --top-n 6 --sort-by folds
python make_nets.py --solid pyr4 --outdir images
python make_nets.py --solid namacha --top-n 3   # 生茶 525mL ペットボトル（ローポリ近似）
python make_nets.py --solid namacha_box         # 生茶が入る直方体の箱
```

- `--solid`: `box` / `tetra` / `prism3` / `prism4` / `prism5` / `pyr4` / `pyr5` / `lshape` / `namacha` / `namacha_box` / `cutout`
- `--top-n`: 保存する枚数（0 で全件、既定 0）
- `--sort-by`: `area`（外接矩形の面積）/ `perimeter`（外周長）/ `folds`（折る回数）
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
    - `sort_by` (任意, `area` / `perimeter` / `folds`, 既定 `area`): 並べ替えの基準
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
        "perimeter": 14.0,
        "fold_count": 5
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
      ],
      "glue_tabs": [
        {
          "face_id": 0,
          "neighbor_face_id": 4,
          "edge_3d": [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
          "edge_2d": [[1.0, 0.0], [1.0, 1.0]],
          "vertices_2d": [[1.0, 0.0], [1.0, 1.0], [1.25, 0.75], [1.25, 0.25]]
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
| `metrics.fold_count` | 折る回数 = 同じ直線上で山折り/谷折りと折り角が同じ軸を 1 回にまとめた本数（→「折る回数の定義」） |
| `vertices_2d` | 展開図上のその面の頂点座標（頂点数は元の面と一致） |
| `parent_face_id` | 展開図の木構造における親の面。根（面0）は `null` |
| `dihedral_angle` | 親の面との**内側の二面角**（度）。凸角は 0〜180、凹角は 180〜360。根は `null` |
| `fold_angle` | 平らな展開図を立体に畳み込むのに必要な回転角 = `180 − dihedral_angle`。凹角では負になる。根は `null` |
| `fold_axis_3d` | 立体上での折り曲げ軸（共有辺の 3D 座標 2 点）。根は `null` |
| `fold_axis_2d` | 展開図上の折り曲げ軸（この面の `vertices_2d` に含まれる 2 点）。根は `null` |
| `glue_tabs` | のりしろの配列。切れている共有辺ごとに、両面のうちちょうど片側へ 1 枚。各要素は `face_id`（付ける面）、`neighbor_face_id`（接着する相手）、`edge_3d` / `edge_2d`（貼り付け辺）、`vertices_2d`（台形の 4 点）を持つ |

非凸の立体では `dihedral_angle` が 180 度を超えます。例えば凹角のある L 字柱では 270 度、対応する `fold_angle` は −90 度です。

#### 折る回数の定義

展開図の折り曲げ軸（全域木の辺）は必ず「面数 − 1」本あります。同じ直線上にあり、かつ折り角（山折り/谷折りとその大きさ）が等しい軸は 1 回の折り操作でまとめて折れるので、`fold_count` は**「直線 + 折り方」ごとにまとめた本数**とします。

- 単位立方体の 11 種はすべて 5（= 6 − 1）。
- L 字柱の 1516 種は 6 または 7（= 8 − 1）。`sort_by: "folds"` では 6 回で折れる展開図が先に来ます。
- 同じ直線上でも、山折りと谷折りが混在する場合や折り角の大きさが違う場合は別の折り操作として数えます。1 回の直線折りでは、その直線上の紙を同じ向き・同じ角度にしか折れないためです。

詳細は `specification.md` の 3.5 を参照してください。

#### 一意性の検算

`test_unique_nets_geometry.py` では、平面図形の合同（全頂点対の距離の多重集合）で一意数を数え直し、両者が一致することを確認しています。

```powershell
pytest -q                  # 通常（slow を除外）
pytest -q -m ""            # 全部（CI 向け）
pytest -q -m slow          # 重いケースだけ
pytest -q --durations=10   # 遅いテストの内訳を見る
```

L 字柱（1516 種）と 6 角柱（354 種）は全域木が 8,100 本あり、1 回の列挙に 7〜9 秒かかります。同じ立体を何度も列挙しないよう、テストは `net_test_cache.py` を経由して結果を共有します。`slow` を付けたケースだけが既定で除外されます。

---

## パフォーマンス計測（開発用）

`tools/` に計測・検証スクリプトがあります。**`fifth.py` の動作には影響しません。**

| スクリプト | 用途 |
| --- | --- |
| `tools/bench_ladder.py` | 複雑度ラダー（box → stairs）の一括ベンチ |
| `tools/bench_repeat.py` | ノイズ対策。n回反復して**中央値**を出す（単発計測では判断しない） |
| `tools/api_bench.py` | API の応答時間を測る（README の表を更新するときに使う） |
| `tools/profile_hot.py` | cProfile でホットスポットを特定 |
| `tools/timing_parts.py` | 列挙 / 署名 / レイアウト / 重なり判定 に分けて計測 |
| `tools/bench_layout_only.py` | `layout_tree` 単体のコスト（vs ベースラインは `git stash` で A/B） |
| `tools/bench_valid_skip.py` | 重なり判定の折り線ペア省略の効果を A/B |
| `tools/probe_tree_adjacent.py` | bbox を通過するペアのうち折り線ペアの割合 |
| `tools/probe_wl_rounds.py` | 色細分のラウンドごとの費用と残存行数（打ち切り案の可否判断） |
| `tools/hash_layout_bits.py` | **全立体の展開図座標をビット単位でハッシュ**（丸めを変えていないことの証明） |
| `tools/hash_metrics_rounded.py` | 全展開図の指標を**9桁丸めでハッシュ**（丸めが ULP 級に変わるときの観測差チェック） |
| `tools/bench_signatures_cpu.py` | 署名フェーズを **CPU 時間**で測る（外部負荷に強い。壁時計が5倍振れるとき用） |
| `tools/bench_pipeline_cpu.py` | 列挙＋一意化の**パイプライン全体**を CPU 時間で測る（A/B 用） |
| `tools/probe_ab_pipeline.py` | **同一プロセス内で新旧を交互に走らせる A/B**（HEAD の `fifth.py` を別モジュールとして読む）。パイプライン全体と `_batch_stabilize` の累計を出す。**この環境で 5% 未満の差を判定できる唯一の方法** |
| `tools/probe_ab_stabilize.py` | 同じ方式で `_batch_stabilize` 単体を比べる（16ms 未満の呼び出しは内側で繰り返す） |
| `tools/probe_pipeline_phases.py` | 現行コードのフェーズ別内訳（列挙 / 署名 / 展開 / 重なり判定）を関数ラップで測る |
| `tools/probe_keys_parts.py` | `_batch_canonical_keys` の部位別（配列化 / 位置 / 接着 / WL / 個別化 / キー化） |
| `tools/probe_stabilize_now.py` | `_batch_stabilize` の部位別（初期化 / 更新 / 色数 / 間引き）と ns/row-round |
| `tools/probe_valid_batch_parts.py` | `_valid_batch` の部位別（前置フィルタ / 取り出し / 交差計算）と交差回数 |
| `tools/check_layout_batch_bits.py` | 窓単位のバッチ展開が `layout_tree` と**ビット一致**することの確認（204,633 本） |
| `tools/probe_window_size.py` | 窓サイズの比較（2048 が最小。出力は完全に同一） |
| `tools/probe_area_filter.py` | 「面積の総和 > 外接矩形」による事前棄却が何割弾けるか（3.7%。不採用の根拠） |
| `tools/probe_batched_intersection.py` | 交差計算の完全バッチ化（判定は一致するが 1.8x 遅い。不採用の根拠） |
| `tools/probe_candidate_order.py` | 候補ペアの走査順を変えても交差回数が減らないことの確認 |
| `tools/probe_wl_collisions.py` | 線形色ハッシュが早期ラウンドで衝突する行数の実測（Cycle 21 の根拠） |
| `tools/probe_ab_enum.py` | 列挙そのものの新旧 A/B（パイプラインの A/B は木を先に作るので列挙が見えない） |
| `tools/probe_is_connected.py` | 連結判定の速い実装を**実際に呼ばれた入力**で比較（同入力・全件一致を確認） |
| `tools/probe_enum_parts.py` | 列挙の内訳（連結判定の回数と時間） |
| `tools/probe_asarray.py` | 窓の木→配列の経路比較（`np.fromiter` が 3.0x） |
| `tools/check_u64_ring.py` | uint64（mod 2^64）の結合法則・分配法則の確認 |
| `tools/check_small_dot_bits.py` | 小さい内積/行列ベクトルがスカラ式とビット一致するかの確認 |
| `tools/check_projection_bits.py` | 投影のバッチ化がビット一致するかの確認（しない） |
| `tools/check_projection_axes_bits.py` | 軸/定数項だけのバッチが**ビット一致**するかの確認（する） |
| `tools/check_projection_batching.py` | 投影の全面バッチが GEMV と一致しないことの確認（Cycle 18 の根拠） |
| `tools/check_wl_update_inplace.py` | WL 更新の in-place 化が**ビット一致**することの確認 |
| `tools/probe_window_size.py` | 署名バッチの窓サイズ比較（2048 が最小） |
| `tools/scan_script_chars.py` | ソース/文書に混入した別言語の文字（文字化け）を検出 |
| `tools/check_keys.py` | 一意数の既知値と照合する |
| `tools/compare_partitions.py` | **高速化前後の「分割の一致」を検証する**（最重要） |

### ⚠️ 高速化したら必ず検証すること

既存の一意数のテスト（`test_unique_net_counts`）は、**高速化前後が同じ数字を返す
壊れたコードと、正しいコードを区別できません**。実際にそれで間違う例は
`reports/v4-optimization.md` の Cycle 2 にあります。

高速化をしたら `tools/compare_partitions.py` と
`test_batched_partition_matches_per_tree_partition` で
**分割そのもの**を比較してください。

計測と最適化の記録は [`reports/v4-optimization.md`](reports/v4-optimization.md) にあります。

---

## 既知の制限

### 展開図の候補数（全域木）に上限がある

展開図は面グラフの全域木と1対1で対応し、その数は面数に対して**指数的に**増えます。面を1枚減らすだけで大きく変わります（7角柱 35,287 → 6角柱 8,100）。

そのためサーバーは列挙に着手する**前**に Matrix-Tree 定理で全域木の数を数え、`MAX_SPANNING_TREES`（既定 **10,000**）を超えていれば列挙せず、`type: "too_many_spanning_trees"` の 422 で返します。数え上げ自体は `O(面数³)` で軽く、この判定に時間がかかりません。

**上限の根拠は「応答時間の予算」であって、正しさや入力の品質の話ではありません。** 上限直前の6角柱（全域木 8,100 本）が約 0.5 秒で返るため、10,000 本は同じオーダーです。同期 HTTP として現実的な範囲に収めるための値です。

| 立体 | 面 | 全域木 | 一意な展開図 | API 実測 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 5角柱 | 7 | 1,805 | 99 | 0.08s | 通る |
| 6角柱 | 8 | 8,100 | 354 | 0.36s | 通る（既定上限の手前） |
| 7角柱 | 9 | 35,287 | 1,290 | 0.01s | 422 |
| 8角柱 | 10 | 150,528 | 4,762 | 0.01s | 422 |

「API 実測」は `TestClient` 経由の1リクエスト往復（利用者が感じる時間）です。7角柱以降は**列挙に着手する前に 422 なので 0.01秒**で返ります。

> **この表は 2026-10-17 に再計測した値**（`tools/api_bench.py`、3回の最小値）。
> それ以前は 5角柱 0.42s / 6角柱 1.63s だった。
> その後 Cycle 1〜15（`reports/v4-optimization.md`）で継続的に高速化しており、
> 現在は 5角柱・6角柱ともさらに短い。数値はマシンと負荷によって変わるため、
> 引用する場合は自分の環境で `tools/api_bench.py` を走らせて確認してください。

参考として、上限を無視して `iter_unique_nets` を最後まで回すと 7角柱 約1.9秒 / 階段柱（10面 / 150,528全域木）約13秒（実測）。8角柱は全域木が 150,528 本あるため 422 の応答だけを上の表に載せています。

### 計測の注意

**この環境の計測は ±19% ばらつきます**（同一コードで 37s〜46s を実測）。
単発の計測で「速くなった」と結論してはいけません。
反復して中央値で比較してください（`tools/bench_repeat.py`）。

### のりしろの計算コスト

`glue_tabs` は返す展開図 1 枚ごとに、切れている辺の両側候補と、面・他ののりしろとの重なりを調べます。実測（この環境）では L 字柱 1516 枚で約 4.3 秒、6 角柱 354 枚で約 0.9 秒（1 枚あたり約 2.6〜2.8ms）です。`top_n` を大きくするときは、この計算時間が展開図の列挙・指標計算に加算されます。

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
