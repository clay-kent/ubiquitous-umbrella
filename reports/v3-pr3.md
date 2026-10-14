# v3 PR3 報告書 — 周長指標の一本化

**基準**: PR2 終了時 = 244 passed、mypy --strict = 27 errors

---

## 1. 何をしたか

同じ「周長」を **2 つの異なる量**として計算していた。

| 置き場所 | 計算式 | 正しいか |
|---|---|---|
| `api_server.metrics_of` | 全面の周長合計 − 2 × 折り曲げ軸の長さ | ✅ 外周長 |
| `make_nets.py` の metric ループ | `2.0 * (w + h)` | ❌ **外接矩形の周長** |

`2*(w+h)` は展開図の外周長ではない。外接矩形の**空の部分**の周まで含む。
L 字型のように凹みのある展開図では、この空部分の分だけ値が大きくなる。

そこで指標の計算を `fifth.net_metrics` に一本化し、API と CLI の両方をこれに繋いだ。

```python
# fifth.py
class NetMetrics(NamedTuple):
    bounding_box_width: float
    bounding_box_height: float
    bounding_box_area: float
    perimeter: float

def net_metrics(polys, tree_edges, vertices, shared_of) -> NetMetrics: ...
```

レスポンスの JSON 形は `api_server.metrics_json` が NamedTuple を dict に戻す。
（NamedTuple はそのままだと JSON の**配列**になる。形は応答仕様なので API 側で決める。）

副次的に `prepare_layout` の中にあった `shared_of` の構築を `fifth.build_shared_of`
に切り出し、`prepare_layout` と `api_server` の双方が使う共有関数にした。

---

## 2. ⚠️ 計画の「並び順は変わらない」は誤りだった

計画 §4 PR3 はこう書いている。

> Per v2 §1.13-11's measurement (prism5, all 99 nets, **zero rank inversions**),
> `top_n` **selection does not change**. Only the **displayed value** changes.

**実測すると違う。** 転倒数（Kendall tau の距離）と top_n の集合一致を計測した。

| 立体 | 一意数 | 転倒数 | n=1 | n=3 | n=5 | n=10 | n=25 |
|---|---|---|---|---|---|---|---|
| cube | 11 | 0 | OK | OK | OK | OK | — |
| tetra | 2 | 0 | OK | — | — | — | — |
| box 1x1.5x2 | 54 | 334 | OK | OK | OK | OK | OK |
| prism3 | 9 | 26 | **NG** | OK | **NG** | — | — |
| pyr4 | 8 | 5 | **NG** | **NG** | **NG** | — | — |
| pyr5 | 15 | 31 | **NG** | **NG** | **NG** | **NG** | — |
| prism4 | 29 | 327 | OK | OK | **NG** | **NG** | **NG** |
| prism5 | 99 | **2586** | **NG** | **NG** | **NG** | **NG** | **NG** |

つまり **prism5 に 2586 個の転倒数**があり、`--top-n` で選ばれる展開図は
ほぼ入れ替わる。立方体だけ完全に一致する（11 枚はすべて外周長 14.0、
外接矩形の周長も同じ順位になる）。

**なぜそうなるか**: 2 つの式は別の量を測っている。立方体の十字形のように
「凹みなく外接矩形を埋める」展開図では順位が一致するが、
凹みのある展開図では一致しない。これは定義が誤っていた的后果であり、
**修正すべきバグそのもの**。

**結論: 変更は正当。表示値だけでなく `--sort-by perimeter` の選択結果も変わる。**
API の応答は変わらない（もともと正しい式を使っていた）。CLI だけが直った。

### やったこと（誤った主張を固定しなかった）

- 計画どおり「並び変わらない」を断定するテストは**書かなかった**（書いたら落ちた）
- 代わりに**定義そのもの**を固定するテストに書き換えた（§3）
- README と `--help` に「選択結果も変わる」ことを明記した

---

## 3. 追加したテスト（+10）

| テスト | 数 | 内容 |
|---|---|---|
| `test_perimeter_equals_face_sum_minus_folded_edges` | 5 | 定義どおり（周長合計 − 2×折り曲げ軸）であることを 5 立体 × 全展開図で確認 |
| `test_unit_cube_net_perimeter_is_14` | 1 | 単位立方体の 11 枚すべてが 14（6×4 − 2×5。手計算できる値） |
| `test_perimeter_is_not_the_bounding_box_perimeter`（fifth） | 1 | 外周長 ≥ 外接矩形周長、かつ L 字柱では厳密に大きい |
| `test_net_metrics_bbox_area_matches_width_times_height` | 1 | 面積 = 幅 × 高さ |
| `test_api_and_cli_compute_the_same_perimeter` | 1 | **API と CLI が同じ数字を出す**（PR3 の本体） |
| `test_perimeter_is_not_the_bounding_box_perimeter`（API） | 1 | 応答の `perimeter` が `2*(w+h)` でないことの直接確認 |

API 応答の値が変わっていないことは、PR2 で入れた golden（`test_api_golden.py`）
が引き続き保証している。

---

## 4. 結果

```
pytest -q  →  254 passed in 223.17s        （244 → +10）
mypy --strict →  26 errors                  （27 → -1）
```

| 完了条件 | 結果 |
|---|---|
| 指標の実装が `fifth.py` に 1 つだけある | ✅ |
| API 応答が意味的に不変 | ✅ golden 6件通過 |
| `--sort-by perimeter` の CLI 出力変更を 1 行で文書化 | ✅ README |
| テスト数 | ✅ 244 → 254 |

CLI の実挙動:
```
$ python make_nets.py --solid prism5 --top-n 3 --sort-by perimeter
Saved: nets\prism5_net_1.png  (外接矩形面積 19.3314, 外周長 19.4046)
Saved: nets\prism5_net_2.png  (外接矩形面積 25.1235, 外周長 19.4046)
Saved: nets\prism5_net_3.png  (外接矩形面積 23.654, 外周長 19.4046)
```
最小外周長の展開図が選ばれる正规的になっている（外接矩形の面積はばらばら）。

---

## 5. 次の PR への引き継ぎ

- **PR4 は計画どおり、手順どおりに進めること。**
  VF2 を削除する前に `test_unique_nets_geometry.py` に
  **lshape（8100 本 → 1516 枚）と prism6（8100 本 → 354 枚）を追加**すること。
  これが計画の必須手順であり、スキップすると最重量ケースが未検証のまま
  VF2 を削ることになる。
- PR4 で消す関数は `build_ribbon_graph` / `is_isomorphic_net` /
  `reverse_ribbon_graph` / `_ribbon_invariant` と
  `iter_unique_nets` の `except ValueError`。
  **`networkx` は dev 依存に移す**（`_brute_force_trees` が使うため）。
- 予算ガードは `SolidError("search_budget_exceeded")` にして
  既存の `except fifth.SolidError` で 422 にする。例外クラスは増やさない。