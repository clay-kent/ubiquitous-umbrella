# v3 PR8 報告書 — ドキュメント（最終）

**基準**: PR7 終了時 = 263 passed、mypy = 0 errors

---

## 1. 計画の (a) — 独立した9点

| # | 対象 | 状態 |
|---|---|---|
| ① | spec 4.1 の `greater_than` → `greater_than_equal` | ✅ |
| ② | README:45 `httpx` → `httpx2` | ✅（**PR4 で未着手だった**） |
| ③ | NaN / Infinity のヘッダ依存の注記 | ✅ |
| ④ | README の性能表を実測値に更新 | ✅（§2） |
| ⑤ | 未 documenting だった4型を追記 | ✅ |
| ⑥ | `SolidRequest.examples` を立方体に差し替え | ✅（§3） |
| ⑦ | `minItems` を `json_schema_extra` で注記 | ✅（§3） |
| ⑧ | spec 3.1 の `orient_faces_outward` を parity union-find に更新 | ✅ |
| ⑨ | spec 4.2 の `spanning_trees`/`limit` を「実数」→「整数」 | ✅ |
| — | 「networkx (VF2)」の削除 | ✅（PR4） |

### ⑤ 追記した型

| 型 | 到達可能か | 補足 |
|---|---|---|
| `face_too_few_vertices` | あり | 1面の頂点数が3未満 |
| `vertices_shape` | あり | `[[x,y,z],...]` でない |
| `vertices_invalid` | あり | 数値配列として解釈できない |
| `invalid_solid` | あり | 上のどれにも当たらない `ValueError` の受け皿 |
| `search_budget_exceeded` | あり（PR4 で導入） | **PR4 の (c) 部品が未着手だったのでここで追加** |

---

## 2. ④ 性能表を実測した（And 自己申告ではなく）

計画は「Update README:200 performance figures to measured values」。
**PR1 で約2.3倍速くなっているので旧値は 使えない。**

### 計測方法

- API 実測: `TestClient` 経由で1リクエスト往復。3回の**最小値**（ウォームアップ1回込み）
- 参考値: `iter_unique_nets` を最後まで回す時間（7/8角柱は API が 422 で止まるため）

### 結果

| 立体 | 面 | 全域木 | 一意 | 旧(README) | API 実測 | 参考(direct) |
|---|---|---|---|---|---|---|
| 5角柱 | 7 | 1,805 | 99 | 0.51s | **0.42s** | 0.51s |
| 6角柱 | 8 | 8,100 | 354 | 2.82s | **1.63s** | 1.98s |
| 7角柱 | 9 | 35,287 | 1,290 | 10.03s | 0.01s (422) | 10.39s |
| 8角柱 | 10 | 150,528 | 4,762 | 48.74s | 0.01s (422) | 63.52s |

### ⚠️ 8角柱について正直に書く

旧値 48.74s はこの環境で**再現しなかった**（実測 57〜74s とばらつく）。
以下を README に明記した:

- 旧値は別マシン・別負荷での計測で、この環境では再現しなかった
- 8角柱は計測ごとに 60〜74秒と振れるため、表には **422 の応答（0.01秒）だけ**を載せる
- 参考値として direct の実測も併記

「速くなった」と書けば PR1 の効果を強調しすぎるし、
「遅くなった」と書けば再現していない比較になる。
**利用者が実際に体験するのは API の応答時間だけ**なので、
表の主列を API 実測に據えた。これは旧表の意味（列挙所要時間）から変わるので、
列名と説明文も変えた。

### 計測中に見つけた自分のバグ

`prism8` の初回計測で `unique = 0` が出た。**コードのバグではなく計測スクリプトのバグ**で、
3回ループした**後**に `len(nets)` を参照していたため、3回目（消費済みジェネレータ）の
0 を出力していた。単体で再実行して 4,762 を確認した。

---

## 3. ⑥⑦ OpenAPI の例と minItems

### ⑥ 例の差し替え（**これが実バグの修正**）

旧 `examples` は:
- `vertices`: 4頂点の四面体
- `faces`: **2面**の断片 `[[0,1,2],[0,2,3]]`

どちらも **その body を投げると 422** になる。つまり OpenAPI の例を
コピペしても動かない。立方体1個ぶん（8頂点・6面）に差し替えた。

**検証**: OpenAPI から取得した `examples` をそのまま POST →
`200` / `total_unique_nets: 11`。

### ⑦ `minItems` は注記だけ

```python
json_schema_extra={"minItems": 4}
```

**検証**（3点）:
1. `/openapi.json` に `minItems: 4` が両フィールドに出る
2. 2頂点・1面の入力を投げると `too_short` ではなく **`too_few_vertices`**
   → 検証していないことの証明

検証にすると Pydantic の `too_short` が先に出て、本 API の日本語 `msg` と
並んでしまう。判定は `validate_solid` が一箇所で持つ、という方針を守った。

---

## 4. ⑧ spec 3.1 の `orient_faces_outward`

旧説明「差分集合で連結成分をまとめ、符号付き体積で外向きに正規化する」は
**実装と違っていた**。現在の実装は parity union-find で
「各辺を逆向きに2面が通る」制約を解いている。

差分集合は orient_faces_outward に一切出てこない（実装は parity union-find、
体積の符号は最後に1回だけ見る）。旧記述は誤り。

併せて**前条件と失敗時の例外**も spec に書いた（PR5 で docstring に入れたものの反映）:

> 前条件: `validate_solid` を先に通していること。前条件を破った入力は
> タプル展開が `ValueError` になる（黙って無視しない）。

---

## 5. ⑨ 「実数」→「整数」

`spanning_trees` / `limit` は `MAX_SPANNING_TREES` と同じ整数。
JSON Schema 側（`api_server.ERROR_RESPONSE`）も `{"type": "integer"}`。
spec だけが「実数」になっていた。

---

## 6. ③ NaN / Infinity のヘッダ依存

`non_finite_vertex` は **JSON として不正な値を投げた場合だけ到達する**。
`application/json` で送ると通信段（クライアントまたはサーバの JSON パーサ）で
先に弾かれ、API には届かない。この型の**検証方法**が spec に書いていなかった。

追記した内容:
- `Content-Type: application/json` では到達しない
- 確認するにはヘッダを `text/plain` にして本文だけ送る（`httpx2` では `content=`）。

---

## 7. 結果

```
pytest -q  →  263 passed            （±0）
mypy       →  Success: no issues found in 3 source files    （±0）
requirements.txt --check  →  exit 0
```

文字化け（ハングル混入・U+FFFD）の機械スキャンを全ソース＋ドキュメントで実行:
**0 件**。

---

## 8. ⚠️ 執筆中に見つけた「私の計測バグ」

spec 4.1 の表を**機械的に突き合わせる**スクリプトを書いて確認した。
ここで自分の検証スクリプトが2回間違えた。

| 書いた期待値 | 実際の返答 | 原因 |
| --- | --- | --- |
| `vertices_shape` | `vertices_invalid` | 検証スクリプトが **ragged 配列**（`[[0,0,0],[1,0],...]`）を送っていた。`vertices_shape` は**幅が3でない矩形配列**で出る |
| `duplicate_vertex`（bowtie） | `vertex_index_out_of_range` | 検証スクリプト内で変数が**使い回され**、別の立体の面インデックスを参照していた |

両方とも **spec の記述は正しく、検証スクリプトが間違っていた**。
修正したスクリプトでの結果:

```
too_few_vertices           -> too_few_vertices
too_few_faces              -> too_few_faces
vertex_index_out_of_range  -> vertex_index_out_of_range
duplicate_face_vertex      -> duplicate_face_vertex
face_too_few_vertices      -> face_too_few_vertices
vertices_shape             -> vertices_shape     ([[0,0],[1,1],...] = 幅2の矩形)
vertices_invalid           -> vertices_invalid   (ragged)
non_planar_face            -> non_planar_face
duplicate_vertex           -> duplicate_vertex
too_many_spanning_trees    -> too_many_spanning_trees
```

### `invalid_solid` は到達を確認できなかった

`invalid_solid` は `except ValueError` の受け皿として**コードには存在する**が、
PR4（`except ValueError` の削除）と PR5（死んだ守護の削除）の後は
到達経路がほぼ無い。**到達を確認できなかったので、
spec の表には「具体的な到達経路は未確認」と明記した**。
「到達可能」と書いて肯定是 instalaない。

## 9. v3 全体としての最終状態

| PR | commit | テスト | mypy |
|---|---|---|---|
| PR1 | `883c07d` | 212 + 2 xfail | — |
| PR2 | `fc4081a` | 244 | 27 |
| PR3 | `49548d8` | 254 | 26 |
| PR4 | `aaf14ed` | 258 | 15 |
| PR5 | `2205b9e` | 258 | 15 |
| PR6 | `d596aff` | 263 | 15 |
| PR7 | `6efdf2b` | 263 | 0 |
| PR8 | 本 commit | 263 | 0 |

- テスト: **149 → 263**（+114）
- mypy: **28 → 0**
- `fifth.py`: 1,082 行 → 998 行（PR3 時点比 84 行削減、VF2 と `__main__` を削除）
- 報告書: `reports/v3-pr1.md` 〜 `v3-pr8.md`

### 計画と実測が食い違った点（4件、再掲）

1. **PR1**: 「`fold_axis_2d` は 1e-9 まで正しい」→ 実際は 1e-8。
   立体が物理的に大きいほど早く壊れる。bbox 正規化後は 1e-14 まで。
2. **PR1**: `is_valid_layout` の式を計画のままだと `prism4` が scale=1 でも 0枚
3. **PR3**: 「転倒数 0、`top_n` 選択は変わらない」→ prism5 で **2586**、
   選択は変わる。`2*(w+h)` は外周長とは別の量。
4. **PR7**: PR8 の性能表の旧値は別環境で再現せず、8角柱は 48.74s → 60〜74s

いずれも「計画の前提を実測で暴いた」ケースで、
いずれも**テストで固定して**引き継いだ。
