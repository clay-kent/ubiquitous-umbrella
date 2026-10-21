# v3 PR6 報告書 — `duplicate_vertex` 検証

**基準**: PR5 終了時 = 258 passed、mypy --strict = 15 errors

---

## 1. 追加したもの

`validate_solid` の早期に1つの検査を追加した。

```python
if len(np.unique(vertices, axis=0)) != n_v:
    raise SolidError(
        "duplicate_vertex",
        "同じ座標を持つ頂点が複数あります。頂点を1つにまとめてください。",
    )
```

面は**インデックス**で頂点を指すので、同じ座標に別インデックスが
2つあると「同じ点に2枚の状態が被る」ことになり、隣接グラフが
定義と食い違う。これを先に弾かないと、原因と無関係な downstream の
エラー（下表）が返って利用者が推測するしかない。

---

## 2. ⚠️ 利益の上限を正直に書く

計画も言っているが、実測するとさらに明確だった。
**この PR は「受理される範囲」を変えない。変わるのは `type` と `msg` だけ。**

### 変更前（実測）

| 入力 | `validate_solid` | API |
|---|---|---|
| bowtie（2立方体が頂点1つ共有） | **受理された** | 422 `not_connected` |
| 重複座標（`v[7] = v[0]`） | 422 `non_planar_face` | 422 `non_planar_face` |

### 変更後（実測）

| 入力 | `validate_solid` | API |
|---|---|---|
| bowtie | 422 `duplicate_vertex` | 422 `duplicate_vertex` |
| 重複座標 | 422 `duplicate_vertex` | 422 `duplicate_vertex` |

- **bowtie**: どちらの段階で見ても 422。`type` が `not_connected` → `duplicate_vertex` に変わっただけ。
- **重複座標**: `non_planar_face` → `duplicate_vertex`。前者は**原因を示していない**
  （平面性问题に見えるが、実際の修正は頂点の統合）。

つまりこれは **エラー分類の改善**であって、「隔離ホールの閉鎖」ではない。
新たに弾かれる入力は無い。

---

## 3. ⚠️ 既存のテスト1件が壊れた — 壊れていたのはテストの方

`test_disconnected_solid_is_rejected` が落ちた。

```python
a = cube_body()
b = cube_body(5.0, 5.0, 5.0)   # ← 原点から伸ばすので (0,0,0) を共有する
```

`cube_body(w,h,d)` は**原点を据えたまま**辺を伸ばす。よって2つの立方体は
頂点 `(0,0,0)` を共有しており、テストの意図（「離れた2成分」）と
入力が食い違っていた。

```
実測: validate_solid: 受理 → API 422 not_connected   （PR6前）
      validate_solid: duplicate_vertex → API 422 duplicate_vertex （PR6後）
```

つまり**従来の `not_connected` は誤分類だった**。追加した検査が
その誤りを正しいものにした。

対応:
1. `test_disconnected_solid_is_rejected` の fixture を**本当に離す**ように修正
   （2つめを平行移動）。これで本来の意図（離れた2成分 → `not_connected`）が
   カバーされるようになった。
2. 共有頂点のままの入力は別のテスト
   `test_cubes_sharing_only_a_vertex_are_duplicate_vertex_not_disconnected`
   として残した。**両方の分類を別々に固定**している。

---

## 4. 追加したテスト（+5）

| ファイル | テスト | 内容 |
|---|---|---|
| `test_fifth.py` | `test_duplicate_vertex_is_rejected` | bowtie を `validate_solid` が拒否 |
| `test_fifth.py` | `test_duplicate_coordinates_are_reported_before_non_planarity` | 重複は `non_planar_face` より先に出る（順序を固定） |
| `test_api_server.py` | `test_bowtie_returns_422_duplicate_vertex` | API が 422 `duplicate_vertex` |
| `test_api_server.py` | `test_duplicate_vertex_is_a_distinguishable_type` | `not_connected` と混ざらない（クライアントが誤誘導されない） |
| `test_api_server.py` | `test_cubes_sharing_only_a_vertex_are_duplicate_vertex_not_disconnected` | 共有頂点版（§3 の旧 fixture） |

計画の「+1」に対して **+5**。理由は2つ:
- 核心の検査と API への写像を**両方**カバーした（計画は API のみ明記）
- §3 の fixture 修正で失われるカバレッジを別テストとして残した

---

## 5. spec の更新

- **4.1 の表**に `duplicate_vertex` の行を追加
- **4.2 のリスト**に「頂点の重複と面の非連結は別物。面を削るでは直らない」を追記

クライアントが `not_connected` に取り違えて「面を削れ」と案内することを防ぐため。

---

## 6. 結果

```
pytest -q  →  263 passed in 379.92s      （258 → +5）
mypy --strict →  15 errors                （±0）
```

| 完了条件 | 結果 |
|---|---|
| bowtie が 422 `duplicate_vertex` | ✅ 実測・テストで確認 |
| `spec 4.1` の表と `4.2` のリストを更新 | ✅ |
| test 数 = PR5 + 1 | **+5**（§4 の理由） |

---

## 7. 次の PR への引き継ぎ

- **PR7（tooling）**: mypy 残り 15 件を 0 にする。内訳は
  `dict`/`list`/`tuple` の type-arg と `calculate_nets` の戻り値注釈。
  `pyproject.toml` に `[tool.mypy]` を追加し、`.gitignore` に
  `.pytest_cache/` `.mypy_cache/` `.ruff_cache/` を足す。
- PR6 は任意扱いだったが完了した。PR8 の (b) 部品
  （`duplicate_vertex` の spec 追記）も本 PR で済んでいる。