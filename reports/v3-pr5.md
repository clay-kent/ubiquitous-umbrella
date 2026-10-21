# v3 PR5 報告書 — 死んだコードの削除

**基準**: PR4 終了時 = 258 passed、mypy --strict = 15 errors

---

## 1. 削除したもの（計画の 12 箇所）

### Group A — 到達不能な失敗モードがない削除

| # | 場所 | 内容 |
|---|---|---|
| 1 | `api_server` | `except HTTPException: raise` — HTTPException は ValueError ではないので下の句に飲まれない |
| 2 | `fifth._dihedral_deg` | `if norm_t == 0.0: return 0.0` — 長さ0の辺は `zero_length_edge` が先に弾く |
| 3 | `fifth.face_geometries` | `if edge is not None:` — 親子関係は全域木から来るので親は必ず隣接 |
| 4 | `fifth.net_metrics` | `if edge is not None:` — 同上 |
| 5 | `fifth._glue_arrays` | `if edge is None: continue` — 同上 |
| 6 | `fifth.layout_tree` | `isinstance(rings, Polygon)` — 面数は 4 以上なので常に配列 |
| 7 | `fifth.validate_solid` | `scale = ... or 1.0` — 全頂点が原点なら `zero_length_edge` が先に弾く |
| 8 | `fifth` | `__main__` ブロックまるごと |
| 9 | `fifth` | `generate_unfolding_nets` |
| 10 | `fifth` | `import io` / `import sys` |

（`layout_tree` の `if edge is None: continue` は **PR2 で既に削除済み**。
`shared_of[(u, v)]` の直接参照に置き換えたときになくなった。）

### Group B — 守護を消すと例外になる削除

| 場所 | 消した守護 | 消した後 |
|---|---|---|
| `fifth.orient_faces_outward` | `if len(holders) != 2: continue` | タプル展開の `ValueError` |
| `fifth.layout_tree` | `if edge is None: continue`（PR2 で削除済） | `shared_of[(u, v)]` の `KeyError` |

どちらも **前条件を破ったときに黙るのではなく例外を投げる**。
例外クラスは増やしていない。

---

## 2. 契約は「検査」ではなく「文書」にした

計画の指示どおり、docstring に**前条件と伝播する例外名**を書いた。

### `orient_faces_outward`

> **前条件**: 呼び出し側は先に `validate_solid` を通していること。
> 各辺がちょうど2面に共有される保証はそこに閉じている。
> これを破った入力（境界あり・非多様体）は下のタプル展開が
> `ValueError: too many values to unpack` になる。黙って無視しない。
> 非可向きの立体は従来どおり `SolidError("non_orientable")`。

### `layout_tree`

（PR2 で削除済み。共用辺の取得は `shared_of[(u, v)]` の直接参照で、
前条件違反は `KeyError`。配置ループの `if edge is None: continue` は
共有しない。）

`assert` は使っていません。`python -O` で消えるためです。

---

## 3. 契約テストを 1 本追加

Group B の削除は「例外が出る」ことが契約なので、それを固定する。

```python
def test_orient_faces_outward_rejects_boundary_input_loudly():
    v, f = cube()
    open_faces = f[:4]                 # 側面を1枚外すと境界になる
    with pytest.raises(ValueError, match="共有"):
        fifth.validate_solid(v, open_faces)      # validate_solid は弾く
    with pytest.raises(ValueError):
        fifth.orient_faces_outward(v, open_faces)  # 飛ばしても黙らない
```

削除した `test_generate_unfolding_nets_end_to_end`（`generate_unfolding_nets` が消えた）と
差し引き **±0**。

---

## 4. ⚠️ 削除が mypy エラーを 1 件増やした（ただし最終的に 0 件）

`isinstance(rings, Polygon)` を消した副作用で、mypy の型絞り込みがなくなり
`list()` に `Polygon | ndarray` を渡すため型エラーが出た。

```
fifth.py:545: error: Argument 1 to "list" has incompatible type
                      "Polygon | ndarray"; expected "Iterable[Polygon]"
```

**実行時の保証は face が 4 個以上なので常に配列**というもので、これは静的には
証明できない。`cast` で明示的に型消去する形にした（実行時チェックは増やさない）。

```python
built = cast(list[Polygon], shapely.polygons(shapely.linearrings(np.stack(coords))))
return list(built)
```

`cast` は PR7 の mypy 0 件目標に必要な工作なので、ここで入れた。

---

## 5. 1 件だけ計画に無いガードを残した

| 場所 | 内容 | 判断 |
|---|---|---|
| `fifth._fold_axis_2d` | `ext = ext or 1.0` | **残す** |

`_fold_axis_2d` は PR2 で自分が追加した関数で、計画の 12 箇所のリストは
PR2 より前に書かれたものなので載っていない。

`ext` が 0 になるのは全ポリゴンの外接矩形が退化成している場合で、
これはあり得ない。ただし 0 のままだと次の行 `pt[0] / ext` が
`ZeroDivisionError` になる。削除すると**エラーが 1 つ分かりにくいものになる**だけで、
到達不能性をカバーする効果はない。残した。

---

## 6. 結果

```
pytest -q  →  258 passed in 133.94s
mypy --strict →  15 errors      （15 → ±0）
```

| 完了条件 | 結果 |
|---|---|
| `fifth.py` が新しい例外クラスを定義しない | ✅ `SolidError` のまま |
| `layout_tree` の `ValueError` が健在で面番号を名指しする | ✅ PR2 で維持、`test_layout_tree_rejects_a_non_spanning_tree` |
| `python fifth.py` が動かない（意図） | ✅ exit code 0 かつ無出力 |
| 12 箇所の削除 | ✅ 残存 site 0（スクリプトで機械確認） |

`fifth.py` は 1082 行 → **998 行**（PR3 終了時の 1082 → PR5 で 998）。
PR5 単体は 84 行削減。

---

## 7. 次の PR への引き継ぎ

- **PR6（`duplicate_vertex`）**: `validate_solid` に
  `np.unique(vertices, axis=0)` 比較を追加。bowtie（2つの立方体が頂点1つ共有）は
  既に 422 `not_connected` で弾かれているので、**受理範囲は変わらず type と
  message が変わるだけ**。「隔離ホールの閉鎖」と書くのは誇大。
- **PR7（tooling）**: mypy 残り 15 件を 0 にする。内訳は
  `dict`/`list`/`tuple` の type-arg と `calculate_nets` の戻り値注釈。
  PR5 で消した型が一部これで確定した。
- `prepare_layout` の戻り値が素の `dict` のままだ（`LayoutContext` は使わない判断）。