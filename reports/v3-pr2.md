# v3 PR2 報告書 — 幾何を fifth.py へ移動

**基準**: PR1 終了時 = 212 passed / 2 xfailed、mypy --strict = 28 errors

---

## 1. 何をしたか

`api_server.py` にあった幾何計算を `fifth.py` に移した。APIサーバは
「HTTP/JSON の性格を持つ層」、`fifth.py` は「幾何を持つ層」に境界が引けた。

| 関数 | 移動先 | 変更 |
|---|---|---|
| `_face_area_3d` / `_dihedral_deg` / `_pair_to_edge` | `fifth.py` | as-is（名前は計画どおり前置`_`のまま） |
| `_fold_axis_2d` | `fifth.py` | **bbox 正規化の修正つき**（§3） |
| `get_layout_parent_relation` | `layout_parent_relation`（新規） | BFS を `_tree_bfs` に一本化 |
| `_face_geometry` | 分割 | 幾何 → `fifth.face_geometries`、JSON化は `api_server` に残置 |
| `_build_solid` の前処理 | `fifth.parse_vertices`（新規） | エラー整形 `_error` は `api_server` に残置 |

追加した型: `FaceGeometry`（NamedTuple）。純粋な幾何だけをここに置き、
JSON のキー名は持たせない。

---

## 2. 親子関係を 1 つの走査に寄せる（BFS の二重実装をやめる）

計画どおり、`layout_tree` を親マップの唯一のソースにした。ただし
`layout_tree` の戻り値型を変えると呼び出し元（`iter_unique_nets` やテスト）が
全部壊れるので、**走査そのものを共有する**形にした。

```python
_tree_bfs(n_faces, tree_edges) -> (parents, [(親, 子), ...])
```

- `layout_parent_relation` … `parents` だけを返す（公開契約）
- `layout_tree` … `(親, 子)` の列で回転を伝播させる

`pairs` を返すのは `parents[v]` が `int | None` で、リスト添字として
型エラーになるため。走査順そのものを渡す形にした。

### 契約（docstring に明記）

> 根は面0。親は木の無向隣接を**幅優先**に走査し、最初に訪れた未訪問の
> 隣接面を親として割り当てる。根の親は None。

---

## 3. `_fold_axis_2d` の bbox 正規化（計画の想定より低い範囲だった）

計画は「`round(9)` が 1e-9 を下回ると壊れる → 1e-10 で 219/300 誤り」と
書いていた。PR1 の計測で **1e-9 ですでに壊れる**と分かっていた（PR1 報告書 §2.2）。

修正（PR1 報告書の案どおり、**正規化を先に**）:

```python
ext = max(p.bounds[2]-p.bounds[0], p.bounds[3]-p.bounds[1] for p in polys) or 1.0
key = lambda pt: (round(pt[0]/ext, EDGE_LEN_DECIMALS), round(pt[1]/ext, EDGE_LEN_DECIMALS))
```

### 修正後の実測（miss, wrong）vs scale=1

| 立体 | 1e-6 | 1e-9 | 1e-10 | 1e-12 | 1e-14 | 1e-16 |
|---|---|---|---|---|---|---|
| box(1,1,1) | ok | ok | ok | ok | ok | ok |
| box(1,1.5,2) | ok | ok | ok | ok | ok | ok |
| tetra | ok | ok | ok | ok | ok | ok |
| prism3 | ok | ok | ok | ok | ok | LEN! |
| prism4 | ok | ok | ok | ok | ok | ok |
| prism5 | ok | ok | ok | ok | ok | ok |
| pyr4 / pyr5 | ok | ok | ok | ok | ok | ok |

**限界が 1e-8 → 1e-14 に進んだ**。1e-16 の prism3 の `LEN!` は
一意数そのものが変わり始める（数量側の限界）で、双方向とも検証対象外。

---

## 4. 追加したテスト

| ファイル | 内容 | 数 |
|---|---|---|
| `test_api_golden.py` | PR2 着手前の API 応答を golden に保存し、**正規化してから**比較 | 6 |
| `test_fifth.py::test_layout_parent_relation_matches_its_documented_contract` | 契約を書き直した独立実装と突き合わせ | 5 |
| `test_fifth.py::test_layout_tree_and_parent_relation_agree` | 親マップが実際の隣接と整合 | 1 |
| `test_fifth.py::test_layout_tree_rejects_a_non_spanning_tree` | 非全域木は面番号を名指して落ちる | 1 |
| `test_scale_invariance.py::test_fold_axis_is_scale_invariant` | 1e-9 / 1e-10 / 1e-12 × 8立体 | 24 |

### golden（`tests/api_golden_v2.json`）

5立体 × top_n=3 の 200 応答をリファクタ前に保存。比較は
「dict を再帰ソート＋全 float を 9 桁に丸める」正規化をして行う。
**生の JSON 文字列は比較していない**（float の repr や dict の挿入順は
意味を変えずに変動する。バイト一致は偽の失敗を生む）。

さらに `test_golden_covers_the_fields_the_move_could_break` で、
golden が本当に `fold_axis_2d` / `fold_axis_3d` / `dihedral_angle` を
含んでいるか検証している（全部 None の golden だと「互いに null だから一致」
だけで通ってしまうため）。

### 削除したテスト

| テスト | 理由 |
|---|---|
| `test_fold_axis_below_1e_8_is_still_broken`（2件 xfail） | `strict=True` なので修正で XPASS した。PR1 で入れた「直ったら落とせ」の合図どおり削除 |
| `test_fold_axis_is_correct_down_to_1e_8`（5件） | 1e-8 の限定が不要に。上の 24 件に統合 |

---

## 5. 結果

```
pytest -q  →  244 passed in 190.66s
```

| 指標 | PR1 | PR2 | 差 |
|---|---|---|---|
| テスト数 | 214（212+2 xfail） | 244 | +30 |
| mypy --strict | 28 | **27** | **-1** |

内訳（214 → 244）:
```
- 2（xfail 削除）
- 5（1e-8 限定テスト削除）
+24（fold_axis を 1e-9/1e-10/1e-12 × 8立体に拡張）
+ 6（golden）
+ 5（親マップ契約）
+ 1（layout_tree 整合）
+ 1（非全域木）
= 244
```

- **`test_fifth.py` に `import api_server` が 0 件**（計画の完了条件 ✅）
- golden 比較が全5立体で 200 一致（意味的等価性 ✅）

---

## 6. 計画との差分 / 判断

1. **`_dihedral_deg` の `if norm_t == 0.0: return 0.0` を残した。**
   計画の「as-is」に従った。PR5 が死んだコードとして削除する。
   先に消すと PR5 の差分が読めなくなる。

2. **`layout_tree` の戻り値は変えない方式を採った。**
   計画は「`layout_tree` の BFS の副産物」を親マップとして使うとしたが、
   戻り値型を変えると `iter_unique_nets` と全テストのタプル分解が壊れる。
   走査を共有する形で同じ目的（定義を 1 つに）を達成した。

3. **前置 `_` のまま移動した。**
   計画が「as-is」と書いているため。跨モジュールで使うのは
   スタイル的に微妙だが、公開名に変えると差分が読めなくなる。
   PR3 で `make_nets` から使うことになるが `fifth._face_area_3d` のままで、
   公開名は付け替えていない。PR7 で公開化が妥当。

4. **`layout_tree` の `if edge is None: continue` が消えた。**
   共有辺の取得が `shared_of[(u, v)]` の直接参照になったため。
   前条件違反時は `KeyError` になる。PR5 Group B の予定と一致。

5. **`test_fifth.py` の既存コメント 1 文字を修正**（`접` → `接`）。
   ハングルの混入。対象ファイル全体が clean であることを確認。

---

## 7. 次の PR への引き継ぎ

- PR3 は `make_nets.py` の `perim = 2.0 * (w + h)` を
  `fifth._face_area_3d` 相当の共通定義に置き換える。指標は 1 実装にまとめる。
- PR4 の VF2 削除前に、`test_unique_nets_geometry.py` に
  **lshape と prism6 を追加**すること（計画の必須手順）。
- `fifth.py` に残った mypy 27 件のうち、VF2 関連（`DiGraph` の type-arg など）は
  PR4 で削除される。PR7 の「0 件」目標は PR4/PR5 後に初めて到達可能になる。