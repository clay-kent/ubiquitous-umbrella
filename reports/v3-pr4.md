# v3 PR4 報告書 — VF2 と catch 句の削除

**基準**: PR3 終了時 = 254 passed、mypy --strict = 26 errors

---

## 1. 何を消したか

| 削除対象 | 場所 |
|---|---|
| `build_ribbon_graph` | `fifth.py` |
| `is_isomorphic_net` | `fifth.py` |
| `reverse_ribbon_graph` | `fifth.py` |
| `_ribbon_invariant` | `fifth.py` |
| `graph_pair` / `buckets` / stage-2 ブロック | `iter_unique_nets` 内 |
| `get_unique_nets_by_ribbon_graph` | `fifth.py` |
| `except ValueError: continue` | `iter_unique_nets` のループ全体 |
| `import networkx as nx` | `fifth.py`（`fifth.py` に `nx.` が 0 件） |

`iter_unique_nets` は 2 段構えから **1 段構成**になった。正規形だけで潰す。

```python
for tree in all_trees:
    sig = _net_signature(ctx, tree)
    mirror_sig = _net_signature(mirror_ctx, tree)
    key = min(sig, mirror_sig)
    if key in exact_seen:
        continue
    exact_seen.add(key)
    polys = layout_tree(...)
    if not is_valid_layout(polys):
        continue
    yield tree, polys
```

---

## 2. ⚠️ 本当の原因は catch 句だった

計画の要点どおり `except ValueError` を削除したが、
**これが削除の理由ではなく、削除の根據**である。

`iter_unique_nets` はループ全体を `except ValueError: continue` で包んでいた。
これは「ループ中の**あらゆる** ValueError を『この木は捨てる』と解釈する」という意味だった。

予算超過は元 `None` を返していたので、これをそのまま SolidError に変えても
**この1行があると握り潰される**。つまり:

| 状態 | 結果 |
|---|---|
| 予算上限 + `except ValueError` あり | 200 で**誤った一意数**（黙った失敗） |
| 予算上限 + `except ValueError` なし | 422 `search_budget_exceeded`（見える失敗） |

句が無ければ `None` の戻り値も必然的に消えるので、2 つは同じ変更の一部。

### 実測（予算の余裕）

| 立体 | 最大仕事量 | `SEARCH_BUDGET` に対する余裕 |
|---|---|---|
| prism5 | 126 | 1587x |
| prism6 / lshape | 数千 | 1500x 以上 |

余裕がありすぎることが「フォールバックは死んでいる」証明なので、
唯一の停止条件は中断。`test_budget_headroom_is_large` で固定した。

---

## 3. 例外クラスは増やしていない

```python
if budget[0] > SEARCH_BUDGET:
    raise SolidError(
        "search_budget_exceeded",
        "展開図の同型判定が計算量上限を超えました。面の数を減らしてください。",
    )
```

既存の `except fifth.SolidError as exc: raise _error(exc.code, str(exc))` が
そのまま 422 に変換する。**新しい例外型 0 個**。

エンドツーエンドで 422 が返ることを
`test_search_budget_exceeded_returns_a_422_with_a_branchable_type` で確認済み:

```
POST /api/net/calculate  →  422
detail[0] = {"type": "search_budget_exceeded", "loc": ["body"], "msg": "..."}
```

---

## 4. 安全ネットを先に広げた（削除より前）

計画の必須手順どおり、**削除を適用する前に**
`test_unique_nets_geometry.py` に lshape と prism6 を追加した。

| 立体 | 全域木 | 一意数 | 走査時間 |
|---|---|---|---|
| L-shape | 8100 | 1516 | 9.3s |
| hexagonal prism | 8100 | 354 | 7.6s |

従来は最大でも prism5（1805 本）止まりで、**最も重い 2 ケースが未検証**だった。
リボングラフの一意数と、平面図形の合同による独立計数が一致することを
11 立体すべてで確認してから VF2 を削った。

---

## 5. 追加・削除したテスト

| | 数 | 内容 |
|---|---|---|
| 削除 | −2 | `test_canonical_form_agrees_with_vf2_isomorphism` / `test_get_unique_nets_by_ribbon_graph_returns_spanning_trees` |
| 追加 | +2 | `test_unique_nets_geometry.py` の lshape / prism6 |
| 追加 | +1 | `test_errors_inside_the_tree_loop_propagate` — ループ中の任意の `ValueError` が**伝播する**ことを固定 |
| 追加 | +1 | `test_search_budget_exceeded_raises_a_named_solid_error` |
| 追加 | +1 | `test_budget_headroom_is_large` |
| 追加 | +1 | `test_search_budget_exceeded_returns_a_422_with_a_branchable_type` |

`test_errors_inside_the_tree_loop_propagate` は「正常系」ではなく
**回帰のガード**。`_net_signature` に例外を注入して、握り潰されないことを確認する。
これが無いと将来また `except ValueError` を書き戻す。

---

## 6. 結果

```
pytest -q  →  258 passed in 279.85s
mypy --strict →  15 errors        （26 → -11）
```

| 完了条件 | 結果 |
|---|---|
| `nx.` が `fifth.py` に 0 件 | ✅ AST で確認 |
| `iter_unique_nets` に `except ValueError` が無い | ✅ AST で確認（Try ノード 0 / handler 0） |
| `networkx` が dev 依存のみ | ✅ `[dependency-groups].dev` に移動 |
| spec / README の「networkx (VF2)」記述を削除 | ✅ |
| `test_unique_nets_geometry.py` に lshape と prism6 | ✅ |
| ループ中の ValueError が伝播することをテスト | ✅ |
| 例外型を増やさない | ✅ 既存の `SolidError` を再利用 |

---

## 7. 次の PR への引き継ぎ

- **PR5 は死んだコード削除**。`fifth.py` の `__main__` ブロック
  （`generate_unfolding_nets` / `import io` / `import sys`）と
  `orient_faces_outward` の `if len(holders) != 2: continue`、
  `_glue_arrays` の `if edge is None: continue`、
  `isinstance(rings, Polygon)`、`scale = ... or 1.0` が対象。
  併せて `_dihedral_deg` の `if norm_t == 0.0: return 0.0` と
  `api_server` の `except HTTPException: raise`。
- `generate_unfolding_nets` を消すと `test_generate_unfolding_nets_end_to_end` が
  落ちる。PR5 で 1 件削除。
- PR5 の `api_server` 側 dead code は PR2 で `fifth.py` に移ったので、
  対象が `fifth.py` に移っている点に注意。
- PR7 の mypy 0 件の残り 15 件は次の内訳のはず:
  `dict`/`list` の type-arg（`prepare_layout` の戻り値など）と
  `calculate_nets` の戻り値注釈。PR6 の後着手する。