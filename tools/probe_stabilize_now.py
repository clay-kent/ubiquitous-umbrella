"""`_batch_stabilize` の現行コードの内訳（更新 / 色数 / 間引き / 初期化）。

Windows の `time.process_time` は刻みが 15.6ms なのでラウンド単位では使えない。
ここは壁時計で back-to-back に部位を測り、比を読む（負荷が乗っても比は保たれる）。

    uv run python tools/probe_stabilize_now.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import fifth
from bench_ladder import SOLIDS


class Acc:
    def __init__(self) -> None:
        self.init = 0.0
        self.update = 0.0
        self.counts = 0.0
        self.counts_partial = 0.0
        self.prune = 0.0
        self.rounds = 0
        self.row_rounds = 0
        self.full_rows = 0
        self.calls = 0
        self.mismatch = 0


ACC = Acc()


def timed_batch_stabilize(ctx, partner, flip):  # type: ignore[no-untyped-def]
    ACC.calls += 1
    t0 = time.perf_counter()
    n_nodes = ctx.n_nodes
    glued = (partner >= 0).astype(np.uint64)
    partner_or_self = np.where(partner >= 0, partner, np.arange(n_nodes)[None, :])
    partner_coef = glued * fifth._WL_C
    tag = glued * fifth._WL_D + flip.astype(np.uint64) * fifth._WL_E
    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    h = eid[None, :] * fifth._WL_E0 + tag
    nxt = ctx.np_nxt
    n_rows = len(h)
    row_ids = np.arange(n_rows, dtype=np.int64)
    out_colors = np.empty((n_rows, n_nodes), dtype=np.uint64)
    out_counts = np.empty(n_rows, dtype=np.int64)
    t_buf = np.empty((n_rows, n_nodes), dtype=np.uint64)
    t2_buf = np.empty((n_rows, n_nodes), dtype=np.uint64)
    nh_a = np.empty((n_rows, n_nodes), dtype=np.uint64)
    nh_b = np.empty((n_rows, n_nodes), dtype=np.uint64)
    flat_buf = np.empty((n_rows, n_nodes), dtype=np.intp)
    use_a = True
    row_off = np.arange(n_rows, dtype=np.intp)[:, None] * n_nodes
    counts = None
    rnd = 0
    ACC.init += time.perf_counter() - t0

    while True:
        rnd += 1
        n = len(h)
        ACC.rounds += 1
        ACC.row_rounds += n
        nh = (nh_a if use_a else nh_b)[:n]
        use_a = not use_a
        t = t_buf[:n]
        t2 = t2_buf[:n]
        t0 = time.perf_counter()
        np.multiply(h, fifth._WL_A, out=nh)
        np.copyto(t, h[:, nxt])
        np.multiply(t, fifth._WL_B, out=t)
        np.add(nh, t, out=nh)
        np.add(partner_or_self, row_off[:n], out=flat_buf[:n])
        np.copyto(t2, h.reshape(-1)[flat_buf[:n].reshape(-1)].reshape(n, n_nodes))
        np.multiply(t2, partner_coef, out=t2)
        np.add(nh, t2, out=nh)
        np.add(nh, tag, out=nh)
        ACC.update += time.perf_counter() - t0
        if rnd <= fifth._WL_MIN_ROUNDS:
            h = nh
            if rnd < fifth._WL_MIN_ROUNDS:
                continue
            t0 = time.perf_counter()
            counts = fifth._distinct_counts(nh)
            ACC.counts += time.perf_counter() - t0
            continue
        assert counts is not None
        t0 = time.perf_counter()
        new_counts = fifth._distinct_counts(nh)
        ACC.counts += time.perf_counter() - t0
        # 「全色が別」に達した行はこれ以上細分できないので、色数を数えずとも
        # 次のラウンドで必ず done になる。その行を除いて数えた場合と比べる。
        full = counts == n_nodes
        part = ~full
        ACC.full_rows += int(full.sum())
        if part.any():
            t0 = time.perf_counter()
            trimmed = fifth._distinct_counts(nh[part])
            ACC.counts_partial += time.perf_counter() - t0
            if not np.array_equal(trimmed, new_counts[part]):
                ACC.mismatch += 1
        t0 = time.perf_counter()
        done = new_counts == counts
        out_colors[row_ids[done]] = nh[done]
        out_counts[row_ids[done]] = new_counts[done]
        if done.all():
            ACC.prune += time.perf_counter() - t0
            break
        keep = ~done
        h = nh[keep]
        partner_or_self = partner_or_self[keep]
        partner_coef = partner_coef[keep]
        tag = tag[keep]
        counts = new_counts[keep]
        row_ids = row_ids[keep]
        ACC.prune += time.perf_counter() - t0

    return out_colors, out_counts


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    fifth._batch_stabilize = timed_batch_stabilize
    t0 = time.perf_counter()
    n = 0
    for _t, _p in fifth.iter_unique_nets(v, faces, iter(trees), fifth.compute_normals(v, faces)):
        n += 1
    total = time.perf_counter() - t0

    parts = {"init": ACC.init, "update": ACC.update, "counts": ACC.counts, "prune": ACC.prune}
    print(f"{name}: nets={n} pipeline={total:.3f}s calls={ACC.calls}")
    print(f"  rounds={ACC.rounds} row_rounds={ACC.row_rounds} "
          f"avg_rows={ACC.row_rounds / max(ACC.rounds, 1):.0f}")
    for k, s in parts.items():
        print(f"  {k:8s} {s:7.3f}s  ({s / total * 100:5.1f}% of pipeline)")
    print(f"  {ACC.update / max(ACC.row_rounds, 1) * 1e9:6.0f} ns/row-round (update)")
    print(f"  全色別の行: {ACC.full_rows} (row-round のうち "
          f"{ACC.full_rows / max(ACC.row_rounds, 1) * 100:.1f}%)  mismatch={ACC.mismatch}")
    print(f"  色数を全部の行で: {ACC.counts:7.3f}s / 部分細分行だけ: {ACC.counts_partial:7.3f}s")


if __name__ == "__main__":
    main()
