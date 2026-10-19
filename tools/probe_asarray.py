"""窓（2048 本の全域木）を numpy 配列に落とす方法の比較。

`_batch_canonical_keys` は毎窓 `np.asarray(window, dtype=np.int64)` で
(n, F-1, 2) の辺配列を作る。これが署名フェーズの約 7%（実測 0.30s / 4.0s）を
占めるので、安い経路があるか見る。順序の偏りを避けるため交互に回す。

    uv run python tools/probe_asarray.py stairs
"""

import itertools
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth

VARIANTS = ("asarray", "fromiter", "flat_fromiter", "array_copy")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from bench_ladder import SOLIDS

    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)
    n_faces = len(faces)
    n_edges = n_faces - 1

    trees = list(islice(fifth.enumerate_spanning_trees(edges, n_faces), 2048))
    n = len(trees)
    flat_ints = [x for tree in trees for e in tree for x in e]
    flat_iter = lambda: iter(flat_ints)  # noqa: E731

    def f_asarray():  # type: ignore[no-untyped-def]
        return np.asarray(trees, dtype=np.int64)

    def f_fromiter():  # type: ignore[no-untyped-def]
        it = itertools.chain.from_iterable(itertools.chain.from_iterable(trees))
        return np.fromiter(it, dtype=np.int64, count=n * n_edges * 2).reshape(n, n_edges, 2)

    def f_flat_fromiter():  # type: ignore[no-untyped-def]
        return np.fromiter(flat_iter(), dtype=np.int64, count=n * n_edges * 2).reshape(n, n_edges, 2)

    buf = np.empty((n, n_edges, 2), dtype=np.int64)

    def f_array_copy():  # type: ignore[no-untyped-def]
        return np.array(trees, dtype=np.int64)

    funcs = {"asarray": f_asarray, "fromiter": f_fromiter,
             "flat_fromiter": f_flat_fromiter, "array_copy": f_array_copy}

    ref = f_asarray()
    for k, fn in funcs.items():
        got = fn()
        if not np.array_equal(got, ref):
            print(f"  ⚠️ {k} が一致しない")

    best = dict.fromkeys(VARIANTS, float("inf"))
    for _ in range(5):
        for k in VARIANTS:
            t0 = time.perf_counter()
            funcs[k]()
            dt = time.perf_counter() - t0
            best[k] = min(best[k], dt)
    print(f"{name}: window={n} trees, F-1={n_edges}")
    for k in VARIANTS:
        print(f"  {k:14s} {best[k] * 1e6:8.1f} us   ({best[k] / n * 1e9:6.0f} ns/tree)")
    del buf


if __name__ == "__main__":
    main()
