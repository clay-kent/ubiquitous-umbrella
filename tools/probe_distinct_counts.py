"""`_distinct_counts` の内訳と、代案（ソート種別・数え方）を比べる。

安定判定は「行ごとの異なる色の個数が増えたか」で行うので、毎ラウンド
(行, n_nodes) のソートが走る。ここが重いので、内訳と代案を測る。

    uv run python tools/probe_distinct_counts.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth

ROWS = 2048
NODES = 48
REPS = 300


def bench(label: str, fn) -> None:
    best = float("inf")
    for _ in range(3):
        t0 = time.process_time()
        for _ in range(REPS):
            fn()
        dt = (time.process_time() - t0) / REPS
        best = min(best, dt)
    print(f"{label:44s} {best * 1e6:9.2f} us")


def main() -> None:
    rng = np.random.default_rng(999)
    h = rng.integers(0, 1 << 62, size=(ROWS, NODES), dtype=np.uint64)
    # 実際は「ほとんど別色」なので、重複を少なめにした配列も用意
    few = rng.integers(0, 12, size=(ROWS, NODES), dtype=np.uint64)

    print(f"rows={ROWS} n_nodes={NODES} reps={REPS}")
    srt = h.copy()
    print("--- 内訳 (distinct) ---")
    bench("h.copy()", lambda: h.copy())
    bench("srt.sort(axis=1)  (in-place, quicksort)", lambda: srt.sort(axis=1))
    bench("np.sort(h, axis=1) (quicksort)", lambda: np.sort(h, axis=1))
    bench("np.sort(h, axis=1, kind='stable')", lambda: np.sort(h, axis=1, kind="stable"))
    bench("np.sort(h, axis=1, kind='heapsort')", lambda: np.sort(h, axis=1, kind="heapsort"))
    bench("(srt[:,1:] != srt[:,:-1]).sum(axis=1)", lambda: (srt[:, 1:] != srt[:, :-1]).sum(axis=1))
    bench("np.count_nonzero(srt[:,1:] != srt[:,:-1], 1)", lambda: np.count_nonzero(srt[:, 1:] != srt[:, :-1], axis=1))
    print("--- 全体 ---")
    bench("_distinct_counts(h)", lambda: fifth._distinct_counts(h))
    bench("_distinct_counts(few)", lambda: fifth._distinct_counts(few))

    def no_copy() -> np.ndarray:
        hh = h.copy()
        hh.sort(axis=1)
        return np.asarray(1 + (hh[:, 1:] != hh[:, :-1]).sum(axis=1), dtype=np.int64)

    bench("copy+sort+diff+sum", no_copy)
    print(f"  (参考) 1行あたりの異なり数 = {fifth._distinct_counts(h)[0]}")


if __name__ == "__main__":
    main()
