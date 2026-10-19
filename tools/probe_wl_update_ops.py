"""バッチ WL の1ラウンドを構成する numpy 演算を個別に測る（相対値だけを見る）。

`_batch_stabilize` は「(行, n_nodes) の uint64 配列を数回通す」ループなので、
どの演算が効いているかを見ないと手当ての優先順位が決まらない。
計測は繰り返し回数で稼いで `process_time` の分解能（15.6ms）を超えさせる。

    uv run python tools/probe_wl_update_ops.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth

ROWS = 2048
NODES = 48
REPS = 400


def bench(label: str, fn) -> None:
    best = float("inf")
    for _ in range(3):
        t0 = time.process_time()
        for _ in range(REPS):
            fn()
        dt = (time.process_time() - t0) / REPS
        best = min(best, dt)
    print(f"{label:38s} {best * 1e6:9.2f} us   ({best * 1e9 / (ROWS * NODES):5.3f} ns/elem)")


def main() -> None:
    rng = np.random.default_rng(12345)
    h = rng.integers(0, 1 << 62, size=(ROWS, NODES), dtype=np.uint64)
    nxt = rng.permutation(NODES).astype(np.intp)
    partner = rng.integers(-1, NODES, size=(ROWS, NODES))
    glued = (partner >= 0).astype(np.uint64)
    poself = np.where(partner >= 0, partner, np.arange(NODES)[None, :]).astype(np.intp)
    coef = glued * np.uint64(0x9E3779B97F4A7C15)
    tag = glued * np.uint64(3) + np.uint64(7)
    nh = np.empty_like(h)
    t = np.empty_like(h)
    A = np.uint64(0x2545F4914F6CDD1D)

    print(f"rows={ROWS} n_nodes={NODES} reps={REPS}")
    bench("out = h * A (in-place out=)", lambda: np.multiply(h, A, out=nh))
    bench("t = h[:, nxt] (fancy index)", lambda: h[:, nxt])
    bench("t[:] = h.take(nxt, axis=1)", lambda: h.take(nxt, axis=1))
    bench("t = take_along_axis(h, poself)", lambda: np.take_along_axis(h, poself, axis=1))
    bench("t *= A (in-place)", lambda: np.multiply(t, A, out=t))
    bench("nh += t (in-place)", lambda: np.add(nh, t, out=nh))
    bench("nh += tag (in-place)", lambda: np.add(nh, tag, out=nh))
    bench("out = h * coef (broadcast 2D)", lambda: np.multiply(h, coef, out=nh))
    bench("np.sort(nh, axis=1)", lambda: np.sort(nh, axis=1))
    bench("_distinct_counts(nh)", lambda: fifth._distinct_counts(nh))

    def full_round() -> None:
        np.multiply(h, A, out=nh)
        t2 = h[:, nxt]
        np.multiply(t2, A, out=t2)
        np.add(nh, t2, out=nh)
        t3 = np.take_along_axis(h, poself, axis=1)
        np.multiply(t3, coef, out=t3)
        np.add(nh, t3, out=nh)
        np.add(nh, tag, out=nh)

    bench("full update (8 passes)", full_round)


if __name__ == "__main__":
    main()
