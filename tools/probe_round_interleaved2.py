"""WL 更新1ラウンドの実装候補を交互に測る（第2版）。

`np.take(..., out=)` を使うと「確保」と「コピー」の両方を消せる。
2 次元 gather を axis=1 で out= できるかも合わせて測る。

    uv run python tools/probe_round_interleaved2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROWS = 2048
NODES = 48
REPS = 60
ROUNDS = 8


def main() -> None:
    rng = np.random.default_rng(7)
    h = rng.integers(0, 1 << 62, size=(ROWS, NODES), dtype=np.uint64)
    nxt = rng.permutation(NODES).astype(np.intp)
    partner = rng.integers(-1, NODES, size=(ROWS, NODES))
    glued = (partner >= 0).astype(np.uint64)
    poself = np.where(partner >= 0, partner, np.arange(NODES)[None, :]).astype(np.intp)
    coef = glued * np.uint64(0x9E3779B97F4A7C15)
    tag = glued * np.uint64(3) + np.uint64(7)
    flat = (poself + np.arange(ROWS, dtype=np.intp)[:, None] * NODES).ravel()
    A = np.uint64(0x2545F4914F6CDD1D)
    Bv = np.uint64(0xD6E8FEB86659FD93)
    h1d = h.reshape(-1)

    nh_b = np.empty_like(h)
    t_b = np.empty_like(h)
    t2_b = np.empty_like(h)
    out_b = np.empty_like(h)

    def v1() -> None:
        nh = h * A
        t = h[:, nxt]
        t *= Bv
        nh += t
        t2 = np.take_along_axis(h, poself, axis=1)
        t2 *= coef
        nh += t2
        nh += tag
        out_b[:] = nh

    def v5() -> None:
        np.multiply(h, A, out=nh_b)
        np.copyto(t_b, h[:, nxt])
        np.multiply(t_b, Bv, out=t_b)
        np.add(nh_b, t_b, out=nh_b)
        np.copyto(t2_b, h1d[flat].reshape(ROWS, NODES))
        np.multiply(t2_b, coef, out=t2_b)
        np.add(nh_b, t2_b, out=nh_b)
        np.add(nh_b, tag, out=nh_b)
        out_b[:] = nh_b

    def v6() -> None:
        np.multiply(h, A, out=nh_b)
        np.copyto(t_b, h[:, nxt])
        np.multiply(t_b, Bv, out=t_b)
        np.add(nh_b, t_b, out=nh_b)
        np.take(h1d, flat, out=t2_b.reshape(-1))
        np.multiply(t2_b, coef, out=t2_b)
        np.add(nh_b, t2_b, out=nh_b)
        np.add(nh_b, tag, out=nh_b)
        out_b[:] = nh_b

    def v9() -> None:
        np.multiply(h, A, out=nh_b)
        np.take(h, nxt, axis=1, out=t_b)
        np.multiply(t_b, Bv, out=t_b)
        np.add(nh_b, t_b, out=nh_b)
        np.take(h1d, flat, out=t2_b.reshape(-1))
        np.multiply(t2_b, coef, out=t2_b)
        np.add(nh_b, t2_b, out=nh_b)
        np.add(nh_b, tag, out=nh_b)
        out_b[:] = nh_b

    variants = {"V1 (現行)": v1, "V5 flat+copyto": v5, "V6 flat+take out=": v6,
                "V9 V6+both take out=": v9}
    best = {k: float("inf") for k in variants}
    for _ in range(ROUNDS):
        for label, fn in variants.items():
            t0 = time.process_time()
            for _ in range(REPS):
                fn()
            dt = (time.process_time() - t0) / REPS
            best[label] = min(best[label], dt)
    for label, dt in best.items():
        print(f"{label:26s} {dt * 1e6:9.2f} us   {dt * 1e9 / ROWS:7.1f} ns/row")

    ref = h * A
    tt = h[:, nxt]
    tt *= Bv
    ref += tt
    t2 = np.take_along_axis(h, poself, axis=1)
    t2 *= coef
    ref += t2
    ref += tag
    for label, fn in variants.items():
        fn()
        print(f"  {label:26s} == V1: {np.array_equal(ref, out_b)}")


if __name__ == "__main__":
    main()
