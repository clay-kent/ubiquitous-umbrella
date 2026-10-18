"""頂点投影そのものを「明示的な左結合」でバッチ化できるか試す。

`(P * AX).sum(axis=1)` は GEMV (`pts @ axis`) と丸めが違った（却下）。
一方、成分ごとに明示して

    P[:,0]*AX[:,0] + P[:,1]*AX[:,1] + P[:,2]*AX[:,2]

と書けば左結合の順で足される。3要素内積 (`a @ b`) がスカラ式と一致したのと
同じ理屈で GEMV と一致するかもしれない。一致すれば投影が
「face 数ぶん」から「1回」に落ちて大きな削減になる。

    uv run python tools/check_projection_explicit_bits.py
"""

import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

rng = np.random.default_rng(0)
F = 10
counts = [4, 4, 4, 4, 6, 4, 4, 6, 4, 4]
M = sum(counts)


def main() -> None:
    n_trials = 20000
    bad_explicit = 0
    bad_fused = 0
    for _ in range(n_trials):
        u_x = rng.standard_normal(3)
        u_y = rng.standard_normal(3)
        origin = rng.standard_normal(3)
        rot = rng.standard_normal((F, 3, 3))
        off = rng.standard_normal((F, 3))
        P = rng.standard_normal((M, 3))

        axes_x = np.einsum("b,fba->fa", u_x, rot)
        axes_y = np.einsum("b,fba->fa", u_y, rot)
        S = off - origin
        cs_x = (S * u_x).sum(axis=1)
        cs_y = (S * u_y).sum(axis=1)

        idx = np.repeat(np.arange(F), counts)
        AX = axes_x[idx]
        AY = axes_y[idx]
        CX = cs_x[idx]
        CY = cs_y[idx]

        # 参照: face ごとの GEMV
        ref_x = np.concatenate(
            [P[o : o + c] @ axes_x[f] + cs_x[f] for f, (o, c) in _off(counts)]
        )
        ref_y = np.concatenate(
            [P[o : o + c] @ axes_y[f] + cs_y[f] for f, (o, c) in _off(counts)]
        )

        # 案1: 成分ごとに明示（左結合）
        ex_x = P[:, 0] * AX[:, 0] + P[:, 1] * AX[:, 1] + P[:, 2] * AX[:, 2] + CX
        ex_y = P[:, 0] * AY[:, 0] + P[:, 1] * AY[:, 1] + P[:, 2] * AY[:, 2] + CY
        if not (np.array_equal(ex_x, ref_x) and np.array_equal(ex_y, ref_y)):
            bad_explicit += 1

        # 案2: (P*AX).sum(axis=1) は以前却下した形（比較用）
        fu_x = (P * AX).sum(axis=1) + CX
        if not np.array_equal(fu_x, ref_x):
            bad_fused += 1

    print(f"explicit differs: {bad_explicit} / {n_trials}")
    print(f"fused(sum) differs: {bad_fused} / {n_trials}")

    # --- 速度 ---
    u_x = rng.standard_normal(3)
    u_y = rng.standard_normal(3)
    origin = rng.standard_normal(3)
    rot = rng.standard_normal((F, 3, 3))
    off = rng.standard_normal((F, 3))
    P = rng.standard_normal((M, 3))
    idx = np.repeat(np.arange(F), counts)

    def current():
        ox, oy, oz = origin.tolist()
        ux0, ux1, ux2 = u_x.tolist()
        uy0, uy1, uy2 = u_y.tolist()
        xs, ys = [], []
        for i in range(F):
            fr = rot[i]
            ax = u_x @ fr
            ay = u_y @ fr
            t0, t1, t2 = off[i].tolist()
            s0, s1, s2 = t0 - ox, t1 - oy, t2 - oz
            pts = P[_off(counts).__next__()[1]] if False else None
            o, c = _slice(i)
            xs.append(P[o : o + c] @ ax + (s0 * ux0 + s1 * ux1 + s2 * ux2))
            ys.append(P[o : o + c] @ ay + (s0 * uy0 + s1 * uy1 + s2 * uy2))
        return np.concatenate(xs), np.concatenate(ys)

    def explicit():
        axes_x = np.einsum("b,fba->fa", u_x, rot)
        axes_y = np.einsum("b,fba->fa", u_y, rot)
        S = off - origin
        cs_x = (S * u_x).sum(axis=1)
        cs_y = (S * u_y).sum(axis=1)
        AX = axes_x[idx]
        AY = axes_y[idx]
        x = P[:, 0] * AX[:, 0] + P[:, 1] * AX[:, 1] + P[:, 2] * AX[:, 2] + cs_x[idx]
        y = P[:, 0] * AY[:, 0] + P[:, 1] * AY[:, 1] + P[:, 2] * AY[:, 2] + cs_y[idx]
        return x, y

    t1 = min(timeit.repeat(current, number=20000, repeat=5)) / 20000
    t2 = min(timeit.repeat(explicit, number=20000, repeat=5)) / 20000
    print(f"current {t1*1e6:7.2f} us   explicit {t2*1e6:7.2f} us   speedup {t1/t2:.2f}x")


def _off(counts):
    o = 0
    for f, c in enumerate(counts):
        yield f, (o, c)
        o += c


_OFFS = None


def _slice(i):
    global _OFFS
    if _OFFS is None:
        _OFFS = [(o, o + c) for _, (o, c) in _off(counts)]
    return _OFFS[i]


if __name__ == "__main__":
    main()
