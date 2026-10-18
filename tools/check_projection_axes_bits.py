"""投影ループの「軸と定数項だけ」を前計算・バッチ化する案のビット一致と速度。

頂点そのものの投影 (pts @ axis) はバッチ化すると丸めが変わる（却下済み）が、
軸 (u @ R) と平行移動の定数項はバッチで作っても**ビット一致**する。
回転・移動を最初から numpy 配列に貯めれば list->array 変換も要らない。

    uv run python tools/check_projection_axes_bits.py
"""

import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

rng = np.random.default_rng(0)
F = 10
counts = [4, 4, 4, 4, 6, 4, 4, 6, 4, 4]


def main() -> None:
    n_trials = 20000
    bad_axes = 0
    bad_coords = 0
    for _ in range(n_trials):
        u_x = rng.standard_normal(3)
        u_y = rng.standard_normal(3)
        origin = rng.standard_normal(3)
        rot = rng.standard_normal((F, 3, 3))
        off = rng.standard_normal((F, 3))
        face_pts = [rng.standard_normal((c, 3)) for c in counts]

        # --- 現行（1 face ずつ）---
        ox, oy, oz = origin.tolist()
        ux0, ux1, ux2 = u_x.tolist()
        uy0, uy1, uy2 = u_y.tolist()
        xs0, ys0 = [], []
        for i in range(F):
            fr = rot[i]
            ax = u_x @ fr
            ay = u_y @ fr
            t0, t1, t2 = off[i].tolist()
            s0, s1, s2 = t0 - ox, t1 - oy, t2 - oz
            pts = face_pts[i]
            xs0.append(pts @ ax + (s0 * ux0 + s1 * ux1 + s2 * ux2))
            ys0.append(pts @ ay + (s0 * uy0 + s1 * uy1 + s2 * uy2))

        # --- 新案（軸と定数項をバッチ）---
        axes_x = np.einsum("b,fba->fa", u_x, rot)
        axes_y = np.einsum("b,fba->fa", u_y, rot)
        S = off - origin
        cs_x = (S * u_x).sum(axis=1)
        cs_y = (S * u_y).sum(axis=1)
        xs1, ys1 = [], []
        for i in range(F):
            pts = face_pts[i]
            xs1.append(pts @ axes_x[i] + cs_x[i])
            ys1.append(pts @ axes_y[i] + cs_y[i])

        if not (
            np.array_equal(axes_x, np.array([u_x @ rot[i] for i in range(F)]))
            and np.array_equal(axes_y, np.array([u_y @ rot[i] for i in range(F)]))
        ):
            bad_axes += 1
        if not (
            all(np.array_equal(a, b) for a, b in zip(xs0, xs1))
            and all(np.array_equal(a, b) for a, b in zip(ys0, ys1))
        ):
            bad_coords += 1

    print(f"axes differ:   {bad_axes} / {n_trials}")
    print(f"coords differ: {bad_coords} / {n_trials}")

    # --- 速度 ---
    u_x = rng.standard_normal(3)
    u_y = rng.standard_normal(3)
    origin = rng.standard_normal(3)
    rot = rng.standard_normal((F, 3, 3))
    off = rng.standard_normal((F, 3))
    face_pts = [rng.standard_normal((c, 3)) for c in counts]

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
            pts = face_pts[i]
            xs.append(pts @ ax + (s0 * ux0 + s1 * ux1 + s2 * ux2))
            ys.append(pts @ ay + (s0 * uy0 + s1 * uy1 + s2 * uy2))
        return np.concatenate(xs), np.concatenate(ys)

    def new():
        axes_x = np.einsum("b,fba->fa", u_x, rot)
        axes_y = np.einsum("b,fba->fa", u_y, rot)
        S = off - origin
        cs_x = (S * u_x).sum(axis=1)
        cs_y = (S * u_y).sum(axis=1)
        xs, ys = [], []
        for i in range(F):
            pts = face_pts[i]
            xs.append(pts @ axes_x[i] + cs_x[i])
            ys.append(pts @ axes_y[i] + cs_y[i])
        return np.concatenate(xs), np.concatenate(ys)

    t1 = min(timeit.repeat(current, number=20000, repeat=5)) / 20000
    t2 = min(timeit.repeat(new, number=20000, repeat=5)) / 20000
    print(f"current {t1*1e6:7.2f} us   new {t2*1e6:7.2f} us   speedup {t1/t2:.2f}x")


if __name__ == "__main__":
    main()
