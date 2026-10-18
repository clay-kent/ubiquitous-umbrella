"""投影ループを「全faceまとめて」にできるか（ビット一致 + 速度）を試す。

現状は face ごとに
    axis_x = u_x @ rot[i]
    x = pts_i @ axis_x + cx_i
を回す。face 数ぶんの numpy ディスパッチが乗るので、全 face の頂点を
1 本の (M,3) に積んで

    axes_x = einsum('b,fba->fa', u_x, rots)           # (F,3)
    x = (P * repeat(axes_x, counts)).sum(axis=1) + repeat(cx, counts)

で1回にまとめたい。**丸め順が変わる可能性があるので先にビット一致を確認する**
（Cycle 5 / 12 の教訓）。

    uv run python tools/check_projection_batch_bits.py
"""

import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

rng = np.random.default_rng(0)
F = 10
counts = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def scalar_dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def main() -> None:
    # --- ビット一致 ---
    diffs = {
        "einsum axes_x vs u@R": 0,
        "P*AX sum vs pts@axis": 0,
        "(S*u).sum vs scalar cx": 0,
    }
    for _ in range(20000):
        u = rng.standard_normal(3)
        rots = rng.standard_normal((F, 3, 3))
        # 1) 軸
        axes = np.einsum("b,fba->fa", u, rots)
        ref_axes = np.array([u @ rots[f] for f in range(F)])
        if not np.array_equal(axes, ref_axes):
            diffs["einsum axes_x vs u@R"] += 1
        # 2) 頂点投影
        P = rng.standard_normal((sum(counts), 3))
        AX = np.repeat(ref_axes, counts, axis=0)
        got = (P * AX).sum(axis=1)
        exp = np.concatenate([P[o : o + c] @ ref_axes[f] for f, (o, c) in _offsets(counts)])
        if not np.array_equal(got, exp):
            diffs["P*AX sum vs pts@axis"] += 1
        # 3) 定数項
        S = rng.standard_normal((F, 3))
        cs = (S * u).sum(axis=1)
        ref_cs = np.array([scalar_dot3(S[f], u) for f in range(F)])
        if not np.array_equal(cs, ref_cs):
            diffs["(S*u).sum vs scalar cx"] += 1

    for k, v in diffs.items():
        print(f"{k:30s} differs {v} / 20000")

    # --- 速度（stairs 相当: F=10, 各face 4頂点 → M=40）---
    u_x = rng.standard_normal(3)
    rots = rng.standard_normal((F, 3, 3))
    offs = rng.standard_normal((F, 3))
    origin = rng.standard_normal(3)
    face_pts = [rng.standard_normal((c, 3)) for c in counts]
    ux = u_x.tolist()

    def per_face():
        ox, oy, oz = origin.tolist()
        xs = []
        for i in range(F):
            axis = u_x @ rots[i]
            t = offs[i]
            cx = (t[0] - ox) * ux[0] + (t[1] - oy) * ux[1] + (t[2] - oz) * ux[2]
            xs.append(face_pts[i] @ axis + cx)
        return np.concatenate(xs)

    idx_rep = np.repeat(np.arange(F), counts)
    row_off = np.zeros(F + 1, dtype=int)
    np.cumsum(counts, out=row_off[1:])
    P = np.concatenate(face_pts)

    def batched():
        axes = np.einsum("b,fba->fa", u_x, rots)
        AX = axes[idx_rep]
        S = offs - origin
        cs = (S * u_x).sum(axis=1)
        return (P * AX).sum(axis=1) + cs[idx_rep]

    t1 = min(timeit.repeat(per_face, number=20000, repeat=5)) / 20000
    t2 = min(timeit.repeat(batched, number=20000, repeat=5)) / 20000
    print(f"per_face {t1*1e6:7.2f} us   batched {t2*1e6:7.2f} us   speedup {t1/t2:.2f}x")


def _offsets(counts):
    o = 0
    for f, c in enumerate(counts):
        yield f, (o, c)
        o += c


if __name__ == "__main__":
    main()
