"""投影の「バッチ化」が GEMV とビット一致するかを確かめる。

`layout_tree` は全 face の頂点を 1 本の (M,3) に積み、面ごとの軸を
行ごとに繰り返した配列と成分ごとに掛けて足す（Cycle 18）。これは速いが、
numpy の GEMV (`pts @ axis`) とは**丸めが一致しない**。

- 一致しない理由: GEMV のカーネルと、要素ごとの積＋和では、
  積和の順序（FMA の有無）が違う。
- それでも採れるのは、座標の差が 9 桁丸めより下（ULP 級）で、
  重なり判定は相対しきい値 1e-7 で堅牢、API golden と
  `tools/hash_metrics_rounded.py` が一致するため。

「バッチ化したいが**ビット一致**が要る」場合は
``tools/check_projection_axes_bits.py``（軸と定数項だけのバッチ）を使う。

    uv run python tools/check_projection_batching.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

F = 10
COUNTS = [4, 4, 4, 4, 6, 4, 4, 6, 4, 4]


def main() -> None:
    rng = np.random.default_rng(0)
    trials = 20000
    bad_fused = 0
    bad_explicit = 0
    for _ in range(trials):
        rot = rng.standard_normal((F, 3, 3))
        u = rng.standard_normal(3)
        P = rng.standard_normal((sum(COUNTS), 3))
        axes = np.einsum("b,fba->fa", u, rot)
        idx = np.repeat(np.arange(F), COUNTS)
        AX = axes[idx]
        ref = np.concatenate(
            [P[o : o + c] @ axes[f] for f, (o, c) in _spans()]
        )
        if not np.array_equal((P * AX).sum(axis=1), ref):
            bad_fused += 1
        if not np.array_equal(
            P[:, 0] * AX[:, 0] + P[:, 1] * AX[:, 1] + P[:, 2] * AX[:, 2], ref
        ):
            bad_explicit += 1
    print(f"(P*AX).sum(axis=1)          differs {bad_fused} / {trials}")
    print(f"explicit left-assoc form    differs {bad_explicit} / {trials}")
    print("→ GEMV とは一致しない。9 桁丸めと golden で観測差がないことを確認して採る。")


def _spans():
    o = 0
    for f, c in enumerate(COUNTS):
        yield f, (o, c)
        o += c


if __name__ == "__main__":
    main()
