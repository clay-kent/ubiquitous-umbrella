"""WL 更新の in-place 化が**ビット単位で同じ**ことを確かめる。

`_batch_stabilize` の更新は 1 本の式で書くと (rows, n_nodes) の一時配列が
9 個できる。実サイズ（stairs: 2048 x 48 uint64 = 786 KB）では確保と
メモリ帯域が支配的なので、逐次 in-place にして確保を 3 個に減らす。

uint64 の掛け算・足し算は**順序で値が変わる**ので、足す順序を元の式と
完全に揃えることが条件。ここではランダム入力で値の一致を確かめる。

    uv run python tools/check_wl_update_inplace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth

N, K = 2048, 48


def main() -> None:
    rng = np.random.default_rng(0)
    nxt = rng.permutation(K)
    trials = 2000
    bad = 0
    for _ in range(trials):
        h = _rand(rng)
        pos = rng.integers(0, K, (N, K)).astype(np.int64)
        coef = _rand(rng)
        tag = _rand(rng)

        old = (
            h * fifth._WL_A
            + h[:, nxt] * fifth._WL_B
            + np.take_along_axis(h, pos, axis=1) * coef
            + tag
        )

        nh = h * fifth._WL_A
        t = h[:, nxt]
        t *= fifth._WL_B
        nh += t
        t2 = np.take_along_axis(h, pos, axis=1)
        t2 *= coef
        nh += t2
        nh += tag
        if not np.array_equal(old, nh):
            bad += 1
    print(f"in-place differs {bad} / {trials}")


def _rand(rng):
    return rng.integers(0, 1 << 40, (N, K)).astype(np.uint64)


if __name__ == "__main__":
    main()
