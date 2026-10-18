"""小さな numpy の内積/行列ベクトル積が、スカラ演算とビット一致するかを確かめる。

一致すれば、numpy のディスパッチ費用（1回あたり ~1.5µs）をスカラ式に置き換えて
丸め演算の順序を変えずに高速化できる。一致しなければ採用しない。
"""

import numpy as np

rng = np.random.default_rng(0)


def scalar_dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def scalar_matvec3(R, v):
    return np.array(
        [
            R[0, 0] * v[0] + R[0, 1] * v[1] + R[0, 2] * v[2],
            R[1, 0] * v[0] + R[1, 1] * v[1] + R[1, 2] * v[2],
            R[2, 0] * v[0] + R[2, 1] * v[1] + R[2, 2] * v[2],
        ]
    )


def main():
    dot_bad = 0
    for trial in range(300000):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        d1 = float(a @ b)
        d2 = scalar_dot3(a, b)
        if d1 != d2:
            dot_bad += 1
            if dot_bad <= 3:
                print(f"DOT3 differs at trial {trial}: {d1!r} vs {d2!r}")
    print(f"DOT3: {dot_bad} / 300000 differ")

    mat_bad = 0
    for trial in range(300000):
        R = rng.standard_normal((3, 3))
        a = rng.standard_normal(3)
        if not np.array_equal(R @ a, scalar_matvec3(R, a)):
            mat_bad += 1
    print(f"MATVEC3: {mat_bad} / 300000 differ")


if __name__ == "__main__":
    main()
