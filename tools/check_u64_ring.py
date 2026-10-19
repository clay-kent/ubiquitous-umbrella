"""uint64（mod 2^64）の足し算が結合法則を満たすことの確認（コメント修正の根拠）。"""

import numpy as np

rng = np.random.default_rng(0)
a = rng.integers(0, 2**64, size=1000, dtype=np.uint64)
b = rng.integers(0, 2**64, size=1000, dtype=np.uint64)
c = rng.integers(0, 2**64, size=1000, dtype=np.uint64)
d = rng.integers(0, 2**64, size=1000, dtype=np.uint64)
x = ((a + b) + c) + d
y = a + ((b + c) + d)
z = (a + c) + (b + d)
print("associative:", bool((x == y).all()), "commutative:", bool((x == z).all()))

w = np.uint64(0x9E3779B97F4A7C15)
p = (a * w) + (b * w)
q = (a + b) * w
print("distributive:", bool((p == q).all()))
