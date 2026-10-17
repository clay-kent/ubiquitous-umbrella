"""複雑度をさらに1段上げた形体での計測（1分タイムアウト付き）。

stairs（10面 / 150,528全域木）が約31秒で、1分上限に近づいている。
このスクリプトで「次の段」がどこまで行けるかを調べる。
全域木が上限を超える形体は 422 相当（列挙前に打ち切り）になるだけなので、
上限を無視して iter_unique_nets を回した時間を測る。
"""

import itertools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth


def prism(n, r=1.0, h=1.0):
    v = [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ] + [
        [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), h]
        for i in range(n)
    ]
    f = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    f += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return np.array(v, dtype=float), f


def stair_profile(n, w=1.0, h=1.0):
    """n 段の階段柱。面数 = n+2、全域木は急剧に増える。"""
    pts = [(0, 0)]
    for i in range(n):
        x = (n - i) * w
        y = i * h
        pts.append((x, y))
        if i < n - 1:
            pts.append((x - w, y))
    k = len(pts)
    v = np.array([[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts])
    f = [list(range(k - 1, -1, -1)), list(range(k, 2 * k))]
    f += [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    return v, f


def cross_profile(a=1.0, b=3.0):
    """十字形の柱（2方向から見て十字）。"""
    pts = [
        (-a, -b), (a, -b), (a, -a), (b, -a), (b, a), (a, a),
        (a, b), (-a, b), (-a, a), (-b, a), (-b, -a), (-a, -a),
    ]
    k = len(pts)
    v = np.array([[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts])
    f = [list(range(k - 1, -1, -1)), list(range(k, 2 * k))]
    f += [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    return v, f


CASES = {
    "prism8": lambda: prism(8),
    "prism9": lambda: prism(9),
    "stairs4": lambda: stair_profile(4),
    "stairs5": lambda: stair_profile(5),
    "cross": cross_profile,
}


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    names = sys.argv[2:] or list(CASES)
    print(f"{'name':10s} {'faces':>5s} {'trees':>14s} {'unique':>8s} {'elapsed':>9s}  status")
    for name in names:
        v, f = CASES[name]()
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        edges = fifth.build_adjacency_graph(faces)
        n_trees = fifth.count_spanning_trees(edges, len(faces))

        t0 = time.perf_counter()
        n_unique = 0
        trees = fifth.enumerate_spanning_trees(edges, len(faces))
        status = "ok"
        for _ in fifth.iter_unique_nets(v, faces, trees, normals):
            n_unique += 1
            if time.perf_counter() - t0 > timeout:
                status = "TIMEOUT"
                break
        elapsed = time.perf_counter() - t0
        print(
            f"{name:10s} {len(faces):5d} {n_trees:14,d} {n_unique:8,d} "
            f"{elapsed:8.1f}s  {status}"
        )


if __name__ == "__main__":
    main()