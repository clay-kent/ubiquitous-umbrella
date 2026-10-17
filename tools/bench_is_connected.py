"""_is_connected の2実装（list版 / bitmask版）を直接叩いて速度を比較する。

プロセス全体の計測は外部ロード（実測で ±5 倍）を受けるため、
対象関数だけを孤立させて測る。判定結果は真偽値だけで、丸め演算无关。
"""

import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def is_connected_list(cur, num_nodes):
    nbrs = [[] for _ in range(num_nodes)]
    for a, b in cur:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen = {0}
    stack = [0]
    while stack:
        c = stack.pop()
        for w in nbrs[c]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == num_nodes


def is_connected_mask(cur, num_nodes):
    nbrs = [0] * num_nodes
    for a, b in cur:
        nbrs[a] |= 1 << b
        nbrs[b] |= 1 << a
    seen = 1
    frontier = 1
    while frontier:
        nxt_mask = 0
        f = frontier
        while f:
            low = f & -f
            nxt_mask |= nbrs[low.bit_length() - 1]
            f ^= low
        frontier = nxt_mask & ~seen
        seen |= frontier
    return seen.bit_count() == num_nodes


def main():
    rng = random.Random(20261017)
    # 面数 10 / 辺数 18 程度（stairs 相当）で、連結なグラフを生成
    n = 10
    cases = []
    for _ in range(2000):
        m = rng.randint(n, 3 * n)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
        cases.append((edges, n))

    # 結果が一致することを先に確認
    for edges, nn in cases[:200]:
        assert is_connected_list(edges, nn) == is_connected_mask(edges, nn)

    reps = 5
    t_list, t_mask = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        for edges, nn in cases:
            is_connected_list(edges, nn)
        t_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for edges, nn in cases:
            is_connected_mask(edges, nn)
        t_mask.append(time.perf_counter() - t0)

    ml = statistics.median(t_list)
    mm = statistics.median(t_mask)
    print(f"cases={len(cases)} reps={reps}")
    print(f"  list version : {ml * 1000:8.2f} ms  ({ml / len(cases) * 1e6:.2f} us/call)")
    print(f"  mask version : {mm * 1000:8.2f} ms  ({mm / len(cases) * 1e6:.2f} us/call)")
    print(f"  speedup      : {ml / mm:.2f}x")


if __name__ == "__main__":
    main()