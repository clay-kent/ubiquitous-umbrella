"""連結判定の実装を、**大きいグラフ**（面数10・辺数10〜30）で比べる。

ここで分かるのは「立体まるごとの面隣接グラフのような大きい入力では、
頂点ごとの隣接ビットマスク + 前線ビットの BFS（frontier）は
隣接リスト + set の BFS（list）に負ける」こと。

⚠️ これだけを見て「ビットマスクは遅い」と結論してはいけない。列挙の再帰が
渡す入力は小さい（stairs で平均 4.8 辺）ので、**実際に呼ばれた入力**で測ると
逆転する。そちらは tools/probe_is_connected.py（30,000 件、全件一致、
同入力での比較）と tools/probe_ab_enum.py（列挙そのものの新旧 A/B）が本命で、
現行の `_is_connected_edges`（到達ビットマスクを辺の走査で伸ばす版）は
そちらで採用した。

    uv run python tools/bench_is_connected.py
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth


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


def is_connected_frontier(cur, num_nodes):
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
    n = 10
    cases = []
    for _ in range(2000):
        m = rng.randint(n, 3 * n)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
        cases.append((edges, n))

    for edges, nn in cases[:200]:
        assert is_connected_list(edges, nn) == is_connected_frontier(edges, nn)
        assert fifth._is_connected(edges, nn) == is_connected_list(edges, nn)

    impls = {"list": is_connected_list, "frontier": is_connected_frontier,
             "current": fifth._is_connected}
    best = dict.fromkeys(impls, float("inf"))
    for _ in range(5):
        for k, fn in impls.items():
            t0 = time.perf_counter()
            for edges, nn in cases:
                fn(edges, nn)
            best[k] = min(best[k], time.perf_counter() - t0)

    print(f"cases={len(cases)} (n={n}, m={n}..{3 * n}) reps=5, min")
    for k in impls:
        ms = best[k] * 1000
        print(f"  {k:9s} {ms:8.2f} ms  ({best[k] / len(cases) * 1e6:5.2f} us/call)")
    print(f"  比 list/current = {best['list'] / best['current']:.2f}x  "
          f"list/frontier = {best['list'] / best['frontier']:.2f}x")


if __name__ == "__main__":
    main()
