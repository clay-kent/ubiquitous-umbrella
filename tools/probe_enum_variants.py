"""`enumerate_spanning_trees` の実装候補を比べる（CPU 時間、出力は同一かも確認）。

出力する木の順序と中身が変わると、代表として選ばれる木が変わり
`hash_layout_bits.py` が落ちる。ここでは「完全に同じ列」を返すことまで確かめる。

    uv run python tools/probe_enum_variants.py stairs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def _connected_triples(cur, num_nodes):
    """(番号, a, b) の列が全頂点を連結にするか。リスト内包を作らない版。"""
    nbrs = [[] for _ in range(num_nodes)]
    for _i, a, b in cur:
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


def enumerate_v2(edges, num_nodes):
    """_is_connected を三つ組対応にしてリスト内包を消しただけの版。"""
    if num_nodes <= 0:
        return
    if num_nodes == 1:
        yield []
        return
    base = [(i, u, v) for i, (u, v) in enumerate(edges) if u != v]
    endpoints = {i: (u, v) for i, u, v in base}
    if not _connected_triples(base, num_nodes):
        return

    def rec(cur, n, chosen):
        pivot = -1
        for i, (_, a, b) in enumerate(cur):
            if a != b:
                pivot = i
                break
        if pivot < 0:
            if n == 1:
                yield [endpoints[i] for i in sorted(chosen)]
            return
        orig, pu, pv = cur[pivot]
        rest = cur[:pivot] + cur[pivot + 1:]
        if _connected_triples(rest, n):
            yield from rec(rest, n, chosen)
        merged = n - 2
        contracted = []
        for i, a, b in rest:
            a2 = merged if (a == pu or a == pv) else a - (pu < a) - (pv < a)
            b2 = merged if (b == pu or b == pv) else b - (pu < b) - (pv < b)
            contracted.append((i, a2, b2))
        yield from rec(contracted, n - 1, chosen + [orig])

    yield from rec(base, num_nodes, [])


def cpu(fn):
    best = 1e9
    out = None
    for _ in range(3):
        t0 = time.process_time()
        out = list(fn())
        best = min(best, time.process_time() - t0)
    return best, out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)
    n = len(faces)

    t_old, out_old = cpu(lambda: fifth.enumerate_spanning_trees(edges, n))
    t_new, out_new = cpu(lambda: enumerate_v2(edges, n))
    same = out_old == out_new
    print(f"{name}: trees={len(out_old)} identical={same}")
    print(f"  current {t_old:.4f}s   v2 {t_new:.4f}s   ratio {t_old / t_new:.3f}")


if __name__ == "__main__":
    main()
