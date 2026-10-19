"""`_is_connected` の速い実装を、**実際に呼ばれた入力**で比べる。

列挙 stairs で連結判定は 33.9 万回・0.38 秒（列挙の 31%）。入力は小さい
（平均 4.8 辺）。旧実装は呼び出し側で `[(a, b) for _, a, b in rest]` と
ペアのリストを作り直してから BFS していた。

比較（結果が全件一致することを先に確認してから交互に計時）:
  (a) old     : ペア列を作り直す + 隣接リスト(list of list) + set の BFS
  (b) triples : (a) の BFS だが (i, a, b) を直接受ける（作り直しなし）
  (c) bitset  : 到達ビットマスクを辺の走査で伸ばす（(i, a, b) を直接受ける）

    uv run python tools/probe_is_connected.py stairs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fifth
from bench_ladder import SOLIDS

Triples = list[tuple[int, int, int]]
SAMPLES: list[tuple[Triples, int]] = []
LIMIT = 30000


def old_bfs(cur: list[tuple[int, int]], num_nodes: int) -> bool:
    nbrs: list[list[int]] = [[] for _ in range(num_nodes)]
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


def triples_bfs(cur: Triples, num_nodes: int) -> bool:
    nbrs: list[list[int]] = [[] for _ in range(num_nodes)]
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


def bitset(cur: Triples, num_nodes: int) -> bool:
    full = (1 << num_nodes) - 1
    reach = 1
    while True:
        added = reach
        for _i, a, b in cur:
            if reach >> a & 1:
                added |= 1 << b
            elif reach >> b & 1:
                added |= 1 << a
        if added == reach:
            return reach == full
        reach = added
        if reach == full:
            return True


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)

    real = fifth._is_connected_edges

    def recorder(e, n):  # type: ignore[no-untyped-def]
        if len(SAMPLES) < LIMIT:
            SAMPLES.append((list(e), n))
        return real(e, n)

    fifth._is_connected_edges = recorder
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    fifth._is_connected_edges = real
    print(f"{name}: trees={len(trees)} samples={len(SAMPLES)}")

    bad = 0
    for e, n in SAMPLES:
        r = old_bfs([(a, b) for _i, a, b in e], n)
        if triples_bfs(e, n) != r or bitset(e, n) != r:
            bad += 1
    print(f"  結果の不一致: {bad}")

    variants = {
        "old": lambda k: old_bfs([(a, b) for _i, a, b in SAMPLES[k][0]], SAMPLES[k][1]),
        "triples": lambda k: triples_bfs(SAMPLES[k][0], SAMPLES[k][1]),
        "bitset": lambda k: bitset(SAMPLES[k][0], SAMPLES[k][1]),
    }
    best = dict.fromkeys(variants, float("inf"))
    for _ in range(4):
        for k, fn in variants.items():
            t0 = time.perf_counter()
            for i in range(len(SAMPLES)):
                fn(i)
            dt = time.perf_counter() - t0
            best[k] = min(best[k], dt)
    m = len(SAMPLES)
    for k, dt in best.items():
        print(f"  {k:8s} {dt:7.3f}s  ({dt / m * 1e6:6.2f} us/call)")


if __name__ == "__main__":
    main()
