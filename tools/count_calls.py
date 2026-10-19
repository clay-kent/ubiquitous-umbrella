"""列挙1回あたりの _is_connected 呼び出し回数を数える（絶対時間に依存しない計測）。"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS

LIMIT = 20000


def main():
    for name in ["prism6", "prism7", "stairs"]:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        edges = fifth.build_adjacency_graph(faces)
        n_trees = fifth.count_spanning_trees(edges, len(faces))
        calls = {"n": 0}
        orig = fifth._is_connected_edges

        def counting(cur, nn, _o=orig):
            calls["n"] += 1
            return _o(cur, nn)

        fifth._is_connected_edges = counting
        try:
            emitted = sum(
                1
                for _ in itertools.islice(
                    fifth.enumerate_spanning_trees(edges, len(faces)), LIMIT
                )
            )
        finally:
            fifth._is_connected_edges = orig
        per_tree = calls["n"] / max(emitted, 1)
        print(
            f"{name:8s} faces={len(faces):2d} total_trees={n_trees:9,d} "
            f"sampled={emitted:6,d} _is_connected calls={calls['n']:9,d} "
            f"({per_tree:.1f} per tree)"
        )


if __name__ == "__main__":
    main()