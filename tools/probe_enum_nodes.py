"""`enumerate_spanning_trees` の再帰ノード数と `_is_connected` の呼び出し回数。

「出力 1 本あたり何回 rec が呼ばれるか」が分かると、1 回あたりの削り代が見える。

    uv run python tools/probe_enum_nodes.py stairs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)

    stats = {"connected": 0, "nodes": 0}
    original = fifth._is_connected

    def counted(cur, num_nodes):
        stats["connected"] += 1
        return original(cur, num_nodes)

    fifth._is_connected = counted
    try:
        trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    finally:
        fifth._is_connected = original

    n = len(trees)
    print(
        f"{name}: trees={n} is_connected_calls={stats['connected']}"
        f" ({stats['connected'] / max(n, 1):.2f}/tree)"
    )


if __name__ == "__main__":
    main()
