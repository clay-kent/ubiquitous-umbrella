"""パイプライン全体（列挙＋一意化）を **CPU 時間**で測る。

外部ロードで壁時計が数倍振れる環境で A/B するための道具。
``time.process_time`` はプロセスが実際に使った CPU 時間なので、
deschedule の影響を受けない（メモリ帯域の競合は残る）。

    uv run python tools/bench_pipeline_cpu.py stairs 3
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)

    best = float("inf")
    for _ in range(reps):
        t0 = time.process_time()
        trees = fifth.enumerate_spanning_trees(edges, len(faces))
        n_valid = sum(1 for _ in fifth.iter_unique_nets(v, faces, trees, normals))
        dt = time.process_time() - t0
        best = min(best, dt)
        print(f"  run cpu={dt:.3f}s valid={n_valid}")
    print(f"{name}: best cpu={best:.3f}s")


if __name__ == "__main__":
    main()
