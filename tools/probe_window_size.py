"""`_LADDER_WINDOW` を変えて、返る展開図列が同一のまま速度だけ変わるかを見る。

窓の大きさは「バッチの粒」なので、numpy の呼び出し固定費が効く局面では
大きいほうが有利、配列が大きくなりすぎると不利。返り値は窓に依存しない
（test_window_size_does_not_change_the_result で固定済み）が、ここでも
木の列とポリゴン座標のハッシュを取って確認する。

    uv run python tools/probe_window_size.py stairs 1024,2048,8192
"""

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def measure(name: str, window: int, reps: int) -> tuple[float, int, str]:
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, f)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))

    original = fifth._LADDER_WINDOW
    fifth._LADDER_WINDOW = window
    best = float("inf")
    n_valid = 0
    digest = ""
    try:
        for _ in range(reps):
            t0 = time.process_time()
            h = hashlib.sha256()
            n_valid = 0
            for tree, polys in fifth.iter_unique_nets(v, faces, iter(trees), normals):
                h.update(np.asarray(tree, dtype=np.int64).tobytes())
                for p in polys:
                    h.update(np.asarray(p.exterior.coords, dtype=np.float64).tobytes())
                n_valid += 1
            dt = time.process_time() - t0
            best = min(best, dt)
            digest = h.hexdigest()[:24]
    finally:
        fifth._LADDER_WINDOW = original
    return best, n_valid, digest


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    windows = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2048").split(",")]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    results = []
    for w in windows:
        cpu, n, digest = measure(name, w, reps)
        results.append((w, cpu, n, digest))
        print(f"window={w:7d}  cpu={cpu:7.3f}s  valid={n:6d}  sha256={digest}")
    base = results[0]
    same = all(r[2] == base[2] and r[3] == base[3] for r in results)
    best = min(results, key=lambda r: r[1])
    print(f"{name}: identical_output={same}  fastest_window={best[0]} ({best[1]:.3f}s)")


if __name__ == "__main__":
    main()
