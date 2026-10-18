"""署名バッチの「窓サイズ」を変えて、署名フェーズの時間を測る。

`_batch_stabilize` は窓ごとに固定費（unique/searchsorted/glue_batch/確保）が乗り、
逆に大きい窓は最初の数ラウンドで扱う行数が増える。両者の兼ね合いを実測する。

    uv run python tools/probe_window_size.py stairs
"""

import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def run(name: str, sizes: list[int]) -> None:
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    print(f"{name}: {len(trees)} trees")
    for size in sizes:
        best = float("inf")
        n_keys = 0
        for _ in range(3):
            rest = iter(trees)
            t0 = time.perf_counter()
            while True:
                window = list(islice(rest, size))
                if not window:
                    break
                keys = fifth._batch_canonical_keys(ctx, mirror_ctx, window)
                n_keys += len(keys)
            dt = time.perf_counter() - t0
            best = min(best, dt)
        print(f"  window={size:6d}  best={best:7.3f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "stairs", [512, 1024, 2048, 4096, 8192])
