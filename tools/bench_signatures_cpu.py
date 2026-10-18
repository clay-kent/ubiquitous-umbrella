"""署名フェーズを CPU 時間で測る（壁時計より外部負荷に強い）。

外部マシン負荷で壁時計が 5 倍ほど振れるので、``time.process_time`` を使う。
プロセスが deschedule されても「実際に使った CPU 時間」を見るので、
他プロセスの影響が相対的に小さい（メモリ帯域の競合は残る）。

    uv run python tools/bench_signatures_cpu.py stairs
"""

import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    ctx = fifth.RibbonContext(v, faces)
    mirror_ctx = fifth.RibbonContext(v, [list(reversed(ff)) for ff in faces])
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    print(f"{name}: {len(trees)} trees, window={fifth._LADDER_WINDOW}")

    best = float("inf")
    for _ in range(reps):
        t0 = time.process_time()
        rest = iter(trees)
        n_keys = 0
        while True:
            window = list(islice(rest, fifth._LADDER_WINDOW))
            if not window:
                break
            n_keys += len(fifth._batch_canonical_keys(ctx, mirror_ctx, window))
        dt = time.process_time() - t0
        best = min(best, dt)
        print(f"  run cpu={dt:.3f}s  keys={n_keys}")
    print(f"  best cpu={best:.3f}s")


if __name__ == "__main__":
    main()
