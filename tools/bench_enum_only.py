"""列挙フェーズだけを孤立計測する（外部ロードの影響を最小化する）。

tools/bench_repeat.py の総時間だと、他フェーズの load 変動に引っ張られる。
ここでは「20000 本の列挙天数」だけを測る。これは明確に 1 つの処理の
所要時間なので、絶対値としても比較の目安になる（かつ総時間より安定する）。
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS

SAMPLE = 20000


def measure(name):
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)
    import itertools

    t0 = time.perf_counter()
    n = sum(
        1
        for _ in itertools.islice(
            fifth.enumerate_spanning_trees(edges, len(faces)), SAMPLE
        )
    )
    return time.perf_counter() - t0, n


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for name in ["prism7", "stairs"]:
        vals = []
        for _ in range(reps):
            el, n = measure(name)
            vals.append(el)
        m = statistics.median(vals)
        print(
            f"{name:8s} enumerate {SAMPLE} trees: median={m:.3f}s "
            f"(min={min(vals):.3f}s max={max(vals):.3f}s) "
            f"= {m / SAMPLE * 1e6:.1f} us/tree"
        )


if __name__ == "__main__":
    main()