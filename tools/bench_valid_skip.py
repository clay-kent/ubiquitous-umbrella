"""is_valid_layout の folded_pairs 省略の効果を孤立計測する。

同じ展開図集合に対して
  (a) 従来: is_valid_layout(polys)          … 全ペアを交差計算
  (b) 新:   is_valid_layout(polys, folded)  … 折り線ペアを省略
を交互に測り、比を出す。絶対値はロードで変わるが比は安定する。
"""

import itertools
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def collect(name, limit_nets):
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    prep = fifth.prepare_layout(v, faces, normals)
    edges = fifth.build_adjacency_graph(faces)
    ctx = fifth.RibbonContext(v, faces)
    mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
    seen = set()
    nets = []
    rest = fifth.enumerate_spanning_trees(edges, len(faces))
    for chunk in iter(lambda: list(itertools.islice(rest, 2048)), []):
        for tree, (a, b) in zip(
            chunk, fifth._batch_canonical_keys(ctx, mctx, chunk)
        ):
            k = min(a, b)
            if k in seen:
                continue
            seen.add(k)
            polys = fifth.layout_tree(v, faces, normals, tree, prep)
            folded = {(min(u, w), max(u, w)) for u, w in tree}
            nets.append((polys, folded))
        if len(nets) >= limit_nets:
            break
    return nets


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for name in ["prism7", "lshape", "stairs"]:
        nets = collect(name, 2500)
        t_old, t_new = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            for polys, _ in nets:
                fifth.is_valid_layout(polys)
            t_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for polys, folded in nets:
                fifth.is_valid_layout(polys, folded)
            t_new.append(time.perf_counter() - t0)
        mo, mn = statistics.median(t_old), statistics.median(t_new)
        print(
            f"{name:8s} nets={len(nets):4d}  full={mo * 1000:8.1f}ms "
            f"skip={mn * 1000:8.1f}ms  speedup={mo / mn:.2f}x"
        )


if __name__ == "__main__":
    main()
