"""layout_tree 単体のコストを孤立計測する（折り曲げ前計算の効果を測る）。

総時間や他フェーズは外部ロードの影響を受けるため、
「一意な展開図ぶんだけ layout_tree を回す」時間だけを測る。
折り曲げ前計算は行列積の回数を減らすので、ここに出るはず。
"""

import itertools
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def collect(name, limit_nets):
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    prep = fifth.prepare_layout(v, faces, normals)
    edges = fifth.build_adjacency_graph(faces)
    ctx = fifth.RibbonContext(v, faces)
    mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
    seen = set()
    trees = []
    rest = fifth.enumerate_spanning_trees(edges, len(faces))
    for chunk in iter(lambda: list(itertools.islice(rest, 2048)), []):
        for tree, (a, b) in zip(chunk, fifth._batch_canonical_keys(ctx, mctx, chunk)):
            k = min(a, b)
            if k in seen:
                continue
            seen.add(k)
            trees.append(tree)
        if len(trees) >= limit_nets:
            break
    return v, faces, normals, prep, trees


def main():
    for name in ["prism7", "lshape", "stairs"]:
        v, faces, normals, prep, trees = collect(name, 2000)
        vals = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for tree in trees:
                fifth.layout_tree(v, faces, normals, tree, prep)
            vals.append(time.perf_counter() - t0)
        m = statistics.median(vals)
        print(
            f"{name:8s} trees={len(trees):5d}  layout_tree total "
            f"median={m * 1000:8.1f}ms  = {m / len(trees) * 1e6:.1f} us/tree  "
            f"(min={min(vals) * 1000:.1f} max={max(vals) * 1000:.1f})"
        )


if __name__ == "__main__":
    main()
