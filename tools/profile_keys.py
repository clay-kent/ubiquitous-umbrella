"""_batch_canonical_keys の中のホットスポットを cProfile で見る。

処理回数と累積時間の両方を出す。絶対値は外部ロードで変わるが、
**相対順位**（どこが重いか）は安定している。
"""

import cProfile
import io
import itertools
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

name = sys.argv[1] if len(sys.argv) > 1 else "prism7"
v, f = SOLIDS[name]
faces = fifth.orient_faces_outward(v, f)
normals = fifth.compute_normals(v, faces)
edges = fifth.build_adjacency_graph(faces)
ctx = fifth.RibbonContext(v, faces)
mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))


def run():
    for chunk in [trees[i : i + 2048] for i in range(0, len(trees), 2048)]:
        fifth._batch_canonical_keys(ctx, mctx, chunk)


pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(18)
print(f"=== {name}: {len(trees)} trees, batch_canonical_keys ===")
print(s.getvalue())
