"""is_valid_layout の1ネットあたりの所要時間を孤立計測する（bounds取得方法だけを変える）。"""

import itertools
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import shapely
from shapely.geometry import Polygon

import fifth
from bench_ladder import SOLIDS

OVERLAP = fifth.OVERLAP_AREA_REL_TOL


def valid_python_loop(polygons):
    """変更前: p.bounds を Python ループで取る逐次版。"""
    n_polys = len(polygons)
    bounds = [p.bounds for p in polygons]
    for i in range(n_polys):
        ax0, ay0, ax1, ay1 = bounds[i]
        for j in range(i + 1, n_polys):
            bx0, by0, bx1, by1 = bounds[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            dx = max(ax1, bx1) - min(ax0, bx0)
            dy = max(ay1, by1) - min(ay0, by0)
            if polygons[i].intersection(polygons[j]).area > OVERLAP * (dx * dx + dy * dy):
                return False
    return True


def main():
    for name in ["lshape", "stairs"]:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        prep = fifth.prepare_layout(v, faces, normals)
        edges = fifth.build_adjacency_graph(faces)
        seen = set()
        ctx = fifth.RibbonContext(v, faces)
        mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
        rest = fifth.enumerate_spanning_trees(edges, len(faces))
        nets = []
        for chunk in iter(lambda: list(itertools.islice(rest, 2048)), []):
            for tree, (a, b) in zip(chunk, fifth._batch_canonical_keys(ctx, mctx, chunk)):
                k = min(a, b)
                if k in seen:
                    continue
                seen.add(k)
                nets.append(fifth.layout_tree(v, faces, normals, tree, prep))
            if len(nets) >= 1500:
                break

        t_old, t_new = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            for p in nets:
                valid_python_loop(p)
            t_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for p in nets:
                fifth.is_valid_layout(p)
            t_new.append(time.perf_counter() - t0)
        mo = statistics.median(t_old)
        mn = statistics.median(t_new)
        print(
            f"{name:8s} nets={len(nets)}  old={mo * 1000:8.1f}ms "
            f"new={mn * 1000:8.1f}ms  ratio={mo / mn:.2f}x"
        )


if __name__ == "__main__":
    main()