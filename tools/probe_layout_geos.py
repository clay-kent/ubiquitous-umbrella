"""展開（`_layout_batch`）のうち、numpy 部分と GEOS 呼び出しの内訳。

shapely の関数はモジュール属性として呼ばれるので、パッケージ側を差し替えて
合計時間を測れる（1 バッチ 1 回なので固定費は無視できる）。

    uv run python tools/probe_layout_geos.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import shapely

import fifth
from bench_ladder import SOLIDS

ST = {"linearrings": 0.0, "polygons": 0.0, "bounds": 0.0,
      "layout": 0.0, "batches": 0, "rows": 0, "calls": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    layout_prep = fifth.prepare_layout(v, faces, normals)

    real_lr = shapely.linearrings
    real_pg = shapely.polygons
    real_bd = shapely.bounds

    def lr(*a, **k):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        r = real_lr(*a, **k)
        ST["linearrings"] += time.perf_counter() - t0
        ST["calls"] += 1
        return r

    def pg(*a, **k):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        r = real_pg(*a, **k)
        ST["polygons"] += time.perf_counter() - t0
        return r

    def bd(*a, **k):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        r = real_bd(*a, **k)
        ST["bounds"] += time.perf_counter() - t0
        return r

    # 展開だけを直接叩く（署名・重なり判定は含めない）。
    trees = []
    for tree in fifth.enumerate_spanning_trees(edges, len(faces)):
        trees.append(tree)
        if len(trees) >= 40000:
            break

    shapely.linearrings = lr
    shapely.polygons = pg
    shapely.bounds = bd
    fifth.shapely.linearrings = lr
    fifth.shapely.polygons = pg
    fifth.shapely.bounds = bd
    t0 = time.perf_counter()
    n = 0
    for i in range(0, len(trees), fifth._LADDER_WINDOW):
        window = trees[i : i + fifth._LADDER_WINDOW]
        fifth._layout_batch(len(faces), window, layout_prep)
        ST["batches"] += 1
        ST["rows"] += len(window)
        n += len(window)
    total = time.perf_counter() - t0
    shapely.linearrings = real_lr
    shapely.polygons = real_pg
    shapely.bounds = real_bd
    fifth.shapely.linearrings = real_lr
    fifth.shapely.polygons = real_pg
    fifth.shapely.bounds = real_bd

    geos = ST["linearrings"] + ST["polygons"] + ST["bounds"]
    print(f"{name}: rows={n} batches={ST['batches']} total={total:.3f}s "
          f"({total / n * 1e6:.1f} us/net)")
    print(f"  numpy + Python   {total - geos:7.3f}s "
          f"({(total - geos) / total * 100:.1f}%)")
    for k in ("linearrings", "polygons", "bounds"):
        print(f"  shapely.{k:11s} {ST[k]:7.3f}s "
              f"({ST[k] / total * 100:.1f}%, {ST[k] / n * 1e6:.2f} us/net)")
    print(f"  GEOS 合計        {geos:7.3f}s ({geos / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
