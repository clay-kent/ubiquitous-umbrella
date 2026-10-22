"""`_layout_batch`（座標の計算）と `_build_polys`（shapely の生成）の内訳。

Cycle 34 で Polygon の生成を `_layout_batch` から切り離したので、
「座標の計算」と「Polygon の生成」を別々に測る。本番は Polygon を
残った行だけ作るため、ここでは **全行ぶん作った場合**（従来と同じ条件）を出す。

    uv run python tools/probe_layout_phases.py stairs
    uv run python tools/probe_layout_phases.py stairs 20     # 窓の数
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS

ST = {"layout": 0.0, "polys": 0.0, "rows": 0, "batches": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    n_window = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    prep = fifth.prepare_layout(v, faces, normals)

    trees = []
    for tree in fifth.enumerate_spanning_trees(edges, len(faces)):
        trees.append(tree)
        if len(trees) >= n_window * fifth._LADDER_WINDOW:
            break

    # 写しではなく本番をそのまま測る（座標が一致することは
    # tools/check_layout_batch_bits.py が全木で確認している）。
    for i in range(0, len(trees), fifth._LADDER_WINDOW):
        window = trees[i : i + fifth._LADDER_WINDOW]
        t0 = time.perf_counter()
        batch = fifth._layout_batch(len(faces), window, prep)
        t1 = time.perf_counter()
        produced = fifth._build_polys(prep, batch.xs, batch.ys)
        t2 = time.perf_counter()
        assert len(produced) == len(window) * len(faces)
        ST["layout"] += t1 - t0
        ST["polys"] += t2 - t1
        ST["rows"] += len(window)
        ST["batches"] += 1

    n = ST["rows"]
    total = ST["layout"] + ST["polys"]
    print(f"{name}: rows={n} batches={ST['batches']} total={total:.3f}s "
          f"({total / n * 1e6:.2f} us/net)")
    for key, label in (("layout", "座標+外接矩形"), ("polys", "Polygon 生成（全行）")):
        t = ST[key]
        print(f"  {label:24s} {t:7.3f}s ({t / total * 100:5.1f}%, {t / n * 1e6:6.2f} us/net)")


if __name__ == "__main__":
    main()
