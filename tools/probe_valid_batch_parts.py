"""`_valid_batch` の内訳を、**本番の関数を包んで**測る（写しを持たない）。

Cycle 35 で写しをやめた。写しは本番とずれるので、ここでは

* `_proven_overlaps`（証明の numpy 計算）
* `_build_polys`（GEOS に残ったネットだけの Polygon 生成）
* `Polygon.intersection`（GEOS の交差計算）

を包んで時間を積み、`_valid_batch` 全体から差し引いて「絞り込み・走査」を出す。

    uv run python tools/probe_valid_batch_parts.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import shapely.geometry

import fifth
from bench_ladder import SOLIDS

ST = {"proof": 0.0, "polys": 0.0, "intersect": 0.0, "total": 0.0, "nets": 0,
      "proven": 0, "calls": 0, "cand": 0, "windows": 0}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    prep = fifth.prepare_layout(v, faces, normals)

    real_proof = fifth._proven_overlaps
    real_polys = fifth._build_polys
    real_valid = fifth._valid_batch
    orig_intersection = shapely.geometry.Polygon.intersection

    def proof(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        hit = real_proof(*args, **kwargs)
        ST["proof"] += time.perf_counter() - t0
        ST["proven"] += int(hit.sum())
        ST["cand"] += int(hit.size)
        return hit

    def polys(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        out = real_polys(*args, **kwargs)
        ST["polys"] += time.perf_counter() - t0
        return out

    def counting(self: object, other: object) -> object:
        t0 = time.perf_counter()
        out = orig_intersection(self, other)  # type: ignore[arg-type]
        ST["intersect"] += time.perf_counter() - t0
        ST["calls"] += 1
        return out

    def valid(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        out = real_valid(*args, **kwargs)
        ST["total"] += time.perf_counter() - t0
        ST["nets"] += len(out[0])
        ST["windows"] += 1
        return out

    fifth._proven_overlaps = proof
    fifth._build_polys = polys
    fifth._valid_batch = valid
    shapely.geometry.Polygon.intersection = counting  # type: ignore[method-assign]

    t0 = time.perf_counter()
    n = 0
    try:
        for _tree, _polys in fifth.iter_unique_nets(
            v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals
        ):
            n += 1
    finally:
        fifth._proven_overlaps = real_proof
        fifth._build_polys = real_polys
        fifth._valid_batch = real_valid
        shapely.geometry.Polygon.intersection = orig_intersection  # type: ignore[method-assign]
    total_pipeline = time.perf_counter() - t0

    total = ST["total"]
    rest = total - ST["proof"] - ST["polys"] - ST["intersect"]
    print(f"{name}: nets={n} pipeline={total_pipeline:.3f}s")
    print(f"  _valid_batch 全体 : {total:7.3f}s  ({ST['windows']} 窓, {ST['nets']} ネット)")
    print(f"  candidates        : {ST['cand']}  proven: {ST['proven']}  "
          f"intersections: {ST['calls']}")
    for key, label in (
        ("proof", "証明（numpy）"),
        ("polys", "Polygon 生成"),
        ("intersect", "GEOS 交差"),
        ("rest", "絞り込み・走査"),
    ):
        t = rest if key == "rest" else ST[key]
        pct = t / total * 100 if total else 0.0
        print(f"  {label:16s} {t:7.3f}s ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
