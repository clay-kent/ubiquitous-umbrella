"""探針の円板による証明で GEOS 呼び出しを省いても、判定が変わらないことを確かめる。

`_valid_batch` を包んで、同じ窓に対して

* 証明あり（本番）
* 証明なし（`_proven_overlaps` を「何も証明しない」に差し替え）

の両方を走らせ、**返る bool 列が完全に一致**することを確かめる。あわせて
`Polygon.intersection` の呼び出し回数を数え、証明で何回減ったかを出す。

    uv run python tools/check_valid_cert.py            # 既定の立体すべて
    uv run python tools/check_valid_cert.py stairs
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

DEFAULT = ["stairs", "lshape", "prism7", "prism6", "prism5", "frustum4", "pyr5"]

STAT = {"batches": 0, "mismatch": 0, "with": 0, "without": 0}
CALLS = {"n": 0}


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    real_valid = fifth._valid_batch
    real_proof = fifth._proven_overlaps
    orig_intersection = shapely.geometry.Polygon.intersection

    def counting(self: object, other: object) -> object:
        CALLS["n"] += 1
        return orig_intersection(self, other)  # type: ignore[arg-type]

    def no_proof(*args: object, **kwargs: object) -> object:
        import numpy as np

        rows = args[0]
        return np.zeros(len(rows), dtype=bool)  # type: ignore[arg-type]

    def wrapper(layout, trees, precomputed):  # type: ignore[no-untyped-def]
        STAT["batches"] += 1
        shapely.geometry.Polygon.intersection = counting  # type: ignore[method-assign]
        CALLS["n"] = 0
        got, _polys = real_valid(layout, trees, precomputed)
        STAT["with"] += CALLS["n"]
        CALLS["n"] = 0
        fifth._proven_overlaps = no_proof  # type: ignore[assignment]
        try:
            ref, _polys2 = real_valid(layout, trees, precomputed)
        finally:
            fifth._proven_overlaps = real_proof  # type: ignore[assignment]
        STAT["without"] += CALLS["n"]
        shapely.geometry.Polygon.intersection = orig_intersection  # type: ignore[method-assign]
        if got != ref:
            STAT["mismatch"] += 1
            print("  不一致の窓:", [i for i, (a, b) in enumerate(zip(got, ref)) if a != b][:8])
        return got, _polys

    fifth._valid_batch = wrapper
    ok = True
    for name in names:
        if name not in SOLIDS:
            print(f"{name}: SOLIDS に無い")
            ok = False
            continue
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, faces)
        edges = fifth.build_adjacency_graph(faces)
        for key in STAT:
            STAT[key] = 0
        t0 = time.perf_counter()
        n = sum(
            1
            for _ in fifth.iter_unique_nets(
                v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals
            )
        )
        dt = time.perf_counter() - t0
        saved = STAT["without"] - STAT["with"]
        pct = saved / STAT["without"] * 100 if STAT["without"] else 0.0
        verdict = "OK" if STAT["mismatch"] == 0 else "NG"
        print(
            f"{name:9s} nets={n:6d} windows={STAT['batches']:4d} "
            f"intersections {STAT['without']:6d} -> {STAT['with']:6d} "
            f"(-{pct:5.1f}%) {verdict} ({dt:.1f}s)"
        )
        ok = ok and STAT["mismatch"] == 0
    fifth._valid_batch = real_valid
    print("ALL OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
