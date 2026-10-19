"""「面の面積の総和 > 外接矩形の面積」で不正な展開図を弾けるかを測る。

有効な展開図では面が互いに重ならないので、面積の総和 = 合体した図形の面積
≤ 全体の外接矩形の面積 が必ず成り立つ。よって

    面積の総和 > 外接矩形の面積 （浮動小数の誤差を許して少しの余裕を付ける）

なら**確実に不正**。この判定は shapely の交差計算を一切呼ばず、座標配列だけで
計算できるので、`_layout_batch` の後・ポリゴン生成の前に置けると、
弾かれるネットではポリゴン生成も 45 ペアの走査も丸ごと省ける。

ここでは実際の一意ネットに対してこの条件がどれだけ弾けるか、
また「有効なネットを誤って弾かないか」を数える。

    uv run python tools/probe_area_filter.py stairs
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS

REL = 1e-9


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)

    stats = {"n": 0, "valid": 0, "rejected": 0, "rejected_invalid": 0, "rejected_valid": 0,
             "worst_ratio_valid": 0.0, "min_ratio_invalid_rejected": float("inf")}
    ratios: list[float] = []

    orig = fifth.is_valid_layout

    def wrapper(polygons, folded_pairs=None):  # type: ignore[no-untyped-def]
        total = 0.0
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for p in polygons:
            total += p.area
            b = p.bounds
            minx = min(minx, b[0])
            miny = min(miny, b[1])
            maxx = max(maxx, b[2])
            maxy = max(maxy, b[3])
        bbox_area = (maxx - minx) * (maxy - miny)
        ratio = total / bbox_area if bbox_area > 0 else float("inf")
        ok = orig(polygons, folded_pairs)
        stats["n"] += 1
        stats["valid"] += int(ok)
        ratios.append(ratio)
        if ratio > 1.0 + REL:
            stats["rejected"] += 1
            if ok:
                stats["rejected_valid"] += 1
            else:
                stats["rejected_invalid"] += 1
                stats["min_ratio_invalid_rejected"] = min(stats["min_ratio_invalid_rejected"], ratio)
        elif ok:
            stats["worst_ratio_valid"] = max(stats["worst_ratio_valid"], ratio)
        return ok

    fifth.is_valid_layout = wrapper
    n = 0
    for _t, _p in fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)), normals):
        n += 1
    fifth.is_valid_layout = orig

    invalid = stats["n"] - stats["valid"]
    print(f"{name}: nets={stats['n']} valid={stats['valid']} invalid={invalid}")
    print(f"  面積和 > 外接矩形 で弾かれる : {stats['rejected']} "
          f"(不正のうち {stats['rejected'] / max(invalid, 1) * 100:.1f}%)")
    print(f"  ⚠️ 有効なのに弾かれた : {stats['rejected_valid']}  (0 であること)")
    print(f"  有効ネットの最大比  : {stats['worst_ratio_valid']:.6f}")
    print(f"  弾かれた不正の最小比: {stats['min_ratio_invalid_rejected']:.6f}")
    hist = [0] * 6
    for r in ratios:
        b = 0 if r <= 1.0 else min(5, int(r))
        hist[b] += 1
    print(f"  比の分布 (<=1, 1-2, 2-3, 3-4, 4-5, >=5): {hist}")


if __name__ == "__main__":
    main()
