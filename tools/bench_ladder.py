"""複雑度ラダー付きベンチマーク。現在実装の全探索の速度を測る。

指標は「面グラフの全域木の全列挙 + 一意化(正規形) + 有望置換(レイアウト・重なり判定・指標)」
の合計所要時間。指標本体（外接矩形最小/外周長）の並べ替えは最後に行う。

このベンチは「実験用のより現実的で複雑なな立体」を複雑度の低い順に並べ、
1つあたり BUDGET 秒の上限で打ち切る（打ち切り時は 'TIMEOUT' 表記で状況を報告）。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth


# 立体定義 -------------------------------------------------------------------

def box(w=1.0, h=1.0, d=1.0):
    v = np.array([
        [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
        [0, 0, d], [w, 0, d], [w, h, d], [0, h, d],
    ], dtype=float)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


def prism(n, r=1.0, h=1.0):
    v = [
        [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), 0.0]
        for i in range(n)
    ] + [
        [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), h]
        for i in range(n)
    ]
    f = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    f += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return np.array(v, dtype=float), f


def pyramid(n, r=1.0, h=1.0):
    v = [
        [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), 0.0]
        for i in range(n)
    ] + [[0.0, 0.0, h]]
    f = [list(range(n - 1, -1, -1))] + [[i, (i + 1) % n, n] for i in range(n)]
    return np.array(v, dtype=float), f


def frustum(n=4, r0=1.0, r1=0.5, h=1.0):
    """n 角錐台（切頭角錐）。屋根の狭い「ビルの屋上を切った」現実的な形。"""
    v = [
        [r0 * np.cos(2 * np.pi * i / n), r0 * np.sin(2 * np.pi * i / n), 0.0]
        for i in range(n)
    ] + [
        [r1 * np.cos(2 * np.pi * i / n), r1 * np.sin(2 * np.pi * i / n), h]
        for i in range(n)
    ]
    top = list(range(n, 2 * n))
    f = [list(range(n - 1, -1, -1)), top]
    f += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return np.array(v, dtype=float), f


def l_shape():
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
    v = np.array([[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts])
    f = [list(range(5, -1, -1)), list(range(6, 12))]
    f += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    return v, f


def stair_case_3():
    """3段の階段を含む直方体（正方形柱を3つ組み合わせた形、6 + 8 = ... 面が増える）。"""
    pts = [
        (0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2), (1, 3), (0, 3),
    ]
    v = np.array([[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts])
    k = len(pts)
    f = [list(range(k - 1, -1, -1)), list(range(k, 2 * k))]
    f += [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    return v, f


SOLIDS = {
    "box": box(1.0, 1.5, 2.0),
    "prism5": prism(5),
    "prism6": prism(6),
    "pyr4": pyramid(4),
    "frustum4": frustum(4),
    "lshape": l_shape(),
    "stairs": stair_case_3(),
    "prism7": prism(7),
}


# ベンチ本体 ------------------------------------------------------------------


def bench_solid(name, vertices, faces, timeout):
    start = time.perf_counter()
    n_faces = len(faces)
    edges = fifth.build_adjacency_graph(faces)
    n_trees = fifth.count_spanning_trees(edges, n_faces)

    result = {
        "name": name,
        "faces": n_faces,
        "trees": n_trees,
        "unique": None,
        "best_bbox_area": None,
        "best_net_id": None,
        "elapsed": None,
        "status": "ok",
    }

    normals = fifth.compute_normals(vertices, faces)
    trees = fifth.enumerate_spanning_trees(edges, n_faces)
    nets = []

    for tree, polys in fifth.iter_unique_nets(vertices, faces, trees, normals):
        nets.append((tree, polys))
        if time.perf_counter() - start > timeout:
            result["elapsed"] = time.perf_counter() - start
            result["status"] = "TIMEOUT"
            break
    else:
        result["elapsed"] = time.perf_counter() - start
    result["unique_seen"] = len(nets)

    if nets:
        shared_of = fifth.build_shared_of(faces)
        scored = [
            fifth.net_metrics(polys, tree, vertices, shared_of).bounding_box_area
            for tree, polys in nets
        ]
        i = int(np.argmin(scored))
        result["best_bbox_area"] = scored[i]
        result["best_net_id"] = i + 1
        # 指標の並べ替え込み込みも測る
        t2 = time.perf_counter()
        order = sorted(range(len(scored)), key=lambda k: scored[k])
        result["sort_ms"] = (time.perf_counter() - t2) * 1000.0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0, help="1立体あたりの上限秒")
    parser.add_argument("--solids", default=None, help="指定した名前の立体だけ測る")
    args = parser.parse_args()

    names = args.solids.split(",") if args.solids else list(SOLIDS)
    print(f"{'name':10s} {'faces':>5s} {'trees':>9s} {'unique':>9s} "
          f"{'best_bbox':>10s} {'elapsed':>9s}  status")
    for name in names:
        v, f = SOLIDS[name]
        r = bench_solid(name, v, f, args.timeout)
        unique = str(r["unique_seen"]) + ("+" if r["status"] == "TIMEOUT" else "")
        print(
            f"{r['name']:10s} {r['faces']:5d} {r['trees']:9,d} {unique:>9s} "
            f"{(str(round(r['best_bbox_area'], 4)) if r['best_bbox_area'] else '-'):>10s} "
            f"{r['elapsed']:9.2f}s  {r['status']}"
        )


if __name__ == "__main__":
    main()
