"""任意の多面体から一意な展開図を取り出し、PNG 画像として保存するお試し用スクリプト。

    python make_nets.py                 # 既定の立体（立方体）の一意な展開図を全部描く
    python make_nets.py --solid lshape  # プリセットを指定
    python make_nets.py --solid prism5 --top-n 5 --sort-by perimeter

API サーバー (api_server.py) と同じ fifth.py を呼び出すので、同じ結果になる。
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import fifth


def box(w: float = 1.0, h: float = 1.0, d: float = 1.0) -> tuple[np.ndarray, list[list[int]]]:
    """幅 w・高さ h・奥行き d の直方体。"""
    vertices = np.array(
        [
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [0, 0, d],
            [w, 0, d],
            [w, h, d],
            [0, h, d],
        ],
        dtype=float,
    )
    faces = [
        [0, 3, 2, 1],  # 底面
        [4, 5, 6, 7],  # 上面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 後面
        [3, 0, 4, 7],  # 左面
    ]
    return vertices, faces


def tetrahedron() -> tuple[np.ndarray, list[list[int]]]:
    """正四面体。"""
    return (
        np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float),
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    )


def prism(n: int, r: float = 1.0, h: float = 1.0) -> tuple[np.ndarray, list[list[int]]]:
    """n 角柱。"""
    return (
        np.array(
            [
                [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), 0.0]
                for i in range(n)
            ]
            + [
                [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), h]
                for i in range(n)
            ],
            dtype=float,
        ),
        [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
        + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)],
    )


def pyramid(n: int, r: float = 1.0, h: float = 1.0) -> tuple[np.ndarray, list[list[int]]]:
    """n 角錐。"""
    return (
        np.array(
            [
                [r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), 0.0]
                for i in range(n)
            ]
            + [[0.0, 0.0, h]],
            dtype=float,
        ),
        [list(range(n - 1, -1, -1))] + [[i, (i + 1) % n, n] for i in range(n)],
    )


def l_shape() -> tuple[np.ndarray, list[list[int]]]:
    """凹角（270 度の内角）を持つ L 字柱。非凸立体の代表例。"""
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)]
    vertices = np.array([[x, y, 0.0] for x, y in pts] + [[x, y, 1.0] for x, y in pts])
    faces = [list(range(5, -1, -1)), list(range(6, 12))]
    faces += [[i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i] for i in range(6)]
    return vertices, faces


def cutout() -> tuple[np.ndarray, list[list[int]]]:
    """立方体を貫通する四角い穴つき（genus=1）。

    全域木が 4000 万通りあり、さらに面の数が多いため展開図の列挙には時間がかかる。
    """
    outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
    inner = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    loop = outer + inner
    top = len(loop)
    vertices = np.array([[x, y, z] for z in (0.0, 1.0) for x, y in loop], dtype=float)
    faces: list[list[int]] = []
    for k in range(4):
        j = (k + 1) % 4
        faces.append([k, j, top + j, top + k])  # 外側の側面
        faces.append([4 + k, 4 + j, top + 4 + j, top + 4 + k])  # 穴の側面
    for base, flip in ((0, False), (top, True)):
        for k in range(4):
            j = (k + 1) % 4
            quad = [base + k, base + j, base + 4 + j, base + 4 + k]
            faces.append(list(reversed(quad)) if flip else quad)
    return vertices, faces


SOLIDS = {
    "box": lambda: box(1.0, 1.5, 2.0),
    "tetra": tetrahedron,
    "prism3": lambda: prism(3),
    "prism4": lambda: prism(4),
    "prism5": lambda: prism(5),
    "pyr4": lambda: pyramid(4),
    "pyr5": lambda: pyramid(5),
    "lshape": l_shape,
    "cutout": cutout,
}


def plot_net(polygons: list[Polygon], title: str, filename: str) -> None:
    """展開図のポリゴンリストを 2D プロットして保存する。"""
    _fig, ax = plt.subplots(figsize=(6, 6))

    all_x: list[float] = []
    all_y: list[float] = []
    for idx, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        all_x.extend(x)
        all_y.extend(y)
        ax.fill(x, y, alpha=0.3, fc="skyblue", ec="blue", lw=2)
        ax.text(
            poly.centroid.x,
            poly.centroid.y,
            str(idx),
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )

    ax.set_aspect("equal", "box")
    ax.set_title(title)
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="任意の多面体の展開図を PNG で保存する。")
    parser.add_argument(
        "--solid",
        default="box",
        choices=sorted(SOLIDS),
        help="描画する立体（既定: box）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="保存する枚数。0 なら一意な展開図をすべて保存する。",
    )
    parser.add_argument(
        "--sort-by",
        default="area",
        choices=["area", "perimeter"],
        help="並べ替えの基準（既定: area）",
    )
    parser.add_argument(
        "--outdir",
        default="nets",
        help="画像の保存先ディレクトリ（既定: nets）",
    )
    args = parser.parse_args()

    vertices, faces = SOLIDS[args.solid]()
    normals = fifth.compute_normals(vertices, faces)
    edges = fifth.build_adjacency_graph(faces)
    n_trees = fifth.count_spanning_trees(edges, len(faces))
    print(f"立体: {args.solid}  頂点={len(vertices)}  面={len(faces)}")
    print(f"全域木: {n_trees} 通り")

    trees = fifth.enumerate_spanning_trees(edges, len(faces))
    nets = list(fifth.iter_unique_nets(vertices, faces, trees, normals))

    metrics = []
    shared_of = fifth.build_shared_of(faces)
    for tree, polys in nets:
        m = fifth.net_metrics(polys, tree, vertices, shared_of)
        metrics.append((m, polys))
    key = (
        (lambda s: s[0].bounding_box_area)
        if args.sort_by == "area"
        else (lambda s: s[0].perimeter)
    )
    metrics.sort(key=key)

    if args.top_n > 0:
        metrics = metrics[: args.top_n]

    suffix = f"（上位 {args.top_n} 個のみ保存）" if args.top_n else ""
    print(f"一意な展開図: {len(metrics)} 個{suffix}")
    os.makedirs(args.outdir, exist_ok=True)

    for idx, (m, polys) in enumerate(metrics, start=1):
        filename = os.path.join(args.outdir, f"{args.solid}_net_{idx}.png")
        plot_net(polys, f"{args.solid} net {idx}  area={m.bounding_box_area:g}", filename)
        print(
            f"Saved: {filename}  "
            f"(外接矩形面積 {m.bounding_box_area:g}, 外周長 {m.perimeter:g})"
        )


if __name__ == "__main__":
    main()
