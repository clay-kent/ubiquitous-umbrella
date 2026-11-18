"""任意の多面体から一意な展開図を取り出し、PNG 画像として保存するお試し用スクリプト。

    python make_nets.py                 # 既定の立体（立方体）の一意な展開図を全部描く
    python make_nets.py --solid lshape  # プリセットを指定
    python make_nets.py --solid prism5 --top-n 5 --sort-by perimeter
    python make_nets.py --review        # のりしろ確認用のレビューシートを 1 枚だけ作る

API サーバー (api_server.py) と同じ fifth.py を呼び出すので、同じ結果になる。
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
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


def lathe(
    profile: list[tuple[float, float]], n: int
) -> tuple[np.ndarray, list[list[int]]]:
    """(半径, 高さ) の輪郭を n 角形断面で回転させた立体。

    輪郭の隣り合う 2 点を結ぶ四角形を側面として n 枚ずつ並べる。
    輪郭を 1 点増やすと面数が n 枚増え、展開図の候補数が急増する。
    """
    angles = 2 * np.pi * np.arange(n) / n
    vertices = np.array(
        [[r * np.cos(a), r * np.sin(a), z] for r, z in profile for a in angles],
        dtype=float,
    )
    faces = [list(range(n - 1, -1, -1))]
    for k in range(len(profile) - 1):
        lo, hi = k * n, (k + 1) * n
        faces += [
            [lo + i, lo + (i + 1) % n, hi + (i + 1) % n, hi + i] for i in range(n)
        ]
    top = (len(profile) - 1) * n
    faces.append(list(range(top, top + n)))
    return vertices, faces


def namacha_bottle() -> tuple[np.ndarray, list[list[int]]]:
    """キリン 生茶 525mL ペットボトルのローポリ近似。

    市販 500mL ペットボトルの実寸は胴径 6.7〜6.9cm・全高 21cm 前後。
    指定の 6.9 はこの胴径（直径）と一致するため、半径 3.45cm として扱う。
    直胴部（細くなり始めるまで）は約 12cm、口栓部は 28mm（半径 1.4cm）、
    キャップは 30mm（半径 1.5cm）。

    展開図の候補数は面数に対して指数的に増えるので、断面は三角形の
    3 バンド（胴・肩・首）に簡略化している。四角断面にすると面数が
    14 枚になり、全域木が約 256 万通りまで増えて現実的な時間で終わらない。
    """
    profile = [
        (3.3, 0.0),  # 底面
        (3.45, 12.0),  # 胴（ラベル部）の上端
        (1.4, 17.0),  # 肩の上端（口栓部の付け根）
        (1.5, 21.0),  # キャップ天面
    ]
    return lathe(profile, 3)


def namacha_box() -> tuple[np.ndarray, list[list[int]]]:
    """生茶 525mL の円形ペットボトルが入る直方体の箱。

    ボトルは胴径 6.9cm・全高 21cm。細くなり始めるまでの直胴部は約 12cm。
    箱はボトル全体が入る最小の直方体とし、クリアランスは考えず
    6.9 × 6.9 × 21cm にした。
    """
    return box(6.9, 21.0, 6.9)


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
    "namacha": namacha_bottle,
    "namacha_box": namacha_box,
    "cutout": cutout,
}


def _draw_net(
    ax: Axes,
    polygons: list[Polygon],
    glue_tabs: list[fifth.GlueTab],
    title: str,
) -> None:
    """1 枚の展開図を ax に描く。

    のりしろを面の**上**に描くのがポイント。旧描画ではのりしろを先に
    薄く塗ってから面を重ねていたため、面と重なったのりしろが隠れ、
    人間が見ても配置を確認できなかった。
    """
    all_x: list[float] = []
    all_y: list[float] = []

    for idx, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        all_x.extend(x)
        all_y.extend(y)
        ax.fill(x, y, fc="#cfe3ff", ec="#1f77b4", lw=1.5, zorder=1)
        ax.text(
            poly.centroid.x,
            poly.centroid.y,
            str(idx),
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            zorder=3,
        )

    for tab in glue_tabs:
        tx = [p[0] for p in tab.vertices_2d] + [tab.vertices_2d[0][0]]
        ty = [p[1] for p in tab.vertices_2d] + [tab.vertices_2d[0][1]]
        all_x.extend(tx)
        all_y.extend(ty)
        ax.fill(
            tx,
            ty,
            fc="#ffb347",
            ec="#d2691e",
            lw=1.2,
            alpha=0.85,
            hatch="//",
            zorder=2,
        )

    ax.set_aspect("equal", "box")
    ax.set_title(f"{title}  (tabs={len(glue_tabs)})", fontsize=10)
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, linestyle="--", alpha=0.4)


LEGEND_HANDLES = [
    Patch(facecolor="#cfe3ff", edgecolor="#1f77b4", label="face"),
    Patch(
        facecolor="#ffb347",
        edgecolor="#d2691e",
        hatch="//",
        label="glue tab",
    ),
]


def plot_net(
    polygons: list[Polygon],
    glue_tabs: list[fifth.GlueTab],
    title: str,
    filename: str,
) -> None:
    """展開図 1 枚を PNG で保存する。"""
    _fig, ax = plt.subplots(figsize=(6, 6))
    _draw_net(ax, polygons, glue_tabs, title)
    _fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=2, frameon=False)
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.close()


def _net_for_review(solid: str) -> tuple[str, list[Polygon], list[fifth.GlueTab]]:
    """レビュー用に、のりしろ付きの代表的な展開図を 1 枚つくる。

    レビューは糊代の付き方を見るためのものなので、既定（糊代なし）ではなく
    従来の大きさを明示して選ぶ。選択自体は API と同じ ``select_nets``。
    """
    vertices, faces = SOLIDS[solid]()
    selection = fifth.select_nets(
        vertices,
        faces,
        fifth.NetSelectionOptions(
            sort_by="area", top_n=1, tab_size=fifth.TabSize.legacy()
        ),
    )
    candidate = selection.nets[0]
    return solid, candidate.polys, list(candidate.tabs)


REVIEW_SOLIDS = ["box", "tetra", "prism5", "pyr4", "lshape", "namacha_box"]


def plot_review_sheet(
    entries: list[tuple[str, list[Polygon], list[fifth.GlueTab]]],
    filename: str,
) -> None:
    """代表的な立体の展開図を 1 枚のシートに並べて保存する。

    人間が「のりしろが実際にどう付くか」を一度で確認するための図。
    """
    ncols = 3
    nrows = (len(entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    flat = list(np.asarray(axes).reshape(-1))
    for ax, (solid, polys, tabs) in zip(flat, entries):
        _draw_net(ax, polys, tabs, solid)
    for ax in flat[len(entries) :]:
        ax.axis("off")
    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(filename, dpi=110)
    plt.close(fig)


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
        choices=["area", "perimeter", "folds"],
        help="並べ替えの基準（既定: area）",
    )
    parser.add_argument(
        "--outdir",
        default="nets",
        help="画像の保存先ディレクトリ（既定: nets）",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="代表立体の展開図を 1 枚のレビュー用シート (review.png) にまとめて保存する",
    )
    # API のクエリ既定は 0（糊代なし）だが、画像は糊代を見るための道具なので
    # CLI では従来の大きさを既定にする。解釈は fifth.TabSize に一本化。
    parser.add_argument(
        "--tab-size-min",
        type=float,
        default=0.0,
        help="のりしろの最小奥行き比（既定 0 = 最大値の 10%%）",
    )
    parser.add_argument(
        "--tab-size-max",
        type=float,
        default=0.25,
        help="のりしろの最大奥行き比（既定 0.25。0 で糊代なし）",
    )
    args = parser.parse_args()

    if args.review:
        os.makedirs(args.outdir, exist_ok=True)
        entries = [_net_for_review(solid) for solid in REVIEW_SOLIDS]
        filename = os.path.join(args.outdir, "review.png")
        plot_review_sheet(entries, filename)
        print(f"Saved: {filename}")
        for solid, _polys, tabs in entries:
            print(f"  {solid}: のりしろ {len(tabs)} 枚")
        return

    vertices, faces = SOLIDS[args.solid]()
    edges = fifth.build_adjacency_graph(faces)
    n_trees = fifth.count_spanning_trees(edges, len(faces))
    print(f"立体: {args.solid}  頂点={len(vertices)}  面={len(faces)}")
    print(f"全域木: {n_trees} 通り")

    try:
        size = fifth.TabSize.of(args.tab_size_min, args.tab_size_max)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selection = fifth.select_nets(
        vertices,
        faces,
        fifth.NetSelectionOptions(
            sort_by=args.sort_by,
            top_n=args.top_n,
            tab_size=size,
        ),
    )

    suffix = f"（上位 {args.top_n} 個のみ保存）" if args.top_n else ""
    print(f"一意な展開図: {selection.total_unique_nets} 個{suffix}")
    print(f"保存: {len(selection.nets)} 枚")
    os.makedirs(args.outdir, exist_ok=True)

    for idx, candidate in enumerate(selection.nets, start=1):
        filename = os.path.join(args.outdir, f"{args.solid}_net_{idx}.png")
        m = candidate.metrics
        plot_net(
            candidate.polys,
            list(candidate.tabs),
            f"{args.solid} net {idx}  area={m.bounding_box_area:g}",
            filename,
        )
        print(
            f"Saved: {filename}  "
            f"(外接矩形面積 {m.bounding_box_area:g}, "
            f"外周長 {m.perimeter:g}, 折る回数 {m.fold_count})"
        )


if __name__ == "__main__":
    main()
