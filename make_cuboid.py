import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import fifth


def make_cuboid(w: float, h: float, d: float) -> tuple[np.ndarray, list[list[int]]]:
    """与えられた 幅(w), 高さ(h), 奥行き(d) の直方体の頂点と面リストを生成する。"""
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


def plot_net(polygons: list[Polygon], title: str, filename: str):
    """展開図のポリゴンリストを2Dプロットして保存する。"""
    fig, ax = plt.subplots(figsize=(6, 6))

    all_x, all_y = [], []
    for idx, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        all_x.extend(x)
        all_y.extend(y)
        ax.fill(x, y, alpha=0.3, fc="skyblue", ec="blue", lw=2)
        # 面のIDを重心に描画
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, str(idx), ha="center", va="center", fontsize=12, weight="bold")

    ax.set_aspect("equal", "box")
    ax.set_title(title)

    # 描画範囲の調整
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.close()


def main():
    # テスト対象の直方体の寸法 (幅 1.0, 高さ 1.5, 奥行き 2.0)
    w, h, d = 1.0, 1.5, 2.0
    print(f"直方体寸法: 幅={w}, 高さ={h}, 奥行き={d}")

    vertices, faces = make_cuboid(w, h, d)

    # fifth.py の展開図生成を呼び出す
    nets = fifth.generate_unfolding_nets(vertices, faces)

    print(f"生成された一意な展開図の総数: {len(nets)}")

    # 全ての展開図画像を保存
    for idx, net in enumerate(nets):
        filename = f"cuboids/cuboid_net_{idx + 1}.png"
        plot_net(net, f"Cuboid Net Pattern {idx + 1}", filename)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
