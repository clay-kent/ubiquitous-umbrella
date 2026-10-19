"""`_layout_batch` だけを同一プロセス内で交互に測る（窓ごとに新旧を入れ替える）。

パイプライン全体の A/B は 1 ラウンド 10 秒級で、負荷の揺れが ±7% 乗るため
2〜3% の差は読めない。ここでは展開（`_layout_batch`）の入力（窓ごとの木の列）を
一度だけ作り、同じ入力に対して新実装と HEAD の実装を **窓ごとに順序を入れ替えて**
走らせ、ラウンドごとの合計比の中央値を取る。変更箇所だけを見るので分散が小さい。

    uv run python tools/probe_ab_layout.py stairs 9
"""

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import fifth
from bench_ladder import SOLIDS
from probe_ab_pipeline import load_old_module


def digest(res: object) -> str:
    polys, bnds = res  # type: ignore[misc]
    # ⚠️ shapely.to_wkb は object 配列を返す（要素は bytes）。tobytes() では
    # ポインタが並ぶだけなので、要素ごとに wkb を繋ぐ。
    h = hashlib.sha256()
    for p in np.asarray(polys, dtype=object).ravel():
        h.update(p.wkb)
    h.update(bnds.tobytes())
    return h.hexdigest()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    window = int(sys.argv[3]) if len(sys.argv) > 3 else fifth._LADDER_WINDOW

    old = load_old_module()
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    prep = fifth.prepare_layout(v, faces, normals)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    windows = [trees[i : i + window] for i in range(0, len(trees), window)]
    print(f"{name}: trees={len(trees)} windows={len(windows)} rows/window={window}")

    # 出力が一致することの確認（全窓）。
    for w in windows:
        a = fifth._layout_batch(len(faces), w, prep)
        b = getattr(old, "_layout_batch")(len(faces), w, prep)
        if digest(a) != digest(b):
            raise SystemExit("出力が不一致")

    pairs: list[tuple[float, float]] = []
    for r in range(rounds):
        t_new = 0.0
        t_old = 0.0
        for i, w in enumerate(windows):
            order = ("new", "old") if (r + i) % 2 == 0 else ("old", "new")
            for label in order:
                fn = fifth._layout_batch if label == "new" else getattr(old, "_layout_batch")
                t0 = time.perf_counter()
                fn(len(faces), w, prep)
                dt = time.perf_counter() - t0
                if label == "new":
                    t_new += dt
                else:
                    t_old += dt
        pairs.append((t_new, t_old))
        print(f"    round {r}: new={t_new:.4f}s old={t_old:.4f}s ratio={t_old / t_new:.4f}")
    ratios = sorted(o / n for n, o in pairs)
    med = ratios[len(ratios) // 2]
    print(f"  new min {min(n for n, _o in pairs):.4f}s  old min {min(o for _n, o in pairs):.4f}s"
          f"  median ratio old/new = {med:.4f}x")


if __name__ == "__main__":
    main()
