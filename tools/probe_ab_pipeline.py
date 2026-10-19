"""パイプライン全体（iter_unique_nets）の新旧を **同一プロセス内で交互に** 測る。

外部ロードで所要時間が数倍揺れる環境では「別プロセスで順番に測る」比較は
当てにならない。ここでは HEAD の fifth.py を別モジュールとして読み込み、
同じ入力に対して新実装と旧実装を交互に走らせ、**比**を取る。負荷は両方に
等しく乗るので比だけが意味を持つ。

出力は「新 / 旧 / 比」と、返る展開図列（木と座標）がビット単位で一致するか。

    uv run python tools/probe_ab_pipeline.py stairs 4
"""

import hashlib
import importlib.util
import itertools
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import fifth
from bench_ladder import SOLIDS


def load_old_module() -> object:
    """HEAD の fifth.py を別名のモジュールとして読み込む。"""
    blob = subprocess.run(
        ["git", "show", "HEAD:fifth.py"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    tmp = Path(tempfile.gettempdir()) / "opencode" / "fifth_head.py"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("fifth_head", tmp)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod: object, v: object, faces: object, trees: list, normals: object) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    for tree, polys in getattr(mod, "iter_unique_nets")(v, faces, iter(trees), normals):
        h.update(np.asarray(tree, dtype=np.int64).tobytes())
        for p in polys:
            h.update(np.asarray(p.exterior.coords, dtype=np.float64).tobytes())
        n += 1
    return n, h.hexdigest()[:24]


def instrument(mod: object) -> dict[str, float]:
    """`_batch_stabilize` の累計時間を持ち帰るためのラッパを噛ませる。"""
    acc = {"stabilize": 0.0, "calls": 0.0}
    orig = getattr(mod, "_batch_stabilize")

    def wrapper(ctx: object, partner: object, flip: object) -> object:
        t0 = time.perf_counter()
        r = orig(ctx, partner, flip)
        acc["stabilize"] += time.perf_counter() - t0
        acc["calls"] += 1
        return r

    mod._batch_stabilize = wrapper  # type: ignore[attr-defined]
    return acc


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    old = load_old_module()
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    if limit:
        trees = trees[:limit]
    print(f"{name}: trees={len(trees)} rounds={rounds}")

    n_new, d_new = run(fifth, v, faces, trees, normals)
    n_old, d_old = run(old, v, faces, trees, normals)
    print(f"  valid new={n_new} old={n_old}  identical={n_new == n_old and d_new == d_old}")

    acc_new = instrument(fifth)
    acc_old = instrument(old)

    # ⚠️ 毎ラウンド同じ順で走らせると「先に走ったほうが有利/不利」という
    # 系統誤差がそのまま比に出る（1 ラウンド 5 秒級なので負荷のドリフトも入る）。
    # 奇数ラウンドで順序を入れ替え、**ペアごとの比の中央値**を取る。
    pairs: list[tuple[float, float]] = []
    for r in range(rounds):
        order = [("new", fifth, acc_new), ("old", old, acc_old)]
        if r % 2 == 1:
            order.reverse()
        dts: dict[str, float] = {}
        for label, mod, acc in order:
            acc["stabilize"] = 0.0
            acc["calls"] = 0.0
            t0 = time.process_time()
            run(mod, v, faces, trees, normals)
            dts[label] = time.process_time() - t0
            if label == "new":
                dts["stab_new"] = acc["stabilize"]
            else:
                dts["stab_old"] = acc["stabilize"]
            print(f"    {label}: {dts[label]:7.3f}s   stabilize={acc['stabilize']:6.3f}s "
                  f"({acc['calls']:.0f} calls)")
        pairs.append((dts["new"], dts["old"]))
    ratios = sorted(o / n for n, o in pairs)
    med = ratios[len(ratios) // 2]
    print(f"  total ratio old/new per round: "
          f"{' '.join(f'{o / n:.3f}' for n, o in pairs)}  median={med:.3f}x")
    print(f"  stabilize new {min(p[0] for p in pairs):.3f}s+   old "
          f"{min(p[1] for p in pairs):.3f}s+")


if __name__ == "__main__":
    main()
