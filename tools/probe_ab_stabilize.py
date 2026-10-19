"""`_batch_stabilize` の新旧実装を **同一プロセス内で交互に** 測る。

外部ロードで時間が 5 倍近く揺れる環境では「別プロセスで順番に測る」比較は
当てにならない（計測のたびに条件が変わる）。ここでは HEAD の fifth.py を
別モジュールとして読み込み、**同じ入力**で新実装と旧実装をラウンドロビンに
呼んで、それぞれの最小値を比べる。負荷は両方に等しく乗るので比が意味を持つ。

    uv run python tools/probe_ab_stabilize.py [solid] [rows] [rounds]

出力は「新 / 旧 / 比」と、出力（色と色数）がビット単位で一致するか。
"""

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
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT).decode().strip()
    blob = subprocess.run(
        ["git", "show", "HEAD:fifth.py"], cwd=root, check=True, capture_output=True
    ).stdout
    tmp = Path(tempfile.gettempdir()) / "opencode" / "fifth_head.py"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("fifth_head", tmp)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    old = load_old_module()
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    trees = list(itertools.islice(fifth.enumerate_spanning_trees(edges, len(faces)), n_rows))
    if len(trees) < n_rows:
        raise SystemExit(f"{name}: 木が {len(trees)} 本しかない")
    ctx = fifth.RibbonContext(v, faces)
    edges_arr = np.asarray(trees, dtype=np.int64)
    positions = ctx.adj_pos_matrix[edges_arr[:, :, 0], edges_arr[:, :, 1]]
    partner, flip = fifth._glue_batch(ctx, positions)

    print(f"{name}: rows={n_rows} n_nodes={ctx.n_nodes} rounds={rounds}")

    def run(mod: object) -> tuple[np.ndarray, np.ndarray]:
        return getattr(mod, "_batch_stabilize")(ctx, partner, flip)

    # まず一致確認（新と旧で色・色数が完全に同じか）
    c_new, k_new = run(fifth)
    c_old, k_old = run(old)
    same = bool(np.array_equal(c_new, c_old) and np.array_equal(k_new, k_old))
    print(f"  identical output: {same}")

    # ⚠️ Windows の process_time は分解能が約 15.6ms。1 回の呼び出しでは 0 と
    # しか出ないので、内側で繰り返して分解能を超えさせる。刻み 1 つが
    # 「15.6ms / reps」に相当するので、比を 1% の精度で見るには 2 秒ぶん回す。
    reps = 1
    while True:
        _t0 = time.process_time()
        for _ in range(reps):
            run(fifth)
        if time.process_time() - _t0 > 2.0:
            break
        reps *= 2

    best = {"new": float("inf"), "old": float("inf")}
    for _ in range(rounds):
        for label, mod in (("new", fifth), ("old", old)):
            t0 = time.process_time()
            for _ in range(reps):
                run(mod)
            dt = (time.process_time() - t0) / reps
            best[label] = min(best[label], dt)
    print(f"  reps={reps} (1 回あたり)")
    print(f"  new {best['new'] * 1e3:.3f}ms   old {best['old'] * 1e3:.3f}ms   "
          f"ratio old/new = {best['old'] / best['new']:.3f}x")


if __name__ == "__main__":
    main()
