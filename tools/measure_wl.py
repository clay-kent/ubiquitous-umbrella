"""WL refinement の内訳を測る: 1 Signaturerefinement の平均回数と、
plain WL のまま全色が別色になる割合（= 個別化が不要な割合）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def instrumented(name):
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    trees = iter(
        fifth.enumerate_spanning_trees(
            fifth.build_adjacency_graph(faces), len(faces)
        )
    )

    stats = {
        "sig": 0,            # シグネチャ計算回数
        "refine": 0,         # _individualize_refine 呼び出し回数
        "wl_rounds": 0,      # WL 試行ラウンド数（全体）
        "plain_complete": 0, # 最初の refine 呼び出しで fully-refined になった数
        "first": True,
    }

    orig_refine = fifth._individualize_refine
    orig_wl = fifth._wl_refine

    def counting_wl(nxt, partner, flip, color):
        out = orig_wl(nxt, partner, flip, color)
        stats["wl_rounds"] += 1
        return out

    def counting_refine(nxt, partner, flip, color, budget):
        stats["refine"] += 1
        first = stats["refine"] == stats["sig"]  # このシグネチャで最初の refine 呼び出し
        # fully-refined 判定は _wl_refine が止まった色数で見る
        result = orig_refine(nxt, partner, flip, color, budget)
        if first and len(set(result)) == len(nxt):
            stats["plain_complete"] += 1
        return result

    fifth._wl_refine = counting_wl
    fifth._individualize_refine = counting_refine

    seen = 0
    try:
        for _ in fifth.iter_unique_nets(v, faces, trees, normals):
            seen += 1
            stats["sig"] += 2  # ctx + mirror の2回
    finally:
        fifth._wl_refine = orig_wl
        fifth._individualize_refine = orig_refine

    n_sig = stats["sig"]
    print(
        f"{name}: unique={seen} signatures={n_sig} "
        f"refine_calls={stats['refine']:.0f} "
        f"plain_WLで完成: {stats['plain_complete']}/{n_sig} "
        f"({100*stats['plain_complete']/n_sig:.1f}%) "
        f"WLラウンド合計={stats['wl_rounds']} "
        f"(シグネチャあたり {stats['wl_rounds']/n_sig:.1f} ラウンド)"
    )
    print(f"  (discarded trees are not signature-uniform; unique nets = {seen})")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["prism6", "lshape"]:
        instrumented(name)
