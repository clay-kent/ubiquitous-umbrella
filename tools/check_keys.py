"""バッチ正規形の一意数が既知値と一致するか確認する（回路の検査用）。

これは pytest ではなく「開発中の計測ツール」。回帰として残すなら
test_fifth.py 側へ移す（tests/ は pytest の置き場所）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
import make_nets


CASES = [
    ("cube", lambda: make_nets.box(1.0, 1.0, 1.0), 11),
    ("box(1,1.5,2)", lambda: make_nets.box(1.0, 1.5, 2.0), 54),
    ("tetra", make_nets.tetrahedron, 2),
    ("prism3", lambda: make_nets.prism(3), 9),
    ("prism4", lambda: make_nets.prism(4), 29),
    ("prism5", lambda: make_nets.prism(5), 99),
    ("pyr4", lambda: make_nets.pyramid(4), 8),
    ("pyr5", lambda: make_nets.pyramid(5), 15),
    # lshape は「重なり除去前」の signature 一意数が 4068（refactoring_memo_v3 PR4 の
    # 実測値と一致）。重なりを持つ展開図は iter_unique_nets の is_valid_layout で
    # 落ちるので、正しく除外されれば最終的な一意数は 1516。
    ("lshape", make_nets.l_shape, 4068),
    ("prism6", lambda: make_nets.prism(6), 354),
    # cutout は全域木 4000 万本なので全列挙が間に合わない（計測対象外）
]


def main():
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    ok = True
    for name, builder, expect in CASES:
        v, f = builder()
        faces = fifth.orient_faces_outward(v, f)
        ctx = fifth.RibbonContext(v, faces)
        mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
        edges = fifth.build_adjacency_graph(faces)
        n_trees = fifth.count_spanning_trees(edges, len(faces))
        keys = set()
        tree_iter = fifth.enumerate_spanning_trees(edges, len(faces))
        import itertools

        while True:
            window_trees = list(itertools.islice(tree_iter, window))
            if not window_trees:
                break
            for a, b in fifth._batch_canonical_keys(ctx, mctx, window_trees):
                keys.add(min(a, b))
        got = len(keys)
        if expect is None:
            print(f"{name:14s} trees={n_trees:12,d} batch_keys={got:8,d} expect=(unknown)")
            continue
        status = "OK" if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"{name:14s} trees={n_trees:12,d} batch_keys={got:8,d} expect={expect:8,d} {status}")
    print("ALL OK" if ok else "SOME MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())