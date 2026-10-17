"""全域木が膨大だが一意な展開図が少ない、立体の調査。

cross（14面 / 全域木 4,380万本）では 891 一意しか出ない。
大半が「重なり」で落ちるなら、**重なりを先に判定したほうが速い**可能性はある。
ただし重なり判定にはレイアウトが必要で、レイアウトは木ごとに違うので
そのままでは前倒しできない。ここでは比率を測るだけ。
"""

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_next import CASES


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cross"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    v, f = CASES[name]()
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)
    n_trees = fifth.count_spanning_trees(edges, len(faces))

    t0 = time.perf_counter()
    n_sig = 0
    n_valid = 0
    n_overlap = 0
    trees = itertools.islice(
        fifth.iter_unique_nets(v, faces, fifth.enumerate_spanning_trees(edges, len(faces)),
                               normals),
        limit,
    )
    for _ in trees:
        n_valid += 1
    elapsed = time.perf_counter() - t0
    print(
        f"{name}: faces={len(faces)} trees={n_trees:,} "
        f"valid_within_limit={n_valid:,} elapsed={elapsed:.1f}s"
    )
    print(f"  → {limit:,} 本の展開図を {elapsed:.1f}s で処理 = {elapsed/limit*1e6:.1f} us/本")


if __name__ == "__main__":
    main()