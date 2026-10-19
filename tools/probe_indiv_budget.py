"""個別化-細分の呼び出し回数と消費予算のピークを測る。

`_WL_MIN_ROUNDS` を入れると WL がより細かく分割されてから止まるため、
個別化に落ちる行が減る（lshape では 9 -> 0）。その確認用。

    uv run python tools/probe_indiv_budget.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from test_fifth import cube_b, l_shape_b, prism4_b, prism5_b, tetrahedron_b


def main() -> None:
    for builder in (cube_b, tetrahedron_b, prism4_b, prism5_b, l_shape_b):
        v, f = builder()
        faces = fifth.orient_faces_outward(v, f)
        normals = fifth.compute_normals(v, f)
        trees = fifth.enumerate_spanning_trees(
            fifth.build_adjacency_graph(faces), len(faces)
        )
        peak = {"v": 0, "calls": 0}

        original = fifth._individualize_refine

        def counting(nxt, partner, flip, color, budget, _o=original):
            peak["calls"] += 1
            try:
                return _o(nxt, partner, flip, color, budget)
            finally:
                peak["v"] = max(peak["v"], budget[0])

        fifth._individualize_refine = counting
        try:
            list(fifth.iter_unique_nets(v, faces, trees, normals))
        finally:
            fifth._individualize_refine = original

        print(
            f"{builder.__name__:14s} calls={peak['calls']:5d}"
            f" peak_budget={peak['v']:6d}"
            f" (SEARCH_BUDGET/100={fifth.SEARCH_BUDGET // 100})"
        )


if __name__ == "__main__":
    main()
