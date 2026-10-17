"""iter_unique_nets の3フェーズ（列挙/署名/展開+重なり）を分けて計測する。"""

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "prism7"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)

    t0 = time.perf_counter()
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    t_enum = time.perf_counter() - t0

    ctx = fifth.RibbonContext(v, faces)
    mctx = fifth.RibbonContext(v, [list(reversed(x)) for x in faces])
    prep = fifth.prepare_layout(v, faces, normals)

    t_sig = 0.0
    t_layout = 0.0
    t_valid = 0.0
    seen = set()
    n_unique = 0
    n_valid = 0

    rest = iter(trees)
    while True:
        t0 = time.perf_counter()
        window = list(itertools.islice(rest, fifth._LADDER_WINDOW))
        if not window:
            break
        key_pairs = fifth._batch_canonical_keys(ctx, mctx, window)
        t_sig += time.perf_counter() - t0

        for tree, (sig, mirror_sig) in zip(window, key_pairs):
            key = min(sig, mirror_sig)
            if key in seen:
                continue
            seen.add(key)
            n_unique += 1
            t0 = time.perf_counter()
            polys = fifth.layout_tree(v, faces, normals, tree, prep)
            t_layout += time.perf_counter() - t0
            t0 = time.perf_counter()
            ok = fifth.is_valid_layout(polys)
            t_valid += time.perf_counter() - t0
            if ok:
                n_valid += 1

    print(f"{name}: trees={len(trees)} unique_keys={n_unique} valid={n_valid}")
    print(f"  enumerate           : {t_enum:8.3f}s")
    print(f"  signatures (batch)  : {t_sig:8.3f}s")
    print(f"  layout_tree         : {t_layout:8.3f}s")
    print(f"  is_valid_layout     : {t_valid:8.3f}s")
    print(f"  total               : {t_enum + t_sig + t_layout + t_valid:8.3f}s")


if __name__ == "__main__":
    main()