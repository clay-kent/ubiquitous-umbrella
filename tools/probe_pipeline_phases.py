"""現行コードのフェーズ別の内訳を、関数をラップして測る。

`tools/timing_parts.py` はフェーズを手で書き直しているためコードとずれる。
ここは `iter_unique_nets` をそのまま走らせて、呼ばれる関数の累計時間だけを
取り出す（列挙・署名・展開・重なり判定）。壁時計で測るが、同一プロセス内の
相対比なので負荷が乗っても順位は保たれる。

    uv run python tools/probe_pipeline_phases.py stairs
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS

TARGETS = ("_batch_canonical_keys", "_layout_batch", "_valid_batch")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    normals = fifth.compute_normals(v, faces)
    edges = fifth.build_adjacency_graph(faces)

    # 列挙はジェネレータなので、ラップすると「最初の next まで」しか測れない。
    # 先に実体化して別に測る（timing_parts.py と同じ扱い）。
    t0 = time.perf_counter()
    trees = list(fifth.enumerate_spanning_trees(edges, len(faces)))
    t_enum = time.perf_counter() - t0

    acc = {k: 0.0 for k in TARGETS}
    calls = {k: 0 for k in TARGETS}

    def wrap(name_: str) -> None:
        orig = getattr(fifth, name_)

        def wrapper(*a: object, **kw: object) -> object:
            t = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                acc[name_] += time.perf_counter() - t
                calls[name_] += 1

        setattr(fifth, name_, wrapper)

    for t in TARGETS:
        wrap(t)

    t0 = time.perf_counter()
    n = 0
    for _tree, _polys in fifth.iter_unique_nets(v, faces, iter(trees), normals):
        n += 1
    total = time.perf_counter() - t0

    print(f"{name}: trees={len(trees)} nets={n}")
    print(f"  {'enumerate_spanning_trees':26s} {t_enum:7.3f}s")
    print(f"  {'pipeline (after enum)':26s} {total:7.3f}s")
    for k in TARGETS:
        print(f"    {k:24s} {acc[k]:7.3f}s  ({acc[k] / total * 100:5.1f}%)  calls={calls[k]}")
    other = total - sum(acc.values())
    print(f"    {'other (dedup/glue/keys)':24s} {other:7.3f}s  ({other / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()
