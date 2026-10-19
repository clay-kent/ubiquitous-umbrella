"""`enumerate_spanning_trees` の出力が HEAD と完全に一致するか（順序込み）。

列挙は「出力の順序」がそのままパイプラインの出力順に効くので、集合ではなく
**列**として比べる。

    uv run python tools/check_enum_bits.py            # bench_ladder の全立体
    uv run python tools/check_enum_bits.py cube box   # 名前を指定
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS
from probe_ab_pipeline import load_old_module


def main() -> None:
    names = sys.argv[1:] or list(SOLIDS)
    old = load_old_module()
    ok = True
    for name in names:
        v, f = SOLIDS[name]
        faces = fifth.orient_faces_outward(v, f)
        edges = fifth.build_adjacency_graph(faces)
        new = list(fifth.enumerate_spanning_trees(edges, len(faces)))
        ref = list(old.enumerate_spanning_trees(edges, len(faces)))
        same = new == ref
        ok = ok and same
        print(f"{name:9s} trees={len(new):7d} ref={len(ref):7d} {'OK' if same else 'NG'}")
    print("ALL OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
