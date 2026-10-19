"""列挙そのもの（enumerate_spanning_trees）の新旧を同一プロセス内で交互に測る。

パイプラインの A/B（probe_ab_pipeline）は木の列を**先に作ってから**渡すので、
列挙の変更は見えない。ここは列挙だけを比べる。

    uv run python tools/probe_ab_enum.py stairs 6
"""

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fifth
from bench_ladder import SOLIDS


def load_old_module() -> object:
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


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "stairs"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    old = load_old_module()
    v, f = SOLIDS[name]
    faces = fifth.orient_faces_outward(v, f)
    edges = fifth.build_adjacency_graph(faces)
    n_faces = len(faces)

    new_trees = list(fifth.enumerate_spanning_trees(edges, n_faces))
    old_trees = list(getattr(old, "enumerate_spanning_trees")(edges, n_faces))
    same = new_trees == old_trees
    print(f"{name}: trees={len(new_trees)} identical={same}")

    pairs: list[tuple[float, float]] = []
    for r in range(rounds):
        order = [("new", fifth), ("old", old)]
        if r % 2 == 1:
            order.reverse()
        dts: dict[str, float] = {}
        for label, mod in order:
            t0 = time.process_time()
            got = list(getattr(mod, "enumerate_spanning_trees")(edges, n_faces))
            dts[label] = time.process_time() - t0
            print(f"    {label}: {dts[label]:7.3f}s  trees={len(got)}")
        pairs.append((dts["new"], dts["old"]))
    print(f"  ratio old/new per round: "
          f"{' '.join(f'{o / n:.3f}' for n, o in pairs)}")
    ratios = sorted(o / n for n, o in pairs)
    print(f"  median={ratios[len(ratios) // 2]:.3f}x  "
          f"min={min(n for n, _o in pairs):.3f}s vs {min(o for _n, o in pairs):.3f}s")


if __name__ == "__main__":
    main()
