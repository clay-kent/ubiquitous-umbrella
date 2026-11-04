"""cProfileで ホットスポットを特定する（複雑度の高い立体で実行）。"""

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fifth
from bench_ladder import SOLIDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solid", default="prism7")
    args = parser.parse_args()

    vertices, faces = SOLIDS[args.solid]
    n_faces = len(faces)
    edges = fifth.build_adjacency_graph(faces)
    normals = fifth.compute_normals(vertices, faces)

    def run():
        trees = fifth.enumerate_spanning_trees(edges, n_faces)
        nets = list(fifth.iter_unique_nets(vertices, faces, trees, normals))
        shared_of = fifth.build_shared_of(faces)
        scored = [
            fifth.net_metrics(
                polys, tree, vertices, faces, normals, shared_of
            ).bounding_box_area
            for tree, polys in nets
        ]
        scored.sort()
        return scored

    pr = cProfile.Profile()
    pr.enable()
    result = run()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    print(s2.getvalue())
    print(f"unique nets: {len(result)}")


if __name__ == "__main__":
    main()
