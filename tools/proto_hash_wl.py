"""uint64 ハッシュ合成（mod 2^64 で折り返す）によるバッチWLのプロトタイプ。

Python の hash() はタプルを多項式として mod 2^64 で畳み込む。
numpy の uint64 演算も同じ折り返しをするので、同じ数学的性质を
ベクトルで再現できる（係数は互いに素な定数）。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fifth
from bench_ladder import SOLIDS

A1, B1, C1, D1 = (np.uint64(x) for x in (0x9E3779B97F4A7C1, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5))
A2, B2, C2, D2 = (np.uint64(x) for x in (0xD6E8FEB86659FD93, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3))
M1 = np.uint64(0xFFFFFFFFFFFFFFFF)


def main():
    v, f = SOLIDS["prism7"]
    faces = fifth.orient_faces_outward(v, f)
    trees = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(faces), len(faces)))
    ctx = fifth.RibbonContext(v, faces)
    n = ctx.n_nodes
    print(f"prism7: trees={len(trees)} half_edges={n}")

    pos = np.asarray([[ctx.adj_pos_of[(u, w)] for u, w in t] for t in trees], dtype=np.int64)
    t0 = time.perf_counter()
    partner = np.full((len(trees), n), -1, dtype=np.int64)
    rows = np.arange(len(trees))[:, None]
    nu, nv = ctx.adj_nu[pos], ctx.adj_nv[pos]
    partner[rows, nu] = nv
    partner[rows, nv] = nu
    flip = np.zeros((len(trees), n), dtype=np.int64)
    fl = ctx.adj_fl[pos]
    flip[rows, nu] = fl
    flip[rows, nv] = fl
    glued = partner >= 0
    por = np.where(glued, partner, np.arange(n)[None, :])
    print(f"glue: {time.perf_counter()-t0:.3f}s")

    uniq = np.unique(ctx.np_edge_len)
    eid = np.searchsorted(uniq, ctx.np_edge_len).astype(np.uint64)
    t0 = time.perf_counter()
    h1 = eid[None, :] * A1 + glued.astype(np.uint64) * B1 + flip.astype(np.uint64) * D1
    h2 = eid[None, :] * A2 + glued.astype(np.uint64) * B2 + flip.astype(np.uint64) * D2
    f64 = flip.astype(np.uint64)

    counts = np.empty(len(trees), dtype=np.int64)
    srt = np.sort(h1, axis=1)
    counts = 1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)

    rounds = 0
    t0 = time.perf_counter()
    unstable = np.ones(len(trees), dtype=bool)
    while unstable.any():
        rounds += 1
        hn1 = h1[:, ctx.np_nxt]
        hn2 = h2[:, ctx.np_nxt]
        hp1 = np.take_along_axis(h1, por, axis=1)
        hp2 = np.take_along_axis(h2, por, axis=1)
        # 接着なしは 0 に倒す（f64 の 0 が番兵）
        glued_u = glued.astype(np.uint64)
        n1 = (h1 * A1 + hn1 * B1 + hp1 * C1 + glued_u * f64 + (1 - glued_u) * D1)
        n2 = (h2 * A2 + hn2 * B2 + hp2 * C2 + glued_u * f64 + (1 - glued_u) * D2)
        srt = np.sort(n1, axis=1)
        ncnt = 1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)
        done = unstable & (ncnt == counts)
        h1[done] = n1[done]
        h2[done] = n2[done]
        unstable &= ~done
        h1[unstable] = n1[unstable]
        h2[unstable] = n2[unstable]
        counts = np.where(done, counts, ncnt)
    t_hash = time.perf_counter() - t0
    print(f"hash WL: {t_hash:.2f}s ({t_hash/len(trees)*1000:.4f} ms/tree), rounds={rounds}")

    # 同一性の検証: cube で 11 になるか
    import make_nets
    v2, f2 = make_nets.box(1.0, 1.5, 2.0)
    f2o = fifth.orient_faces_outward(v2, f2)
    ctx2 = fifth.RibbonContext(v2, f2o)
    t2 = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(f2o), len(f2o)))
    pos2 = np.asarray([[ctx2.adj_pos_of[(u, w)] for u, w in t] for t in t2], dtype=np.int64)

    def keys_for(context, positions):
        nn = context.n_nodes
        pt = np.full((len(positions), nn), -1, dtype=np.int64)
        rws = np.arange(len(positions))[:, None]
        nu2, nv2 = context.adj_nu[positions], context.adj_nv[positions]
        pt[rws, nu2] = nv2
        pt[rws, nv2] = nu2
        flp = np.zeros((len(positions), nn), dtype=np.int64)
        fl2 = context.adj_fl[positions]
        flp[rws, nu2] = fl2
        flp[rws, nv2] = fl2
        gl = pt >= 0
        pr = np.where(gl, pt, np.arange(nn)[None, :])
        uq = np.unique(context.np_edge_len)
        ee = np.searchsorted(uq, context.np_edge_len).astype(np.uint64)
        hh1 = ee[None, :] * A1 + gl.astype(np.uint64) * B1 + flp.astype(np.uint64) * D1
        hh2 = ee[None, :] * A2 + gl.astype(np.uint64) * B2 + flp.astype(np.uint64) * D2
        ff = flp.astype(np.uint64)
        gg = gl.astype(np.uint64)
        ss = np.sort(hh1, axis=1)
        cc = 1 + (ss[:, 1:] != ss[:, :-1]).sum(axis=1)
        uns = np.ones(len(positions), dtype=bool)
        while uns.any():
            xn1 = hh1[:, context.np_nxt]
            xn2 = hh2[:, context.np_nxt]
            xp1 = np.take_along_axis(hh1, pr, axis=1)
            xp2 = np.take_along_axis(hh2, pr, axis=1)
            m1 = hh1 * A1 + xn1 * B1 + xp1 * C1 + gg * ff + (1 - gg) * D1
            m2 = hh2 * A2 + xn2 * B2 + xp2 * C2 + gg * ff + (1 - gg) * D2
            s2 = np.sort(m1, axis=1)
            c2 = 1 + (s2[:, 1:] != s2[:, :-1]).sum(axis=1)
            d = uns & (c2 == cc)
            hh1[d] = m1[d]
            hh2[d] = m2[d]
            uns &= ~d
            hh1[uns] = m1[uns]
            hh2[uns] = m2[uns]
            cc = np.where(d, cc, c2)
        out = []
        for r in range(len(positions)):
            pair = sorted(zip(hh1[r].tolist(), hh2[r].tolist()))
            out.append(tuple(pair))
        return out, cc

    mirror_ctx2 = fifth.RibbonContext(v2, [list(reversed(x)) for x in f2o])
    k1, _ = keys_for(ctx2, pos2)
    k2, _ = keys_for(mirror_ctx2, pos2)
    kk = {min(a, b) for a, b in zip(k1, k2)}
    print(f"box(1,1.5,2) keys via hash batch: {len(kk)} (expect 54)")

    vc, fc = make_nets.box(1.0, 1.0, 1.0)
    fco = fifth.orient_faces_outward(vc, fc)
    c3 = fifth.RibbonContext(vc, fco)
    t3 = list(fifth.enumerate_spanning_trees(fifth.build_adjacency_graph(fco), 6))
    p3 = np.asarray([[c3.adj_pos_of[(u, w)] for u, w in t] for t in t3], dtype=np.int64)
    m3 = fifth.RibbonContext(vc, [list(reversed(x)) for x in fco])
    a3, _ = keys_for(c3, p3)
    b3, _ = keys_for(m3, p3)
    print(f"cube keys via hash batch: {len({min(x, y) for x, y in zip(a3, b3)})} (expect 11)")


if __name__ == "__main__":
    main()