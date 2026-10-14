"""PR2 のリファクタで API 応答が変わっていないことの固定。

幾何を api_server.py から fifth.py へ移すPureな移動のはずだが、
``fold_axis_*`` と ``parent_face_id`` は親子関係の定義に依存している。
PR2 着手前の応答を golden として保存し、**正規化してから**突き合わせる。

⚠️ 生の JSON 文字列は比較しない。float の repr や dict の挿入順は
意味を変えずに変動するので、バイト一致は偽の失敗を生む。
"""

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

import api_server
import make_nets

client = TestClient(api_server.app, raise_server_exceptions=False)
URL = "/api/net/calculate"
GOLDEN = pathlib.Path(__file__).parent / "tests" / "api_golden_v2.json"


def normalise(obj):
    """dict を再帰的にソートし、float を9桁に丸めて比較可能にする。"""
    if isinstance(obj, dict):
        return {k: normalise(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    if isinstance(obj, (int, float)):
        return round(float(obj), 9)
    return obj


def _body(builder, sort_by):
    v, f = builder()
    return {
        "vertices": [[float(c) for c in p] for p in v.tolist()],
        "faces": f,
        "top_n": 3,
        "sort_by": sort_by,
    }


CASES = [
    ("cube", lambda: make_nets.box(1, 1, 1), "area"),
    ("box1x2x3", lambda: make_nets.box(1, 2, 3), "area"),
    ("tetra", make_nets.tetrahedron, "perimeter"),
    ("prism4", lambda: make_nets.prism(4), "area"),
    ("pyr4", lambda: make_nets.pyramid(4), "perimeter"),
]


@pytest.mark.parametrize(
    ("name", "builder", "sort_by"), CASES, ids=[c[0] for c in CASES]
)
def test_api_response_matches_pre_refactor_golden(name, builder, sort_by):
    """リファクタ前の応答と、正規化してから一致すること。"""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    r = client.post(URL, json=_body(builder, sort_by))
    assert r.status_code == 200, r.text[:400]
    assert normalise(r.json()) == golden[name]


def test_golden_covers_the_fields_the_move_could_break():
    """golden が本当に ``fold_axis_*`` / ``parent_face_id`` を含んでいるか。

    親を持つ面の axis が全部 None の golden だと、
    「お互い null だから一致している」だけで通ってしまう。歯止め。
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    checked = 0
    for name, payload in golden.items():
        assert payload["nets"], name
        for net in payload["nets"]:
            for face in net["faces"]:
                if face["parent_face_id"] is None:
                    continue
                assert face["fold_axis_3d"] is not None, name
                assert face["fold_axis_2d"] is not None, name
                assert face["dihedral_angle"] is not None, name
                checked += 1
    assert checked > 0