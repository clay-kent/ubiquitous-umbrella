"""baseline と変更後で、同じ tie の_net が同じ顔並びになるか比較する。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import api_server
import make_nets


def probe(label):
    client = TestClient(api_server.app)
    v, f = make_nets.prism(4)
    r = client.post(
        "/api/net/calculate",
        json={
            "vertices": [[float(x) for x in p] for p in v.tolist()],
            "faces": f,
            "top_n": 3,
            "sort_by": "area",
        },
    )
    j = r.json()
    print(f"--- {label} ---")
    for i, n in enumerate(j["nets"]):
        parents = [fa["parent_face_id"] for fa in n["faces"]]
        print(f"  net{i + 1}: parents={parents} area={n['metrics']['bounding_box_area']:.9f}")


if __name__ == "__main__":
    probe(sys.argv[1] if len(sys.argv) > 1 else "current")