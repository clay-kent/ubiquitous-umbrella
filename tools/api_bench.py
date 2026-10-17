"""API の実測（TestClient 経由の1リクエスト往復）を取る。README の表用。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import make_nets
from fastapi.testclient import TestClient

import api_server

CASES = [
    ("5角柱", lambda: make_nets.prism(5)),
    ("6角柱", lambda: make_nets.prism(6)),
    ("7角柱", lambda: make_nets.prism(7)),
    ("8角柱", lambda: make_nets.prism(8)),
]


def main():
    client = TestClient(api_server.app)
    for name, builder in CASES:
        v, f = builder()
        body = {
            "vertices": [[float(x) for x in p] for p in v.tolist()],
            "faces": f,
            "top_n": 1,
            "sort_by": "area",
        }
        client.post("/api/net/calculate", json=body)  # ウォームアップ
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = client.post("/api/net/calculate", json=body)
            times.append(time.perf_counter() - t0)
        j = r.json()
        if r.status_code == 200:
            total = j["total_unique_nets"]
            best = j["nets"][0]["metrics"]["bounding_box_area"]
            print(
                f"{name:6s} 200 min={min(times):.2f}s total_unique={total:6d} "
                f"best_bbox={best:.4f}"
            )
        else:
            err = j["detail"][0]
            print(
                f"{name:6s} {r.status_code} min={min(times):.2f}s "
                f"type={err['type']} spanning_trees={err.get('spanning_trees')}"
            )


if __name__ == "__main__":
    main()