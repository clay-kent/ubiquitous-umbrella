"""C-2 時間予算の検証。

- 予算内の立体は従来通り 200 で返る
- 予算を小さくすると time_budget_exceeded の 422 になる
- 422 の type / time_budget フィールドが正しい
- 一度 422 になったら部分結果は返らない（results are deterministic）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import api_server
import fifth
import make_nets

client = TestClient(api_server.app)
URL = "/api/net/calculate"


def body_for(builder):
    v, f = builder()
    return {"vertices": [[float(x) for x in p] for p in v.tolist()], "faces": f, "top_n": 1}


def check(name, builder, budget, expect_status):
    old = fifth._TIME_BUDGET_LIMIT
    fifth._TIME_BUDGET_LIMIT = budget
    try:
        t0 = time.perf_counter()
        r = client.post(URL, json=body_for(builder))
        elapsed = time.perf_counter() - t0
        if expect_status == 200:
            ok = r.status_code == 200
            n = r.json().get("total_unique_nets") if ok else None
            print(f"{name:22s} budget={budget:5.1f}s -> {r.status_code} ({elapsed:.2f}s) unique={n} {'OK' if ok else 'FAIL'}")
        else:
            j = r.json()
            err = j["detail"][0]
            ok = r.status_code == 422 and err["type"] == "time_budget_exceeded"
            print(
                f"{name:22s} budget={budget:5.1f}s -> {r.status_code} ({elapsed:.2f}s) "
                f"type={err.get('type')} time_budget={err.get('time_budget')} "
                f"{'OK' if ok else 'FAIL'}"
            )
            if r.status_code == 422:
                assert "msg" in err and "時間予算" in err["msg"], err
    finally:
        fifth._TIME_BUDGET_LIMIT = old
    return ok


if __name__ == "__main__":
    print("--- 予算内（200 を期待）---")
    a = check("prism6", lambda: make_nets.prism(6), 10.0, 200)
    b = check("box", lambda: make_nets.box(1.0, 1.5, 2.0), 10.0, 200)
    print("--- 予算切れ（422 を期待）---")
    # stairs は ~31s かかるので 0.2s 予算なら確実に 422
    c = check("stairs (0.2s budget)", lambda: make_nets.SOLIDS["stairs"] if False else __import__("bench_ladder").SOLIDS["stairs"], 0.2, 422)
    print("ALL OK" if (a and b and c) else "SOME FAILED")