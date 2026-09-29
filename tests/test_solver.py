"""求解器与 API 的端到端测试（pytest）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    CHECK_TOL,
    INDISTINGUISHABLE_TOL,
    SolverNotConverged,
    ValidationError,
    analyze_pipe_outages,
    locate_single_leak,
    solve_network,
    validate_leak_payload,
    validate_payload,
    _network_with_extra_demand,
)

client = TestClient(app)


def base_loop():
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
            {"id": "C", "type": "demand", "demand": 1.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "B", "end": "C", "r": 1.0},
            {"id": "P4", "start": "C", "end": "S", "r": 1.0},
            {"id": "P5", "start": "S", "end": "B", "r": 2.5},
        ],
        "min_head": 10.0,
    }


def test_single_pipe_closed_form():
    """单管：q = 需水量，解析解 H_A = H_S - r·q|q|。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 2.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["source_supply"] == pytest.approx(2.0, abs=1e-12)
    assert res["heads"]["A"] == pytest.approx(26.0, abs=1e-9)
    assert res["flows"]["P1"] == pytest.approx(2.0, abs=1e-12)
    assert res["max_node_residual"] < CHECK_TOL
    assert res["max_pipe_residual"] < CHECK_TOL


def test_reversed_pipe_has_negative_flow():
    """反向录入时带符号流量为负，物理流向仍为 S→A。"""
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 2.0},
        ],
        "pipes": [{"id": "P1", "start": "A", "end": "S", "r": 1.0}],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["flows"]["P1"] < 0
    assert res["pipe_results"][0]["direction"] == "S → A"
    assert res["heads"]["A"] == pytest.approx(26.0, abs=1e-9)


def test_loop_network_balance_and_head_law():
    res = solve_network(validate_payload(base_loop()))
    assert res["converged"] is True
    assert res["source_supply"] == pytest.approx(3.0, abs=1e-9)
    assert abs(res["balance_residual"]) < CHECK_TOL

    net = validate_payload(base_loop())
    idx = {n["id"]: i for i, n in enumerate(net.nodes)}
    H, q, r = res["heads"], res["flows"], {p["id"]: p["r"] for p in net.pipes}
    # 独立复核每根管道 H起-H终 = r·q|q|
    for p in net.pipes:
        assert H[p["start"]] - H[p["end"]] == pytest.approx(
            r[p["id"]] * q[p["id"]] * abs(q[p["id"]]), abs=CHECK_TOL)
    # 独立复核每个用水节点净流入 = 需水量
    for k, node in enumerate(net.nodes):
        if node["type"] == "demand":
            inflow = 0.0
            for p in net.pipes:
                if p["start"] == node["id"]:
                    inflow -= q[p["id"]]
                elif p["end"] == node["id"]:
                    inflow += q[p["id"]]
            assert inflow == pytest.approx(node["demand"], abs=CHECK_TOL)


def test_below_min_head_flag():
    payload = base_loop()
    payload["min_head"] = 29.0  # 所有用水节点水头约 28.7
    res = solve_network(validate_payload(payload))
    assert set(res["below_min_head"]) == {"A", "B", "C"}
    assert all(n["below_min_head"] for n in res["node_results"] if n["id"] != "S")


def test_zero_demand_nodes_allowed():
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 0.0},
            {"id": "B", "type": "demand", "demand": 3.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 2.0},
        ],
        "min_head": 0.0,
    }
    res = solve_network(validate_payload(payload))
    assert res["source_supply"] == pytest.approx(3.0, abs=1e-9)
    # B 点水头 = 30 - 1·9 - 2·9 = 3
    assert res["heads"]["B"] == pytest.approx(3.0, abs=1e-8)


# ---------------------------- 校验类用例 ---------------------------- #
@pytest.mark.parametrize("mutate,expected", [
    (lambda p: p["nodes"].__setitem__(1, {"id": "S", "type": "source", "head": 1}),
     "重复"),
    (lambda p: p["nodes"].__setitem__(1, {"id": "X", "type": "tank", "head": 1}),
     "类型"),
    (lambda p: p["nodes"].__setitem__(0, {"id": "S", "type": "source", "head": float("nan")}),
     "水头"),
    (lambda p: p["nodes"].__setitem__(1, {"id": "A", "type": "demand", "demand": -1}),
     "需水量"),
    (lambda p: p["pipes"].append({"id": "PX", "start": "S", "end": "A", "r": 1}),
     "重复"),
    (lambda p: p["pipes"][0].__setitem__("r", 0), "阻力"),
    (lambda p: p["pipes"][0].__setitem__("r", -2), "阻力"),
    (lambda p: p["pipes"].append({"id": "P9", "start": "S", "end": "S", "r": 1}),
     "自环"),
    (lambda p: p["pipes"].append({"id": "P9", "start": "S", "end": "ZZ", "r": 1}),
     "不是已定义"),
    (lambda p: p["nodes"].__setitem__(0, {"id": "S", "type": "demand", "demand": 1}),
     "水源"),
])
def test_validation_errors(mutate, expected):
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    mutate(payload)
    with pytest.raises(ValidationError) as exc:
        validate_payload(payload)
    assert expected in str(exc.value)


def test_node_and_pipe_count_limits():
    # 11 个节点
    nodes = [{"id": "S", "type": "source", "head": 1.0}]
    nodes += [{"id": f"N{i}", "type": "demand", "demand": 1.0} for i in range(10)]
    with pytest.raises(ValidationError, match="节点数"):
        validate_payload({"nodes": nodes, "pipes": [], "min_head": 0})

    # 0 根管段
    with pytest.raises(ValidationError, match="管段数"):
        validate_payload({
            "nodes": [
                {"id": "S", "type": "source", "head": 1.0},
                {"id": "A", "type": "demand", "demand": 1.0},
            ],
            "pipes": [], "min_head": 0,
        })


def test_disconnected_graph_rejected():
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
        ],
        "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
        "min_head": 0.0,
    }
    with pytest.raises(ValidationError, match="连通"):
        validate_payload(payload)


def test_non_finite_values_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {
            "nodes": [
                {"id": "S", "type": "source", "head": bad},
                {"id": "A", "type": "demand", "demand": 1.0},
            ],
            "pipes": [{"id": "P1", "start": "S", "end": "A", "r": 1.0}],
            "min_head": 0.0,
        }
        with pytest.raises(ValidationError, match="有限"):
            validate_payload(payload)


# ---------------------------- API 用例 ---------------------------- #
def test_api_solve_ok():
    resp = client.post("/api/solve", json=base_loop())
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    r = data["result"]
    assert r["source_supply"] == pytest.approx(3.0, abs=1e-9)
    assert len(r["pipe_results"]) == 5


def test_api_validation_error_message():
    payload = base_loop()
    payload["pipes"][0]["r"] = -1
    resp = client.post("/api/solve", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["ok"] is False and "阻力" in data["error"]


def test_api_rejects_nan_token():
    body = '{"nodes":[{"id":"S","type":"source","head":NaN},' \
           '{"id":"A","type":"demand","demand":1}],' \
           '"pipes":[{"id":"P1","start":"S","end":"A","r":1}],"min_head":0}'
    resp = client.post("/api/solve", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "JSON" in resp.json()["error"]


def test_api_empty_body():
    resp = client.post("/api/solve", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_static_index_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "闭环供水管网" in resp.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"


# ---------------------------- 随机网络数值稳健性 ---------------------------- #
def test_random_connected_networks():
    rng = np.random.default_rng(0)
    n_ok = 0
    for trial in range(60):
        n = int(rng.integers(2, 11))
        edges = {tuple(sorted((i, int(rng.integers(0, i))))) for i in range(1, n)}
        target = int(rng.integers(len(edges), min(16, n * (n - 1) // 2) + 1))
        possible = list({(i, j) for i in range(n) for j in range(i + 1, n)} - edges)
        extra = min(target - len(edges), len(possible))
        perm = rng.permutation(len(possible))[:extra]
        edges.update(possible[k] for k in perm)
        src = int(rng.integers(0, n))
        nodes = [
            {"id": f"N{i}", "type": "source",
             "head": float(rng.uniform(20, 100))} if i == src
            else {"id": f"N{i}", "type": "demand",
                  "demand": float(rng.uniform(0, 5))}
            for i in range(n)
        ]
        pipes = [
            {"id": f"P{k}", "start": f"N{i}", "end": f"N{j}",
             "r": float(10 ** rng.uniform(-3, 3))}
            for k, (i, j) in enumerate(edges, 1)
        ]
        res = solve_network(validate_payload(
            {"nodes": nodes, "pipes": pipes, "min_head": -1e9}))
        assert res["max_node_residual"] < CHECK_TOL
        assert res["max_pipe_residual"] < CHECK_TOL
        assert math.isclose(res["source_supply"], res["total_demand"], abs_tol=CHECK_TOL)
        n_ok += 1
    assert n_ok == 60


# ---------------------------- 逐根停管分析 ---------------------------- #
def _tree_payload():
    """树状网：每根管段都是桥，移除必有失供。

        S --P1-- A(需 2) --P2-- B(需 3) --P3-- C(需 4)
    """
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 50.0},
            {"id": "A", "type": "demand", "demand": 2.0},
            {"id": "B", "type": "demand", "demand": 3.0},
            {"id": "C", "type": "demand", "demand": 4.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "B", "end": "C", "r": 1.0},
        ],
        "min_head": 0.0,
    }


def test_outage_tree_all_disconnected_sorted_by_unserved_demand():
    a = analyze_pipe_outages(validate_payload(_tree_payload()))
    stats = [(s["pipe_id"], s["status"]) for s in a["scenarios"]]
    assert stats == [
        ("P1", "disconnected"),  # 失供 2+3+4=9
        ("P2", "disconnected"),  # 失供 3+4=7
        ("P3", "disconnected"),  # 失供 4
    ]
    first = a["scenarios"][0]
    assert first["unserved_nodes"] == ["A", "B", "C"]
    assert first["unserved_demand"] == pytest.approx(9.0)
    assert "min_demand_head" not in first  # 断联情景不生成水头
    assert a["scenarios"][1]["unserved_nodes"] == ["B", "C"]
    assert a["scenarios"][1]["unserved_demand"] == pytest.approx(7.0)
    assert a["scenarios"][2]["unserved_nodes"] == ["C"]
    assert [s["rank"] for s in a["scenarios"]] == [1, 2, 3]
    assert a["counts"] == {"disconnected": 3, "connected": 0, "not_converged": 0}


def _loop_branch_payload():
    """环 + 两根枝：P4、P5 为桥，P1～P3 构成 S-A-B-S 环。

        S ──P1── A(需1) ──P2── B(需1) ──P4── D(需5)
         │                  │                    （挂枝）
         └──────P3──────────┘
        A ──P5── E(需2)（挂枝）
    """
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 40.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
            {"id": "D", "type": "demand", "demand": 5.0},
            {"id": "E", "type": "demand", "demand": 2.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "B", "end": "S", "r": 1.0},
            {"id": "P4", "start": "B", "end": "D", "r": 1.0},
            {"id": "P5", "start": "A", "end": "E", "r": 1.0},
        ],
        "min_head": 0.0,
    }


def test_outage_mixed_grouping_and_connected_fields():
    payload = _loop_branch_payload()
    a = analyze_pipe_outages(validate_payload(payload))
    statuses = [(s["pipe_id"], s["status"]) for s in a["scenarios"]]
    # 断联情景在前，按失供量降序：P4 失供 D(5) > P5 失供 E(2)
    assert statuses[:2] == [("P4", "disconnected"), ("P5", "disconnected")]
    # 随后是三个连通情景，按最低用水节点水头升序
    connected = [s for s in a["scenarios"] if s["status"] == "connected"]
    assert len(connected) == 3
    # 失去 S-B 直通管 P3 时最不利，其次失去 S-A 管 P1，最后失去环中段 P2
    assert [s["pipe_id"] for s in connected] == ["P3", "P1", "P2"]
    heads = [s["min_demand_head"] for s in connected]
    assert heads == sorted(heads)

    p4 = a["scenarios"][0]
    assert p4["unserved_nodes"] == ["D"]
    assert p4["unserved_demand"] == pytest.approx(5.0)

    # 连通情景字段完整，且与独立重解一致
    net = validate_payload(payload)
    for s in connected:
        idx = next(i for i, p in enumerate(net.pipes) if p["id"] == s["pipe_id"])
        sub = type(net)(nodes=net.nodes,
                        pipes=net.pipes[:idx] + net.pipes[idx + 1:],
                        min_head=net.min_head)
        res = solve_network(sub)
        demand_ids = [n["id"] for n in net.nodes if n["type"] == "demand"]
        assert s["min_demand_head"] == pytest.approx(
            min(res["heads"][nid] for nid in demand_ids), abs=1e-9)
        assert s["below_min_head"] == res["below_min_head"]
        assert s["solution"]["source_supply"] == pytest.approx(
            res["source_supply"], abs=1e-9)

    # 基线与各情景共用同一限值；baseline 中低水头节点集合随分析返回
    assert a["baseline"]["converged"] is True
    assert set(a["baseline_below_min_head"]) == set(a["baseline"]["below_min_head"])


def test_outage_new_below_min_head_nodes():
    """选择使基线合格、但移除 P1 后 B 点水头跌破限值的阈值。"""
    payload = _loop_branch_payload()
    net = validate_payload(payload)
    baseline = solve_network(net)
    # 找到一个移除后最低水头下降最多的环管，以其最低水头与基线最低水头之间取限值
    a0 = analyze_pipe_outages(net)
    worst = min((s for s in a0["scenarios"] if s["status"] == "connected"),
                key=lambda s: s["min_demand_head"])
    threshold = (worst["min_demand_head"] +
                 min(baseline["heads"][n["id"]]
                     for n in net.nodes if n["type"] == "demand")) / 2
    payload["min_head"] = threshold
    a = analyze_pipe_outages(validate_payload(payload))
    base_below = set(a["baseline_below_min_head"])
    assert base_below == set()  # 限值取在两者之间，基线应全部合格
    target = next(s for s in a["scenarios"] if s["pipe_id"] == worst["pipe_id"])
    assert target["new_below_min_head"], "最差情景应出现比基线新增的低水头节点"
    assert set(target["new_below_min_head"]).issubset(set(target["below_min_head"]))
    # 其余连通情景的“新增”集合 = 情景低水头集合 − 基线低水头集合
    for s in a["scenarios"]:
        if s["status"] == "connected":
            assert s["new_below_min_head"] == [
                nid for nid in s["below_min_head"] if nid not in base_below]


def test_outage_tie_keeps_pipe_entry_order():
    """同分情景保持管段录入顺序：base_loop 关于 S-B 对角线镜像对称，
    移除 P1（S-A）与 P4（C-S）最低水头相同；移除 P2（A-B）与 P3（B-C）亦相同。"""
    payload = base_loop()
    a = analyze_pipe_outages(validate_payload(payload))
    # 无断联（原图连通，单管移除后仍连通）
    assert a["counts"]["disconnected"] == 0
    ordered = a["scenarios"]
    by_id = {s["pipe_id"]: s for s in ordered}
    assert by_id["P1"]["min_demand_head"] == pytest.approx(
        by_id["P4"]["min_demand_head"], abs=1e-9)
    assert by_id["P2"]["min_demand_head"] == pytest.approx(
        by_id["P3"]["min_demand_head"], abs=1e-9)
    # 录入顺序相同的 P1 排在 P4 前、P2 排在 P3 前
    assert by_id["P1"]["rank"] < by_id["P4"]["rank"]
    assert by_id["P2"]["rank"] < by_id["P3"]["rank"]
    # 整体仍按最低水头非降序
    hs = [s["min_demand_head"] for s in ordered]
    assert hs == sorted(hs)
    assert [s["rank"] for s in ordered] == list(range(1, len(ordered) + 1))


def test_outage_single_nonconverged_scenario_marked_and_last(monkeypatch):
    """某一情景单独不收敛：标明并置于最后，其余情景照常给出。"""
    payload = _loop_branch_payload()
    real_solve = solve_network
    call = {"n": 0}

    def fake_solve(net):
        # analyze_pipe_outages 的第 2 次求解 = 移除第 0 根管（P1）的情景
        call["n"] += 1
        if call["n"] == 2:
            raise SolverNotConverged("模拟 P1 情景不收敛")
        return real_solve(net)

    monkeypatch.setattr("app.solver.solve_network", fake_solve)
    a = analyze_pipe_outages(validate_payload(payload))
    assert a["counts"] == {"disconnected": 2, "connected": 2, "not_converged": 1}
    last = a["scenarios"][-1]
    assert last["status"] == "not_converged"
    assert last["pipe_id"] == "P1"
    assert "模拟 P1 情景不收敛" in last["error"]
    # 断联在前、连通居中、不收敛最后
    order = [s["status"] for s in a["scenarios"]]
    assert order == ["disconnected", "disconnected", "connected", "connected",
                     "not_converged"]
    assert all("min_demand_head" in s for s in a["scenarios"]
               if s["status"] == "connected")


# ---------------------------- 停管分析 API ---------------------------- #
def test_api_analyze_outages_ok():
    resp = client.post("/api/analyze-outages", json=_tree_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    a = data["analysis"]
    assert [s["pipe_id"] for s in a["scenarios"]] == ["P1", "P2", "P3"]
    assert a["scenarios"][0]["unserved_demand"] == pytest.approx(9.0)
    assert a["baseline"]["converged"] is True


def test_api_analyze_outages_validation_error():
    payload = _tree_payload()
    payload["pipes"][0]["r"] = 0
    resp = client.post("/api/analyze-outages", json=payload)
    assert resp.status_code == 422
    assert "阻力" in resp.json()["error"]


def test_api_analyze_outages_empty_body():
    resp = client.post("/api/analyze-outages", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


# ---------------------------- 单点漏损定位 ---------------------------- #
def _leak_loop_payload():
    """5 节点、6 管环状管网（不对称），用于漏损定位。"""
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 40.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
            {"id": "C", "type": "demand", "demand": 1.0},
            {"id": "D", "type": "demand", "demand": 1.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "B", "end": "C", "r": 1.0},
            {"id": "P4", "start": "C", "end": "S", "r": 1.0},
            {"id": "P5", "start": "S", "end": "B", "r": 2.5},
            {"id": "P6", "start": "A", "end": "D", "r": 1.5},
        ],
        "min_head": 0.0,
    }


def _leak_input(node_id, leak, measure_ids, upper=3.0):
    """以 node_id 增加 leak 的真值解造实测水头，组装漏损分析输入。"""
    payload = _leak_loop_payload()
    net = validate_payload(payload)
    truth = solve_network(_network_with_extra_demand(net, node_id, leak))
    payload["measurements"] = [
        {"node": mid, "head": truth["heads"][mid]} for mid in measure_ids]
    payload["leak_upper_bound"] = upper
    return payload


def test_leak_locates_known_node_and_value():
    payload = _leak_input("B", 0.7, ["A", "C", "D"])
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)

    assert res["verdict"] == "located"
    assert res["leak_detected"] is True
    assert res["leak_node"] == "B"
    assert res["leak_value"] == pytest.approx(0.7, abs=1e-7)

    top = res["candidates"][0]
    assert top["node_id"] == "B" and top["rank"] == 1
    assert top["rmse"] < 1e-6
    assert top["at_bound"] is False
    # 预测水头与各测点实测一致，误差符号 = 预测 − 实测
    assert [p["node"] for p in top["predicted_heads"]] == ["A", "C", "D"]
    for p in top["predicted_heads"]:
        assert p["predicted_head"] == pytest.approx(p["measured_head"], abs=1e-6)
        assert abs(p["error"]) < 1e-6
    # 基线（无漏损）误差明显且大于最优候选
    assert res["baseline_rmse"] > 0.5
    assert res["baseline_rmse"] > top["rmse"]
    assert len(res["baseline_predicted_heads"]) == 3
    # 候选覆盖全部用水节点，按 RMSE 升序
    assert [c["node_id"] for c in res["candidates"]] == ["B", "A", "C", "D"]
    rmses = [c["rmse"] for c in res["candidates"]]
    assert rmses == sorted(rmses)
    assert res["counts"] == {"fitted": 4, "fit_failed": 0}


def test_leak_evaluates_upper_endpoint_and_marks_bound():
    """真值漏量取在上限处时，最优漏量等于上限并标记 at_bound。"""
    # 4 个测点保证最优（恰在上限）与次优差距远超 1e-4，不被标为无法区分
    payload = _leak_input("C", 1.5, ["A", "B", "C", "D"], upper=1.5)
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)
    assert res["verdict"] == "located"
    assert res["leak_node"] == "C"
    assert res["leak_value"] == pytest.approx(1.5, abs=1e-8)
    top = res["candidates"][0]
    assert top["at_bound"] is True
    assert top["rmse"] < 1e-8
    # 次优与最优差距应超过无法区分阈值，确认不是“蹭”到边界的假最优
    assert res["candidates"][1]["rmse"] - top["rmse"] > INDISTINGUISHABLE_TOL


def test_leak_baseline_no_leak_means_no_detection():
    """实测与无漏损基线一致：最优漏量为 0，不判定存在漏损。"""
    payload = _leak_loop_payload()
    net = validate_payload(payload)
    base = solve_network(net)
    payload["measurements"] = [
        {"node": mid, "head": base["heads"][mid]} for mid in ("A", "C")]
    payload["leak_upper_bound"] = 2.0
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)

    assert res["verdict"] == "no_leak"
    assert res["leak_detected"] is False
    assert res["leak_node"] is None and res["leak_value"] == 0.0
    assert res["baseline_rmse"] == pytest.approx(0.0, abs=1e-12)
    # 每个候选的最优漏量都为 0（误差同为 0，同分按录入顺序 A 居首）
    assert res["candidates"][0]["node_id"] == "A"
    assert all(c["leak"] == pytest.approx(0.0, abs=1e-9) for c in res["candidates"])
    # 同分时最优候选自身也在“无法区分”集合里，但因漏量为 0 仍判无漏损
    assert set(res["indistinguishable_nodes"]) == {"A", "B", "C", "D"}


def _symmetric_leak_payload(leak_node="A", leak=0.6):
    """对称树：M1—M2 为对称轴，A/A2 关于 M1 对称、B/B2 关于 M2 对称。

    只测对称轴上的 M1、M2 时，同压子网上的候选（M1、A、A2）无法区分。
    """
    payload = {
        "nodes": [
            {"id": "S", "type": "source", "head": 50.0},
            {"id": "M1", "type": "demand", "demand": 1.0},
            {"id": "M2", "type": "demand", "demand": 1.0},
            {"id": "A", "type": "demand", "demand": 0.5},
            {"id": "A2", "type": "demand", "demand": 0.5},
            {"id": "B", "type": "demand", "demand": 0.5},
            {"id": "B2", "type": "demand", "demand": 0.5},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "M1", "r": 0.5},
            {"id": "P2", "start": "M1", "end": "M2", "r": 0.5},
            {"id": "P3", "start": "M1", "end": "A", "r": 1.0},
            {"id": "P4", "start": "M1", "end": "A2", "r": 1.0},
            {"id": "P5", "start": "M2", "end": "B", "r": 1.0},
            {"id": "P6", "start": "M2", "end": "B2", "r": 1.0},
        ],
        "min_head": 0.0,
    }
    net = validate_payload(payload)
    truth = solve_network(_network_with_extra_demand(net, leak_node, leak))
    payload["measurements"] = [
        {"node": "M1", "head": truth["heads"]["M1"]},
        {"node": "M2", "head": truth["heads"]["M2"]},
    ]
    payload["leak_upper_bound"] = 2.0
    return payload


def test_leak_indistinguishable_symmetric_candidates():
    payload = _symmetric_leak_payload("A", 0.6)
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)

    assert res["verdict"] == "indistinguishable"
    assert res["leak_detected"] is False
    assert res["leak_node"] is None
    indist = res["indistinguishable_nodes"]
    assert set(indist) == {"M1", "A", "A2"}
    # 同分按节点录入顺序：M1(序1)、A(序3)、A2(序4)
    assert indist == ["M1", "A", "A2"]
    best = res["best_rmse"]
    for c in res["candidates"]:
        if c["node_id"] in set(indist):
            assert c["indistinguishable"] is True
            assert abs(c["rmse"] - best) <= INDISTINGUISHABLE_TOL
        else:
            assert c["indistinguishable"] is False
            assert c["rmse"] - best > INDISTINGUISHABLE_TOL


def test_leak_indistinguishable_does_not_fire_when_gap_large():
    """非对称管网定位 B 时，最优与次优差距远超 1e-4，不标无法区分。"""
    payload = _leak_input("B", 0.7, ["A", "C", "D"])
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)
    assert res["indistinguishable_nodes"] == ["B"]
    assert res["verdict"] == "located"


# ---------------------------- 漏损定位输入校验 ---------------------------- #
@pytest.mark.parametrize("mutate,expected", [
    (lambda p: p.update(measurements=[{"node": "A", "head": 30.0}]), "2～8"),
    (lambda p: p.update(measurements=[{"node": "A", "head": 30.0}] * 9), "2～8"),
    (lambda p: p.update(measurements="nope"), "measurements"),
    (lambda p: p.update(measurements=[{"node": "S", "head": 40.0},
                                      {"node": "A", "head": 30.0}]), "用水节点"),
    (lambda p: p.update(measurements=[{"node": "A", "head": 30.0},
                                      {"node": "ZZ", "head": 30.0}]), "用水节点"),
    (lambda p: p.update(measurements=[{"node": "A", "head": 30.0},
                                      {"node": "A", "head": 29.0}]), "互异"),
    (lambda p: p.update(measurements=[{"node": "A", "head": float("nan")},
                                      {"node": "C", "head": 29.0}]), "有限"),
    (lambda p: p.update(measurements=[{"node": "A"},
                                      {"node": "C", "head": 29.0}]), "实测水头"),
    (lambda p: p.update(leak_upper_bound=0.0,
                        measurements=[{"node": "A", "head": 30.0},
                                      {"node": "C", "head": 29.0}]), "正数"),
    (lambda p: p.update(leak_upper_bound=-2.0,
                        measurements=[{"node": "A", "head": 30.0},
                                      {"node": "C", "head": 29.0}]), "正数"),
    (lambda p: p.update(leak_upper_bound=float("inf"),
                        measurements=[{"node": "A", "head": 30.0},
                                      {"node": "C", "head": 29.0}]), "有限"),
])
def test_leak_validation_errors(mutate, expected):
    payload = _leak_loop_payload()
    payload["measurements"] = [
        {"node": "A", "head": 30.0}, {"node": "C", "head": 29.0}]
    payload["leak_upper_bound"] = 3.0
    mutate(payload)
    with pytest.raises(ValidationError) as exc:
        validate_leak_payload(payload)
    assert expected in str(exc.value)


def test_leak_missing_measurements_key():
    payload = _leak_loop_payload()
    payload["leak_upper_bound"] = 3.0
    with pytest.raises(ValidationError, match="measurements"):
        validate_leak_payload(payload)


def test_leak_network_validation_still_applies():
    """管网本身非法（这里阻力为 0）时沿用既有校验分支。"""
    payload = _leak_loop_payload()
    payload["pipes"][0]["r"] = 0
    payload["measurements"] = [
        {"node": "A", "head": 30.0}, {"node": "C", "head": 29.0}]
    payload["leak_upper_bound"] = 3.0
    with pytest.raises(ValidationError, match="阻力"):
        validate_leak_payload(payload)


def test_leak_baseline_not_converged_propagates(monkeypatch):
    """基线不收敛时以 SolverNotConverged 说明原因（由 API 转为 422）。"""
    monkeypatch.setattr(
        "app.solver.solve_network",
        lambda net: (_ for _ in ()).throw(
            SolverNotConverged("模拟基线不收敛")))
    payload = _leak_loop_payload()
    with pytest.raises(SolverNotConverged, match="基线"):
        net, meas, upper = validate_leak_payload({
            **payload,
            "measurements": [{"node": "A", "head": 30.0},
                             {"node": "C", "head": 29.0}],
            "leak_upper_bound": 3.0,
        })
        locate_single_leak(net, meas, upper)


def test_leak_single_candidate_fit_failure_marked_and_last(monkeypatch):
    """单个候选在所有漏量上都不收敛：标记 fit_failed 排末尾，不影响其他候选。"""
    real_solve = solve_network
    baseline_nodes: list[dict] | None = None

    def fake_solve(net):
        nonlocal baseline_nodes
        if baseline_nodes is None:
            baseline_nodes = net.nodes  # 第一次调用即无漏损基线
            return real_solve(net)
        # 候选网络中恰好一个节点字典是复制出来的；若复制的是 D 节点则令其失败
        for i, node in enumerate(net.nodes):
            if node is not baseline_nodes[i] and node["id"] == "D":
                raise SolverNotConverged("模拟 D 候选不收敛")
        return real_solve(net)

    monkeypatch.setattr("app.solver.solve_network", fake_solve)

    payload = _leak_input("B", 0.7, ["A", "C"])
    net, meas, upper = validate_leak_payload(payload)
    res = locate_single_leak(net, meas, upper)

    last = res["candidates"][-1]
    assert last["node_id"] == "D"
    assert last["status"] == "fit_failed"
    assert last["rank"] == 4
    assert "模拟 D 候选不收敛" in last["error"]
    assert "rmse" not in last
    # 其余候选与结论不受影响
    assert res["counts"] == {"fitted": 3, "fit_failed": 1}
    assert res["leak_node"] == "B"
    assert [c["node_id"] for c in res["candidates"][:3]] == ["B", "A", "C"]


# ---------------------------- 漏损定位 API ---------------------------- #
def test_api_locate_leak_ok():
    payload = _leak_input("B", 0.7, ["A", "C", "D"])
    resp = client.post("/api/locate-leak", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    a = data["analysis"]
    assert a["verdict"] == "located"
    assert a["leak_node"] == "B"
    assert a["leak_value"] == pytest.approx(0.7, abs=1e-7)
    assert a["baseline"]["converged"] is True
    assert a["candidates"][0]["predicted_heads"][0]["node"] == "A"


def test_api_locate_leak_validation_error():
    payload = _leak_input("B", 0.7, ["A", "C", "D"])
    payload["measurements"] = payload["measurements"][:1]  # 只剩 1 个测点
    resp = client.post("/api/locate-leak", json=payload)
    assert resp.status_code == 422
    assert "2～8" in resp.json()["error"]


def test_api_locate_leak_duplicate_measurement():
    payload = _leak_loop_payload()
    payload["measurements"] = [
        {"node": "A", "head": 30.0}, {"node": "A", "head": 29.0}]
    payload["leak_upper_bound"] = 3.0
    resp = client.post("/api/locate-leak", json=payload)
    assert resp.status_code == 422
    assert "互异" in resp.json()["error"]


def test_api_locate_leak_empty_body():
    resp = client.post("/api/locate-leak", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
