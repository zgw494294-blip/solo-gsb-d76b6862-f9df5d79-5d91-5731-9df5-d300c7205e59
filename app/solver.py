"""
闭环供水管网平衡台 —— 稳态水力求解器。

数学模型（管段 p 按录入方向 i→j，流量 q_p 为带符号标量）：

    压降方程  H_i - H_j = r_p · q_p · |q_p|
    节点守恒  净流入 = 需水量 d_k            （用水节点 k）

水源节点给定固定水头 H_s，其余节点给定非负需水量 d_k；
未知量为各用水节点水头 H_k 与各管段带符号流量 q_p，方程数与未知量相等，
构成可由 Newton 法（SciPy）求解的平方非线性方程组。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize

# 对外校验容差：节点守恒与每段压降的残差必须不超过该值
CHECK_TOL = 1.0e-6
# 求解器内部目标残差（比交付校验更严，留出余量）
SOLVER_TARGET_TOL = 1.0e-9

MIN_NODES, MAX_NODES = 2, 10
MIN_PIPES, MAX_PIPES = 1, 16


class ValidationError(ValueError):
    """提交的管网描述不合法。"""


class SolverNotConverged(RuntimeError):
    """非线性方程组未能收敛到要求容差。"""


@dataclass(frozen=True)
class Network:
    nodes: list[dict[str, Any]]  # [{"id","type","head"|"demand"}]，保持录入顺序
    pipes: list[dict[str, Any]]  # [{"id","start","end","r"}]
    min_head: float


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _as_finite(value: Any, label: str, *, nonneg: bool = False,
               positive: bool = False) -> float:
    """把外部输入转成有限 float，按要求检查符号；非法时抛中文 ValidationError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label}必须是有限数值。")
    f = float(value)
    if not math.isfinite(f):
        raise ValidationError(f"{label}必须是有限数值（不允许 NaN 或无穷大）。")
    if positive and f <= 0.0:
        raise ValidationError(f"{label}必须为正数，当前为 {f:g}。")
    if nonneg and f < 0.0:
        raise ValidationError(f"{label}不能为负值，当前为 {f:g}。")
    return f


def validate_payload(payload: Any) -> Network:
    """完整校验 API 输入，返回规范化 Network；任何非法情形抛 ValidationError。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是包含 nodes、pipes 的对象。")

    raw_nodes = payload.get("nodes")
    raw_pipes = payload.get("pipes")
    if not isinstance(raw_nodes, list):
        raise ValidationError("nodes 必须是节点对象数组。")
    if not isinstance(raw_pipes, list):
        raise ValidationError("pipes 必须是管段对象数组。")

    if not (MIN_NODES <= len(raw_nodes) <= MAX_NODES):
        raise ValidationError(
            f"节点数必须在 {MIN_NODES}～{MAX_NODES} 之间，当前为 {len(raw_nodes)} 个。")
    if not (MIN_PIPES <= len(raw_pipes) <= MAX_PIPES):
        raise ValidationError(
            f"管段数必须在 {MIN_PIPES}～{MAX_PIPES} 之间，当前为 {len(raw_pipes)} 根。")

    # ---- 节点 -------------------------------------------------------------
    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_count = 0
    for k, raw in enumerate(raw_nodes, start=1):
        if not isinstance(raw, dict):
            raise ValidationError(f"第 {k} 个节点必须是对象。")
        nid = raw.get("id")
        if not isinstance(nid, str) or not nid.strip():
            raise ValidationError(f"第 {k} 个节点缺少有效的节点编号（非空字符串）。")
        nid = nid.strip()
        if nid in seen_ids:
            raise ValidationError(f"节点编号“{nid}”重复，节点编号必须唯一。")
        seen_ids.add(nid)

        ntype = raw.get("type")
        if ntype not in ("source", "demand"):
            raise ValidationError(
                f"节点“{nid}”的类型必须是 source（水源）或 demand（用水节点）。")

        if ntype == "source":
            head = _as_finite(raw.get("head"), f"水源节点“{nid}”的固定水头")
            nodes.append({"id": nid, "type": "source", "head": head})
            source_count += 1
        else:
            demand = _as_finite(
                raw.get("demand"), f"用水节点“{nid}”的需水量", nonneg=True)
            nodes.append({"id": nid, "type": "demand", "demand": demand})

    if source_count == 0:
        raise ValidationError("必须恰好指定 1 个水源节点（固定水头），当前为 0 个。")
    if source_count > 1:
        raise ValidationError(
            f"必须恰好指定 1 个水源节点，当前存在 {source_count} 个。")

    min_head = _as_finite(payload.get("min_head"), "最低水头限值")

    # ---- 管段 -------------------------------------------------------------
    pipes: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    seen_pipe_ids: set[str] = set()
    for k, raw in enumerate(raw_pipes, start=1):
        if not isinstance(raw, dict):
            raise ValidationError(f"第 {k} 根管段必须是对象。")
        pid = raw.get("id", f"P{k}")
        if not isinstance(pid, str) or not pid.strip():
            raise ValidationError(f"第 {k} 根管段的编号必须是非空字符串。")
        pid = pid.strip()
        if pid in seen_pipe_ids:
            raise ValidationError(f"管段编号“{pid}”重复，管段编号必须唯一。")
        seen_pipe_ids.add(pid)

        start = raw.get("start")
        end = raw.get("end")
        if not isinstance(start, str) or not start.strip():
            raise ValidationError(f"管段“{pid}”缺少起点编号。")
        start = start.strip()
        if not isinstance(end, str) or not end.strip():
            raise ValidationError(f"管段“{pid}”缺少终点编号。")
        end = end.strip()
        if start not in seen_ids:
            raise ValidationError(f"管段“{pid}”的起点“{start}”不是已定义的节点。")
        if end not in seen_ids:
            raise ValidationError(f"管段“{pid}”的终点“{end}”不是已定义的节点。")
        if start == end:
            raise ValidationError(
                f"管段“{pid}”的起点与终点都是“{start}”，不允许自环管段。")

        edge_key = tuple(sorted((start, end)))
        if edge_key in seen_edges:
            raise ValidationError(
                f"管段“{pid}”的端点对 {start}–{end} 与已有管段重复，"
                f"同一对节点之间只允许一根管段。")
        seen_edges.add(edge_key)

        resistance = _as_finite(raw.get("r"), f"管段“{pid}”的阻力 r", positive=True)
        pipes.append({"id": pid, "start": start, "end": end, "r": resistance})

    # ---- 无向连通性（并查集） ---------------------------------------------
    parent = {n["id"]: n["id"] for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pipe in pipes:
        a, b = find(pipe["start"]), find(pipe["end"])
        if a != b:
            parent[a] = b

    root = find(nodes[0]["id"])
    unreachable = [n["id"] for n in nodes if find(n["id"]) != root]
    if unreachable:
        raise ValidationError(
            "管网必须是无向连通图，以下节点无法经由管段与其他节点连通："
            + "、".join(unreachable) + "。")

    return Network(nodes=nodes, pipes=pipes, min_head=min_head)


# --------------------------------------------------------------------------- #
# 稳态求解
# --------------------------------------------------------------------------- #
def _bfs_tree_initial(net: Network, index: dict[str, int], source: int,
                      demands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """以水源为根做 BFS 生成树：用子树需水量摊派初始流量，沿程估算初始水头。

    返回 (全节点初始水头, 全部管段初始带符号流量)。
    """
    n = len(net.nodes)
    m = len(net.pipes)
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
    for p, pipe in enumerate(net.pipes):
        u, v = index[pipe["start"]], index[pipe["end"]]
        adj[u].append((v, p, 1.0))   # 录入方向 u->v 时 A[u,p]=+1
        adj[v].append((u, p, -1.0))  # 反向遍历时 q 系数为 -1

    par = [-1] * n
    pedge = [-1] * n
    psign = [0.0] * n
    order: list[int] = []
    seen = [False] * n
    seen[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, p, s in adj[u]:
            if not seen[v]:
                seen[v] = True
                par[v], pedge[v], psign[v] = u, p, s
                queue.append(v)

    q0 = np.zeros(m)
    subtree = demands.copy()
    with np.errstate(all="ignore"):
        for v in reversed(order):
            if v == source:
                continue
            # 水由父节点流向子节点；管段录入方向与父->子一致 (s=+1) 时 q>0
            q0[pedge[v]] += -psign[v] * subtree[v]
            subtree[par[v]] += subtree[v]
        # 循环结束后 subtree[v] 即边 par[v]→v 承担的子树总需水量

        h0 = np.zeros(n)
        h0[source] = net.nodes[source]["head"]
        for v in order:
            if v == source:
                continue
            r_e = net.pipes[pedge[v]]["r"]
            h0[v] = h0[par[v]] - r_e * subtree[v] * abs(subtree[v])

    return h0, q0


def _newton_backtrack(fun, jac, x0: np.ndarray, *, max_iter: int = 300,
                      tol: float = SOLVER_TARGET_TOL):
    """带 Armijo 回退的全量 Newton 法，作为 SciPy 求解器之外的保险策略。"""
    x = x0.astype(float).copy()
    f = fun(x)
    norm = float(np.max(np.abs(f))) if np.all(np.isfinite(f)) else math.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        if norm <= tol:
            return x, iterations, norm
        try:
            delta = np.linalg.solve(jac(x), -f)
        except np.linalg.LinAlgError:
            return x, iterations, norm
        if not np.all(np.isfinite(delta)):
            return x, iterations, norm
        step = 1.0
        accepted = False
        for _ in range(60):
            xn = x + step * delta
            fn = fun(xn)
            if np.all(np.isfinite(fn)):
                nn = float(np.max(np.abs(fn)))
                if nn < (1.0 - 1.0e-4 * step) * norm:
                    x, f, norm = xn, fn, nn
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return x, iterations, norm
    return x, iterations, norm


def solve_network(net: Network) -> dict[str, Any]:
    """求稳态解并按 1e-6 容差复核；不收敛时抛 SolverNotConverged。"""
    n = len(net.nodes)
    m = len(net.pipes)
    ids = [node["id"] for node in net.nodes]
    index = {nid: i for i, nid in enumerate(ids)}
    source = next(i for i, node in enumerate(net.nodes) if node["type"] == "source")
    demand_positions = [i for i in range(n) if i != source]
    n_dem = n - 1

    fixed_head = np.zeros(n)
    demands = np.zeros(n)
    for i, node in enumerate(net.nodes):
        if node["type"] == "source":
            fixed_head[i] = node["head"]
        else:
            demands[i] = node["demand"]
    h_source = fixed_head[source]
    d = demands[demand_positions]

    # 节点-管段关联矩阵 A（按录入方向：起点 +1，终点 -1），故 A·q 为净流出
    A = np.zeros((n, m))
    r = np.zeros(m)
    for p, pipe in enumerate(net.pipes):
        A[index[pipe["start"]], p] = 1.0
        A[index[pipe["end"]], p] = -1.0
        r[p] = pipe["r"]
    Ad = A[demand_positions, :]

    def residual(x: np.ndarray) -> np.ndarray:
        h_dem, q = x[:n_dem], x[n_dem:]
        h = fixed_head.copy()
        h[demand_positions] = h_dem
        f = np.empty(n_dem + m)
        # 用水节点：净流出 = -需水量，即净流入 = 需水量
        with np.errstate(all="ignore"):
            f[:n_dem] = Ad @ q + d
            f[n_dem:] = A.T @ h - r * q * np.abs(q)  # H起-H终 = r·q·|q|
        return f

    def jacobian(x: np.ndarray) -> np.ndarray:
        q = x[n_dem:]
        J = np.zeros((n_dem + m, n_dem + m))
        J[:n_dem, n_dem:] = Ad
        J[n_dem:, :n_dem] = Ad.T
        with np.errstate(all="ignore"):
            np.fill_diagonal(J[n_dem:, n_dem:], -2.0 * r * np.abs(q))
        return J

    # ---- 初始点：BFS 摊派 + 若干平坦兜底 ---------------------------------
    h_tree, q_tree = _bfs_tree_initial(net, index, source, demands)
    starts: list[np.ndarray] = [
        np.concatenate([h_tree[demand_positions], q_tree]),
        np.concatenate([np.full(n_dem, h_source), np.zeros(m)]),
        np.concatenate([np.full(n_dem, h_source - 1.0), np.zeros(m)]),
    ]

    def residual_norm(xx: np.ndarray) -> float:
        ff = residual(xx)
        return float(np.max(np.abs(ff))) if np.all(np.isfinite(ff)) else math.inf

    best_x: np.ndarray | None = None
    best_norm = math.inf
    best_method = ""
    best_iters = 0

    for x0 in starts:
        for method, options in (
            ("hybr", {"col_deriv": True, "xtol": 1.0e-12, "maxfev": 4000}),
            ("lm", {"col_deriv": True, "xtol": 1.0e-12, "maxiter": 2000,
                    "factor": 0.1}),
        ):
            try:
                sol = optimize.root(
                    residual, x0, jac=jacobian, method=method, options=options)
                x_try = np.asarray(sol.x, dtype=float)
                nrm = residual_norm(x_try) if np.all(np.isfinite(x_try)) else math.inf
                iters = int(getattr(sol, "njev", None) or getattr(sol, "nfev", 0) or 0)
            except Exception:  # noqa: BLE001 - 任何数值异常都转由兜底策略处理
                nrm, x_try, iters = math.inf, x0, 0
            if nrm < best_norm:
                best_norm, best_x, best_method, best_iters = nrm, x_try, method, iters
        x_bt, iters_bt, nrm_bt = _newton_backtrack(residual, jacobian, x0)
        if nrm_bt < best_norm:
            best_norm, best_x, best_method, best_iters = (
                nrm_bt, x_bt, "newton-backtrack", iters_bt)
        if best_norm <= SOLVER_TARGET_TOL:
            break

    if best_x is None or best_norm > SOLVER_TARGET_TOL:
        raise SolverNotConverged(
            "非线性方程组未能收敛到 1e-9 内部容差"
            f"（最佳残差 ‖f‖∞ = {best_norm:.3e}）。"
            "请检查阻力与需水量的数量级是否过于悬殊后重试。")

    # ---- 组装结果并独立复核 ----------------------------------------------
    h = fixed_head.copy()
    h[demand_positions] = best_x[:n_dem]
    q = best_x[n_dem:].copy()
    q[q == 0.0] = 0.0  # 归一化 -0.0，便于界面显示
    net_inflow = -A @ q  # A·q 为净流出，取负即净流入

    pipe_results: list[dict[str, Any]] = []
    max_pipe_res = 0.0
    for p, pipe in enumerate(net.pipes):
        u, v = index[pipe["start"]], index[pipe["end"]]
        qp = float(q[p])
        dh = float(h[u] - h[v])
        pipe_res = dh - r[p] * qp * abs(qp)
        max_pipe_res = max(max_pipe_res, abs(pipe_res))
        if qp > 0.0:
            direction = f"{pipe['start']} → {pipe['end']}"
        elif qp < 0.0:
            direction = f"{pipe['end']} → {pipe['start']}"
        else:
            direction = "零流量（无流向）"
        pipe_results.append({
            "id": pipe["id"],
            "start": pipe["start"],
            "end": pipe["end"],
            "r": float(r[p]),
            "flow": qp,
            "direction": direction,
            "head_start": float(h[u]),
            "head_end": float(h[v]),
            "head_drop": dh,
            "head_loss": abs(dh),
            "residual": float(pipe_res),
        })

    node_results: list[dict[str, Any]] = []
    max_node_res = 0.0
    below: list[str] = []
    for i, node in enumerate(net.nodes):
        row: dict[str, Any] = {
            "id": node["id"],
            "type": node["type"],
            "head": float(h[i]),
            "net_inflow": float(net_inflow[i]),
            "below_min_head": bool(h[i] < net.min_head),
        }
        if h[i] < net.min_head:
            below.append(node["id"])
        if node["type"] == "source":
            row["fixed_head"] = node["head"]
            row["outflow"] = float(-net_inflow[i])  # 水源经管道净流出 = 供水量（正值）
        else:
            row["demand"] = node["demand"]
            node_res = float(net_inflow[i] - demands[i])
            row["residual"] = node_res
            max_node_res = max(max_node_res, abs(node_res))
        node_results.append(row)

    total_demand = float(np.sum(demands))
    source_supply = float(-net_inflow[source])  # 水源净流出 = 总供水量
    balance_gap = source_supply - total_demand

    checks_ok = (
        max_node_res <= CHECK_TOL
        and max_pipe_res <= CHECK_TOL
        and abs(balance_gap) <= CHECK_TOL
    )
    if not checks_ok:  # 理论上不会发生：求解器内部容差更严
        raise SolverNotConverged(
            "解虽已求出，但按 1e-6 容差复核未通过："
            f"节点最大残差 {max_node_res:.3e}，管段最大残差 {max_pipe_res:.3e}，"
            f"供耗差 {balance_gap:.3e}。")

    return {
        "converged": True,
        "tolerance": CHECK_TOL,
        "method": best_method,
        "iterations": best_iters,
        "solver_residual_inf": best_norm,
        "node_results": node_results,
        "pipe_results": pipe_results,
        "heads": {node["id"]: float(h[index[node["id"]]]) for node in net.nodes},
        "flows": {pipe["id"]: float(q[p]) for p, pipe in enumerate(net.pipes)},
        "source_node": net.nodes[source]["id"],
        "source_supply": source_supply,
        "total_demand": total_demand,
        "balance_residual": balance_gap,
        "max_node_residual": max_node_res,
        "max_pipe_residual": max_pipe_res,
        "min_head": net.min_head,
        "below_min_head": below,
    }


# --------------------------------------------------------------------------- #
# 逐根停管情景分析
# --------------------------------------------------------------------------- #
def _reachable_nodes(net: Network, disabled: int) -> set[str]:
    """无向 BFS：返回不使用第 disabled 根管段时，从水源可达的全部节点编号。"""
    source_id = next(n["id"] for n in net.nodes if n["type"] == "source")
    adj: dict[str, list[str]] = {n["id"]: [] for n in net.nodes}
    for p, pipe in enumerate(net.pipes):
        if p == disabled:
            continue
        adj[pipe["start"]].append(pipe["end"])
        adj[pipe["end"]].append(pipe["start"])
    seen = {source_id}
    queue = deque([source_id])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def analyze_pipe_outages(net: Network) -> dict[str, Any]:
    """对每根管段各构造一个“移除该管”的独立情景并评估后果。

    情景分为三类：
      * disconnected —— 有用水节点与水源断开，列出失供节点与失供需水量之和，
        不生成水头；
      * connected    —— 全部节点仍与水源连通，重新稳态求解，列出最低用水节点
        水头、低于当前限值的节点及相对基线新增的低水头节点；
      * not_converged —— 该情景单独不收敛，标明后排在最后，不影响其余情景。

    排序：断联情景按失供量降序在前，连通情景按最低用水节点水头升序随后，
    同分保持管段录入顺序；不收敛情景一律排最后（同分同样保持录入顺序）。
    """
    baseline = solve_network(net)
    demand_map = {n["id"]: n["demand"] for n in net.nodes if n["type"] == "demand"}
    baseline_below = set(baseline["below_min_head"])

    scenarios: list[dict[str, Any]] = []
    for p, pipe in enumerate(net.pipes):
        scenario: dict[str, Any] = {
            "rank": 0,
            "pipe_id": pipe["id"],
            "pipe_index": p,
            "removed_pipe": {"id": pipe["id"], "start": pipe["start"],
                             "end": pipe["end"], "r": pipe["r"]},
        }
        reachable = _reachable_nodes(net, p)
        unserved = [nid for nid in demand_map if nid not in reachable]
        if unserved:
            scenario.update({
                "status": "disconnected",
                "unserved_nodes": unserved,
                "unserved_demand": float(sum(demand_map[nid] for nid in unserved)),
            })
            scenarios.append(scenario)
            continue

        # 仍连通：在剔除该管段的子网上重新求解
        sub_net = Network(
            nodes=net.nodes,
            pipes=net.pipes[:p] + net.pipes[p + 1:],
            min_head=net.min_head,
        )
        try:
            result = solve_network(sub_net)
        except SolverNotConverged as exc:
            scenario.update({"status": "not_converged", "error": str(exc)})
            scenarios.append(scenario)
            continue

        heads = result["heads"]
        below = result["below_min_head"]
        demand_ids = list(demand_map)
        scenario.update({
            "status": "connected",
            "min_demand_head": float(min(heads[nid] for nid in demand_ids)),
            # 水头并列时 min 取录入顺序的第一个节点
            "min_demand_head_node": min(demand_ids, key=lambda nid: heads[nid]),
            "below_min_head": below,
            "new_below_min_head": [nid for nid in below if nid not in baseline_below],
            "source_supply": result["source_supply"],
            "total_demand": result["total_demand"],
            "solution": result,
        })
        scenarios.append(scenario)

    def sort_key(s: dict[str, Any]) -> tuple[int, float, int]:
        if s["status"] == "disconnected":
            # 失供量越大越靠前：取负实现降序
            return (0, -s["unserved_demand"], s["pipe_index"])
        if s["status"] == "connected":
            return (1, s["min_demand_head"], s["pipe_index"])
        return (2, 0.0, s["pipe_index"])

    scenarios.sort(key=sort_key)
    for rank, s in enumerate(scenarios, start=1):
        s["rank"] = rank

    counts = {"disconnected": 0, "connected": 0, "not_converged": 0}
    for s in scenarios:
        counts[s["status"]] += 1

    return {
        "min_head": net.min_head,
        "baseline": baseline,
        "baseline_below_min_head": baseline["below_min_head"],
        "scenarios": scenarios,
        "counts": counts,
    }
