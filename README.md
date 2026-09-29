# 闭环供水管网平衡台

本地运行的**闭环供水管网稳态平衡**工具：在浏览器中录入节点（1 个定水头水源 + 若干非负需水量的用水节点）与管段（正阻力），后端用 **Python / FastAPI + SciPy** 求解整个管网的稳态水头分布、各管段**带符号流量**与水源总供水量，并按 **1e-6 容差**独立复核节点水量守恒与每根管段的压降定律；前端用**原生 HTML/CSS/JavaScript** 提供可编辑表格、可拖拽的网络图、流向箭头与低水头节点告警。

## 一条命令启动

```bash
docker compose up --build
```

启动后在本机浏览器打开：

```
http://localhost:8000
```

停止：`Ctrl+C`（后台运行用 `docker compose down`）。

> 需要 Docker 20.10+（支持 Compose v2 插件）。端口被占用时修改 `docker-compose.yml` 里的 `"8000:8000"` 左侧端口即可。

## 不用 Docker 的本地运行方式

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

同样访问 http://localhost:8000 。

## 使用说明

1. **节点表**：2～10 个、编号唯一；类型二选一：
   - `水源`：恰好 1 个，填写**固定水头** H_s（有限数值）；
   - `用水`：填写**非负需水量** d。
2. **管段表**：1～16 根、编号唯一；选择**起点/终点**（必须不同，且任意两节点间最多一根管），填写**正阻力 r**。
3. 顶部填写**最低水头限值**，点击 **求解**。
4. 结果区展示：
   - 水源总供水量、总需水量、供耗差；
   - 各节点水头（低于限值的行与图上节点**标红**）；
   - 各管段带符号流量 q、实际流向、`H起−H终` 与压降残差；
   - 网络图上箭头表示**实际流向**（蓝色与录入方向同向，橙色为反向），数字为带符号流量。
5. 点击 **载入示例** 可快速查看一个 5 节点、7 管段的环状管网。
6. **逐根停管分析**：先完成一次稳态求解得到基线，再点击顶栏的 **停管分析**，
   后端对**每根管段各构造一个独立情景**（仅停用该管，其余不变）：
   - **断联情景**：移除后若有用水节点与水源断开，列出**失供节点**与**失供需水量之和**，
     不为该情景生成水头；
   - **连通情景**：仍全部连通则重新求解，列出**最低用水节点水头**、**低于当前最低水头限值的节点**
     以及**比基线新增的低水头节点**；
   - 某个情景单独**不收敛**时单独标明并**排在最后**，不影响其余情景。
   - 排序：断联情景按失供需水量**降序**排在最前，连通情景按最低用水节点水头**升序**随后，
     同分保持管段录入顺序。
   - 只要**编辑节点、管段或最低水头限值**，旧的停管分析立即清除（需要重新点击“停管分析”）。

输入有问题（非有限数值、负需水量、阻力非正、重复编号、端点悬空、图不连通、水源数 ≠ 1 等）或求解器未收敛时，页面会用中文说明具体原因。

## 水力模型

管段 p 按录入方向记 i→j，带符号流量 q_p（正：i→j；负：j→i），采用二次压降定律：

```
H_i − H_j = r_p · q_p · |q_p|
```

每个用水节点 k 满足水量守恒（**净流入 = 需水量**）：

```
Σ_p a_{k,p} · q_p = −d_k
```

其中 a_{k,p} 为节点-管段关联矩阵（起点 +1、终点 −1），故 A·q 为节点净流出。水源节点给定固定水头，因此未知量（n−1 个用水节点水头 + m 个管段流量）与方程数恰好相等，构成平方非线性方程组，用 Newton 型方法求解：

- 以水源为根做 BFS 生成树，按子树需水量摊派初始流量、沿程估算初始水头；
- 依次尝试 `scipy.optimize.root` 的 `hybr`（MINPACK）与 `lm`（Levenberg–Marquardt），并配有带 Armijo 回退的自实现全量 Newton；
- 内部目标残差 1e-9，解出后再以 **1e-6** 容差独立复核每段压降、每个节点守恒以及水源供水量与总需水量之差。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/solve` | 入参 `{nodes, pipes, min_head}`，返回水头、流量、流向、供水量与各项残差；非法输入返回 `422`/`400` 与中文原因 |
| `POST` | `/api/analyze-outages` | 入参与 `/api/solve` 相同；先求基线，再对每根管段各评估一个“单独停用”情景并按风险排序返回 |
| `POST` | `/api/locate-leak` | 在 `/api/solve` 入参上增加 `measurements`（2～8 个用水节点实测水头）与正的 `leak_upper_bound`；逐节点拟合单点漏量并按测点 RMSE 排序 |
| `GET` | `/api/health` | 健康检查 |

请求示例：

```json
{
  "min_head": 0,
  "nodes": [
    {"id": "S", "type": "source", "head": 30.0},
    {"id": "A", "type": "demand", "demand": 2.0}
  ],
  "pipes": [
    {"id": "P1", "start": "S", "end": "A", "r": 1.0}
  ]
}
```

### 逐根停管分析 `POST /api/analyze-outages`

> 停管分析与求解共用同一服务、同一端口，**不需要任何额外服务或新的启动步骤**——
> `docker compose up --build` 后即可直接调用。

请求体与 `/api/solve` 完全一致（沿用同一套输入校验）。响应中 `analysis.baseline` 为基线稳态解
（结构与 `/api/solve` 的 `result` 相同），`analysis.scenarios` 为每根管段一个、已按风险排序的情景：

- `status: "disconnected"`：含 `unserved_nodes`（失供节点，按节点录入顺序）与
  `unserved_demand`（失供需水量之和），**不返回水头**；
- `status: "connected"`：含 `min_demand_head` / `min_demand_head_node`（最低用水节点水头及节点）、
  `below_min_head`（低于当前限值的节点）、`new_below_min_head`（比基线新增的低水头节点）
  以及 `solution`（该子网上的完整稳态解）；
- `status: "not_converged"`：含 `error`（中文原因），该情景单独排最后，其余情景不受影响。

排序键为：断联（组 0）按 `-unserved_demand`、连通（组 1）按 `min_demand_head` 升序、
不收敛（组 2）最后；组内同分按管段录入顺序。`counts` 给出三类情景的数量。

curl 调用示例：

```bash
curl -s http://localhost:8000/api/analyze-outages \
  -H 'Content-Type: application/json' \
  -d '{
    "min_head": 0,
    "nodes": [
      {"id": "S", "type": "source", "head": 50.0},
      {"id": "A", "type": "demand", "demand": 2.0},
      {"id": "B", "type": "demand", "demand": 3.0}
    ],
    "pipes": [
      {"id": "P1", "start": "S", "end": "A", "r": 1.0},
      {"id": "P2", "start": "A", "end": "B", "r": 1.0}
    ]
  }'
```

返回（节选）：两根管都是桥，停用 P1 会让 A、B 同时失供（Σd=5），排在停用 P2（仅 B 失供，Σd=3）之前：

```json
{
  "ok": true,
  "analysis": {
    "counts": {"disconnected": 2, "connected": 0, "not_converged": 0},
    "scenarios": [
      {"rank": 1, "pipe_id": "P1", "status": "disconnected",
       "unserved_nodes": ["A", "B"], "unserved_demand": 5.0},
      {"rank": 2, "pipe_id": "P2", "status": "disconnected",
       "unserved_nodes": ["B"], "unserved_demand": 3.0}
    ]
  }
}
```

### 单点漏损定位 `POST /api/locate-leak`

> 与求解共用同一服务、同一端口，**不需要任何额外服务或新的启动步骤**——
> `docker compose up --build` 后即可直接调用。

在既有管网描述（`nodes` / `pipes` / `min_head`，沿用同一套校验）之外增加：

- `measurements`：**2～8 个**实测水头测点，每个形如
  `{"node": "用水节点编号", "head": 实测水头（有限数值）}`；
  测点必须**互异且全部是用水节点**（不能是水源），顺序即测点录入顺序。
- `leak_upper_bound`：**正的有限数**，作为每个候选节点附加漏量的统一搜索上限。

排查假设为**至多一个用水节点增加漏量**（漏量视为该节点额外的节点出流量，
叠加在原需水量上）。处理流程：

1. 先求**无漏损基线**并给出测点水头均方根误差（RMSE）；基线不收敛直接返回
   `422` 并说明原因。
2. 对**每个用水节点**，在漏量区间 `[0, leak_upper_bound]` 上最小化
   测点预测水头相对实测水头的 RMSE：先在含两个端点的粗网格上定位最优区间，
   再做有界 Brent 细化，**端点 0 与上限都会显式参与最终比较**。
3. 返回无漏损基线误差，以及每个候选的**拟合漏量、各测点预测水头与逐点误差、
   RMSE**；候选按 RMSE **升序**排列，同分按节点录入顺序。
4. 若**至少两个**候选的 RMSE 与最优值相差不超过 **1e-4**，将它们
   （含最优候选本身）一并标为 `indistinguishable`（无法区分）；
   最优漏量为 0 时 `verdict` 为 `no_leak`，**不判定存在漏损**。
5. 某个候选在整个区间上都无法得到收敛解时标为
   `status: "fit_failed"`（含中文原因）并排末尾，不影响其余候选。

顶层结论字段：

- `verdict`：`located`（唯一定位）、`indistinguishable`（多候选同分无法区分）、
  `no_leak`（最优漏量为 0）、`fit_failed`（全部候选拟合失败）之一；
- `leak_detected` / `leak_node` / `leak_value`：仅在 `located` 时给出节点与漏量；
- `indistinguishable_nodes`：与最优误差相差不超过 1e-4 的候选节点（按录入顺序）；
- `best_rmse`、`baseline_rmse`、`counts`（`fitted` / `fit_failed` 数量）。

curl 调用示例：

```bash
curl -s http://localhost:8000/api/locate-leak \
  -H 'Content-Type: application/json' \
  -d '{
    "min_head": 0,
    "nodes": [
      {"id": "S", "type": "source", "head": 40.0},
      {"id": "A", "type": "demand", "demand": 1.0},
      {"id": "B", "type": "demand", "demand": 1.0},
      {"id": "C", "type": "demand", "demand": 1.0},
      {"id": "D", "type": "demand", "demand": 1.0}
    ],
    "pipes": [
      {"id": "P1", "start": "S", "end": "A", "r": 1.0},
      {"id": "P2", "start": "A", "end": "B", "r": 1.0},
      {"id": "P3", "start": "B", "end": "C", "r": 1.0},
      {"id": "P4", "start": "C", "end": "S", "r": 1.0},
      {"id": "P5", "start": "S", "end": "B", "r": 2.5},
      {"id": "P6", "start": "A", "end": "D", "r": 1.5}
    ],
    "measurements": [
      {"node": "A", "head": 36.60731},
      {"node": "C", "head": 37.11871},
      {"node": "D", "head": 35.10731}
    ],
    "leak_upper_bound": 3.0
  }'
```

返回（节选）：实测水头由“B 点额外漏 0.7”的真值解生成，分析唯一定位到
`B`、漏量 `0.7`，无漏损基线 RMSE 约 0.853：

```json
{
  "ok": true,
  "analysis": {
    "verdict": "located",
    "leak_detected": true,
    "leak_node": "B",
    "leak_value": 0.7,
    "baseline_rmse": 0.8525,
    "best_rmse": 2.0e-10,
    "indistinguishable_nodes": ["B"],
    "counts": {"fitted": 4, "fit_failed": 0},
    "baseline_predicted_heads": [
      {"node": "A", "measured_head": 36.60731,
       "predicted_head": 37.5006, "error": 0.8933}
    ],
    "candidates": [
      {"rank": 1, "node_id": "B", "status": "fitted", "leak": 0.7,
       "rmse": 2.0e-10, "at_bound": false, "indistinguishable": true,
       "predicted_heads": [
         {"node": "A", "measured_head": 36.60731,
          "predicted_head": 36.60731, "error": -2.2e-10}
       ]}
    ]
  }
}
```

测点互异、测点必须为用水节点、水头与上限为有限数且上限为正等要求不满足时，
返回 `422` 与中文原因；请求体为空或 JSON 非法返回 `400`。

## 输入约束汇总

- 节点 2～10 个，编号为非空且唯一的字符串；
- 恰好 1 个 `source`（固定水头，有限数值），其余为 `demand`（需水量为非负有限数）；
- 管段 1～16 根，编号唯一，起点/终点均为已定义且互不相同的节点，同一对节点之间只允许一根管段，阻力 r 为正有限数；
- 管网必须是**无向连通图**（孤立节点/孤岛一律拒绝）；
- 不接受 `NaN` / `Infinity`（含 JSON 文本中的 `NaN`/`Infinity` 记号）。

`/api/locate-leak` 额外要求：

- `measurements` 为 2～8 个测点，`node` 必须是**已定义的用水节点**（不能是水源），
  各测点**互异**，`head` 为有限数值；
- `leak_upper_bound` 为正的有限数值。

## 项目结构

```
.
├── app/
│   ├── main.py            # FastAPI 入口：/api/solve、/api/analyze-outages、/api/locate-leak、/api/health、静态页面
│   ├── solver.py          # 输入校验 + Newton 稳态求解 + 1e-6 复核 + 逐根停管情景分析 + 单点漏损定位
│   └── static/
│       ├── index.html     # 原生页面（含停管分析入口与结果表）
│       ├── style.css
│       └── app.js         # 可编辑表格、力图布局、拖拽、流向绘制、停管分析
├── tests/test_solver.py   # pytest：解析解、环状网、校验分支、API、随机压力测试、停管分析排序
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 测试

```bash
pip install -r requirements.txt pytest httpx
PYTHONPATH=. pytest -q
```

测试包含单管解析解、反向录入符号、环状网独立复核、低水头标记、各类非法输入、`NaN` JSON 拒绝，以及 60 个随机连通管网（2～10 节点、最多 16 管、阻力跨 6 个数量级）的收敛性与残差检查；停管分析部分覆盖树状网全断联按失供量排序、环+枝混合分组排序、比基线新增低水头节点、同分保持录入顺序、单情景不收敛单独排最后，以及 `/api/analyze-outages` 的 200/422/400 分支。单点漏损定位部分覆盖已知漏点漏量的唯一定位、漏量恰在上限端点时标记 `at_bound`、实测与基线一致时判 `no_leak`、对称测点下多候选 1e-4 同分判 `indistinguishable`（含同分按录入顺序）、各类测点/上限非法输入、基线不收敛透传、单候选全部漏量拟合失败时标 `fit_failed` 排末尾，以及 `/api/locate-leak` 的 200/422/400 分支。
