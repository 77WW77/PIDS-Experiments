# MAGIC 适配 Kellect 日志（实体级端到端检测）

本目录只包含**为适配 Kellect 日志所改编/新增的代码**与**小批量数据检测结果**，不含 MAGIC 完整源码。

- 原论文：**MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning**（USENIX Security 2024）
- 原仓库：[FDUDSDE/MAGIC](https://github.com/FDUDSDE/MAGIC)（完整源码、数据、README 见原仓库）

## 背景

MAGIC 官方输入是 DARPA TC / StreamSpot / Wget 等解析好的实体级图数据
（`data/<ds>/train.pkl` / `test.pkl` / `malicious.pkl`，NetworkX `node_link_data` 格式）。
本实验用 Windows ETW 采集器 **Kellect** 采集的几分钟小批量日志（`01_kellect_logs/exp3_3/`）替代：

- `fixed.json` = 良性基线：服务端从不碰敏感文件，只回固定串
- `read.json`   = 攻击：服务端真实读取 `secret.txt` 并经本地 HTTP 发出（模拟数据外泄）

## 适配内容

| 文件 | 类型 | 说明 |
|---|---|---|
| `utils/kellect_parser.py` | 新增 | Kellect 日志 → MAGIC 实体级数据三件套（train/test/malicious.pkl）。复用 PIDS 仓库 `03_graph_builder/build_graph.py` 的建图逻辑（进程/文件/网络端点实体识别、边方向统一控制流） |
| `model/eval.py` | 改编 | 官方阈值选择对 trace/theia/cadets 有"峰值性能"特判，其余数据集兜底 `threshold[-1]`（切得极严）。新增分支：非官方数据集改用标准 F1 最优阈值 `np.argmax(f1)` |
| `run_kellect.py` | 新增 | 一键编排：进入 MAGIC 仓库目录 → 生成数据 → `train.py --dataset kellect` → `eval.py --dataset kellect` |

## 运行方式

1. 按[原仓库](https://github.com/FDUDSDE/MAGIC) README 安装依赖（PyTorch、DGL、scikit-learn），并把 MAGIC 仓库放到本仓库 `05_magic_code/MAGIC`；
2. 将 `utils/kellect_parser.py`、`model/eval.py` 放入对应目录（覆盖/新增）；
3. 准备 Kellect 实验日志 `01_kellect_logs/exp3_3/{fixed,read}.json`（由 PIDS 仓库 `run_collect.ps1` 采集）；
4. 运行 `python run_kellect.py`，端到端结果自动写入 `results/kellect_eval_raw.txt`。

## 结果（小批量）

见 [`results/kellect_eval_result.txt`](results/kellect_eval_result.txt)。

要点：

- **AUC 0.6993**，F1 0.6，TP=3 / FP=2 / FN=2 / TN=272
- 检出的 3 个恶意实体 = 3 个网络端点（数据外泄通道）——在 3 层 GAT 下，端点的多跳邻域编码了"服务端读取 secret.txt"的行为，是该图下最敏感的实体
- FP=2 为测试窗口新加载的系统 DLL（kernel.appcore.dll、winrnr.dll），属"良性但未见"的概念漂移
- FN=2 为 secret.txt 与读取它的进程：单热类型特征下与良性同类同构，score≈0

## 局限（观察结论）

- 官方仓库未提供论文 §4.4 模型自适应（反馈回注 + 折扣机制）的可执行实现，本目录亦未包含；
- 单热类型特征不含路径/内容信息，文件级细粒度攻击（如"哪个文件被读了"）无法在嵌入层区分——这是该场景下检测盲点的根源。
