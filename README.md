# PIDS-Experiments

基于内核审计日志的溯源图（Provenance Graph）入侵检测学习实验。

## 目录结构

```
├── 02_experiments/        # 被测程序（文件操作 / 父子进程 / HTTP 网络）
├── 03_graph_builder/      # 日志 → 溯源图构建与实验分析脚本
├── 04_graph_learning/     # 图学习探针（GraphSAGE/GCN/GAT/GraphMAE/TGN）
├── 05_magic_code/         # MAGIC 适配 Kellect（实体级端到端检测）
└── run_collect.ps1        # Kellect 采集启动器（需管理员权限）
```

## 环境要求

- Windows 10 1809+（Kellect 采集器依赖管理员权限 + ETW）
- Python 3.10、PyTorch、torch_geometric、networkx
- KellectAgent-Windows（来自 [acising/KellectAgent-Windows](https://github.com/acising/KellectAgent-Windows)）

## 快速开始

```powershell
# 1. 采集（管理员 PowerShell）
.\run_collect.ps1 -Exp 3_2 -Script 02_experiments\parent_child.py

# 2. 建图
py -3 03_graph_builder\build_graph.py

# 3. 图学习探针
.\.venv\Scripts\python.exe 04_graph_learning\graphsage_probe.py
```

## MAGIC 适配 Kellect（实体级端到端检测）

对顶会论文 **MAGIC**（*Detecting Advanced Persistent Threats via Masked Graph Representation Learning*, USENIX Security 2024）的开源代码做 Kellect 日志适配：

- 原仓库：[FDUDSDE/MAGIC](https://github.com/FDUDSDE/MAGIC)
- 本仓库只保留**适配代码 + 小批量检测结果**（不含 MAGIC 完整源码），详见 [`05_magic_code/MAGIC_kellect/`](05_magic_code/MAGIC_kellect/README.md)
- 结果摘要：AUC **0.6993**，TP=3 / FP=2 / FN=2（恶意实体 = 3 个网络外泄端点全部检出）
