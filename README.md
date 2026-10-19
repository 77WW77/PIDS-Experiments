# PIDS-Experiments

基于内核审计日志的溯源图（Provenance Graph）入侵检测学习实验。

## 目录结构

```
├── 02_experiments/        # 被测程序（文件操作 / 父子进程 / HTTP 网络）
├── 03_graph_builder/      # 日志 → 溯源图构建与实验分析脚本
├── 04_graph_learning/     # 图学习探针（GraphSAGE/GCN/GAT/GraphMAE/TGN）
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
