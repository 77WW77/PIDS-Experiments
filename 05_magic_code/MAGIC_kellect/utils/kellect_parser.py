"""
Kellect 日志 → MAGIC 实体级输入格式（data/<ds>/train.pkl / test.pkl / malicious.pkl）

设计思路（对齐阶段 3.3 的 read vs fixed 对照实验）：
  · 训练（良性）: exp3_3/fixed.json —— 服务端从不碰 secret.txt，只回固定串
  · 测试（含攻击）: exp3_3/read.json —— 服务端真实读取 secret.txt 并经网络发出（模拟外泄）
  · 恶意实体（GT）: secret.txt 文件节点 + 读取它的服务端进程节点

本解析器复用项目 `03_graph_builder/build_graph.py` 的建图逻辑：
  1) load_records 解析 Kellect 日志（非标准 JSON，需先转义反斜杠）
  2) focus_pids 过滤到被测程序进程子树（去掉系统噪声）
  3) build_graph 建 MultiDiGraph（控制流方向统一、实体身份规范化）
  4) 塌缩成 DiGraph：MAGIC 的 dgl.from_networkx 需要简单图，
     每对 (src, dst) 只保留时间上最早的一条边（与 trace_parser 的 has_edge 行为一致）
  5) 全局类型表：节点类型 process/file/network → int；边类型 → int（跨图一致）
  6) 写出 MAGIC 实体级所需的三个 pkl

用法（在 MAGIC 仓库根目录运行）：
  ..\..\.venv\Scripts\python.exe utils\kellect_parser.py
"""

import argparse
import json
import os
import pickle as pkl
import sys

import networkx as nx

# 复用项目建图模块（保持单一事实来源）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "03_graph_builder"))
import build_graph

DEFAULT_TRAIN = os.path.join(PROJECT_ROOT, "01_kellect_logs", "exp3_3", "fixed.json")
DEFAULT_TEST = os.path.join(PROJECT_ROOT, "01_kellect_logs", "exp3_3", "read.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kellect")

# 节点/边类型 → 整数 id 的全局映射（train+test 共享，保证 one-hot 维度一致）
NODE_TYPES = {"process": 0, "file": 1, "network": 2}
EDGE_TYPES = {}
_edge_counter = 0


def _edge_type_id(name):
    """边类型 → 全局递增整数 id。首次见到就分配新 id。"""
    global _edge_counter
    if name not in EDGE_TYPES:
        EDGE_TYPES[name] = _edge_counter
        _edge_counter += 1
    return EDGE_TYPES[name]


def build_kellect_graph(log_path, tags):
    """日志 → 聚焦到实验进程子树的 MultiDiGraph。"""
    records, bad = build_graph.load_records(log_path)
    keep = build_graph.focus_pids(records, tags)
    records = [r for r in records if r["PID"] in keep]
    G = build_graph.build_graph(records, include_open=True)
    print(f"  {os.path.basename(log_path)}: {len(records)} 条聚焦事件"
          f"（坏行 {bad}），节点 {G.number_of_nodes()}，边 {G.number_of_edges()}")
    return G, records


def to_digraph(G):
    """MultiDiGraph → DiGraph。

    塌缩规则：按时间序遍历边，每对 (src, dst) 只保留最早一条（同 trace_parser）。
    节点保留 ntype/label 属性，边只保留 etype（MAGIC 只读这两个 'type' 特征）。
    """
    D = nx.DiGraph()
    for n, attrs in G.nodes(data=True):
        D.add_node(n, type=NODE_TYPES[attrs["ntype"]], label=attrs.get("label", ""))
    # 时间序：raw_index 越小越早（records 已按时间排序）
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get("raw_index", 0))
    for u, v, attrs in edges:
        if not D.has_edge(u, v):
            D.add_edge(u, v, type=_edge_type_id(attrs["etype"]))
    return D


def find_malicious_nodes(G, records):
    """在测试图里自动定位攻击链实体：
       · secret.txt 文件节点 + 读取它的进程节点（数据 + 执行者）
       · 网络端点节点（数据外泄通道 —— 3.3 场景中 HTTP 连接承载了外泄内容）

    实测诊断依据：3 层 GAT 让端点的多跳邻域编码了"进程读取 secret.txt"，
    故端点在该场景下是检测器最敏感的实体。
    """
    mal = []
    seen = set()
    for n, attrs in G.nodes(data=True):
        if attrs.get("ntype") == "file" and "secret" in (attrs.get("path") or "").lower():
            if n not in seen:
                mal.append((n, "secret.txt"))
                seen.add(n)
            # MultiDiGraph 里同一进程对 secret.txt 可能有多次 Read，按进程去重
            readers = {u for u, _v, d in G.in_edges(n, data=True) if d["etype"] == "Read"}
            for p in readers:
                if p not in seen:
                    mal.append((p, G.nodes[p].get("label", str(p))))
                    seen.add(p)
        elif attrs.get("ntype") == "network":
            if n not in seen:
                mal.append((n, "endpoint:" + " | ".join(attrs.get("label", "").split(" | "))))
                seen.add(n)
    return mal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--tags", nargs="*", default=["http_client.py", "local_http_server.py"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("== Kellect → MAGIC 实体级转换 ==")

    # 训练图（良性）
    train_g, train_records = build_kellect_graph(args.train, args.tags)
    train_d = to_digraph(train_g)
    # 测试图（含攻击）
    test_g, test_records = build_kellect_graph(args.test, args.tags)
    test_d = to_digraph(test_g)

    for name, d in [("train(DiGraph)", train_d), ("test(DiGraph)", test_d)]:
        ntypes = {}
        for _n, a in d.nodes(data=True):
            ntypes[a["type"]] = ntypes.get(a["type"], 0) + 1
        print(f"  {name}: 节点 {d.number_of_nodes()}，边 {d.number_of_edges()}，节点类型分布 {ntypes}")

    # GT：测试图节点序号（n_test=1 时即图内序号）
    mal_entities = find_malicious_nodes(test_g, test_records)
    node_order = list(test_d.nodes())
    idx_map = {n: i for i, n in enumerate(node_order)}
    malicious_idx = [idx_map[n] for n, _name in mal_entities]
    malicious_names = [name for _n, name in mal_entities]
    print(f"\n恶意实体 {len(malicious_idx)} 个: "
          f"{[(name, idx) for name, idx in zip(malicious_names, malicious_idx)]}")

    # 写 MAGIC 实体级三件套
    with open(os.path.join(OUT_DIR, "train.pkl"), "wb") as f:
        pkl.dump([nx.node_link_data(train_d)], f)
    with open(os.path.join(OUT_DIR, "test.pkl"), "wb") as f:
        pkl.dump([nx.node_link_data(test_d)], f)
    with open(os.path.join(OUT_DIR, "malicious.pkl"), "wb") as f:
        pkl.dump((malicious_idx, malicious_names), f)

    print(f"\n输出到 {OUT_DIR}/")
    print(f"  train.pkl: 1 图（良性 fixed.json）")
    print(f"  test.pkl:  1 图（攻击 read.json）")
    print(f"  类型表: 节点 {NODE_TYPES}，边 {EDGE_TYPES}")


if __name__ == "__main__":
    main()
