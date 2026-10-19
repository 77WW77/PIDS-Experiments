"""
阶段 5 探针 ④：TGN（Temporal Graph Networks）—— 简化版（记忆路线）

────────────────────────────────────────────────────────────────
与前三个探针的本质差别
────────────────────────────────────────────────────────────────
  SAGE/GCN/GAT/GraphMAE 都在【静态聚合图】上干活：
    把整个窗口的事件压成一张图，节点特征（度数、事件类型分布）
    是按【全窗口】统计的 —— 预测 t 时刻的事件时，特征里已经
    含了 t 之后的信息（未来泄漏，因果被破坏）。

  TGN 处理【事件流】：每条事件 (src, dst, t, type) 按时间进来，
  每个节点有一份【记忆】(memory) 随事件滚动更新。预测 t 时刻
  的事件，只可能用到 t 之前更新过的记忆 —— 严格因果。

  因此本探针【故意不用】静态 16 维特征（那会把未来塞进输入），
  节点初始记忆全 0，信息完全来自事件流的因果历史。

────────────────────────────────────────────────────────────────
数据与任务
────────────────────────────────────────────────────────────────
  数据：exp3_3 网络场景（read.json），建图后按边时间戳抽事件流。
  切分：按时间前 60% 训练 / 后 40% 测试（严格时序，无泄漏）。
  任务：时序链接预测 —— 给定当前记忆，预测 (src, dst) 这条边
        会不会发生。正样本 = 真实事件；负样本 = 随机换 dst。
  指标：AUC（正样本分数 > 负样本分数的比例），随机基线 = 0.5。

  诚实边界：
  · exp3_3 全是正常日志，没有攻击。AUC 高只说明"学会了正常的
    时序模式"，不等于"能发现攻击"（可重建 ≠ 可检测的同款陷阱）。
  · 本实现是 TGN 框架的【记忆路线】，省略了 TGAT 式邻居
    attention 聚合层 —— 保留 memory + GRU 更新 + 时间编码 + 链接预测。
────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────
    .\.venv\Scripts\python.exe 04_graph_learning\tgn_probe.py --epochs 5
"""

import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_graph_builder"))
import build_graph as BG          # noqa: E402  复用 4.3 建图

LOG = os.path.join(ROOT, "01_kellect_logs", "exp3_3", "read.json")
TAGS = BG.SCRIPT_TAGS             # ("http_client.py", "local_http_server.py")
ETYPES = ["Create", "Read", "Write", "CreateFile", "Open", "Delete",
          "Connect", "Accept", "Send", "Recv", "Disconnect"]


# ============================================================ 事件流
def load_event_stream(path, tags):
    """日志 → 建图 → 有向事件流 [(t, src, dst, etype_id), ...]，按时间升序。"""
    records, bad = BG.load_records(path)
    keep = BG.focus_pids(records, tags=tags)
    records = [r for r in records if r["PID"] in keep]
    G = BG.build_graph(records, include_open=True)
    nodes = list(G.nodes())
    nid = {n: i for i, n in enumerate(nodes)}
    events = []
    for u, v, _k, d in G.edges(keys=True, data=True):
        events.append((d["ts"], nid[u], nid[v], ETYPES.index(d["etype"])))
    events.sort(key=lambda e: e[0])       # 严格按时间升序
    return nodes, events


# ============================================================ 时间编码
def time_enc(dt_sec, dim):
    """cosine 时间编码（思路同 Transformer 位置编码），输入 dt 秒。"""
    out = []
    for k in range(dim // 2):
        w = 1.0 / (10000 ** (2.0 * k / dim))
        out.append(torch.cos(w * dt_sec))
        out.append(torch.sin(w * dt_sec))
    return torch.stack(out, dim=-1)


# ============================================================ 简化 TGN
class TGN(nn.Module):
    """简化版 TGN：节点记忆 + GRU 更新 + 时间编码 + 链接预测头。

    记忆是【状态】不是参数：初始全 0，每条真实事件触发更新，
    更新结果被下一条事件消费。可学习的是 msg/gru/t_proj/score 的权重。
    """

    def __init__(self, num_nodes, hidden=32, n_etypes=11, time_dim=16):
        super().__init__()
        self.hidden = hidden
        self.time_dim = time_dim
        # 记忆是【状态】不是参数：普通属性（非 buffer），随事件滚动替换。
        self.mem = torch.zeros(num_nodes, hidden)
        self.last_update = [0.0] * num_nodes
        self.etype_emb = nn.Embedding(n_etypes, hidden)
        self.t_proj = nn.Linear(time_dim, hidden)
        self.msg = nn.Linear(2 * hidden + hidden, hidden)   # src,dst,etype
        self.gru = nn.GRUCell(hidden, hidden)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def reset_memory(self, first_t):
        self.mem.zero_()
        self.last_update = [float(first_t)] * self.mem.size(0)

    def _repr(self, node, t):
        """节点当前表示 = 记忆(状态, detach) + 时间编码(距上次更新多久)。

        记忆是纯状态不进计算图：可学习的是 t_proj/score/etype_emb/msg/gru
        的权重，记忆本身只是按事件滚动更新的值。
        """
        dt_sec = torch.log1p(torch.tensor(
            (t - self.last_update[node]) / 1e7, dtype=torch.float32))
        return self.mem[node].detach() + self.t_proj(
            time_enc(dt_sec, self.time_dim))

    def score_edge(self, s, d, e, t):
        """打分一条候选边（不更新记忆）。"""
        hs = self._repr(s, t)
        hd = self._repr(d, t)
        ee = self.etype_emb(torch.tensor(e))
        return self.score(torch.cat([hs, hd, ee], dim=-1))

    def update_mem(self, s, d, e, t):
        """真实事件发生后滚动更新两端记忆。

        简化：整个记忆更新在 no_grad 下执行（msg/gru 视为固定规则，
        本轮探针不训练它们 —— 记忆是"状态"，学习集中在 t_proj/score）。
        """
        with torch.no_grad():
            ee = self.etype_emb(torch.tensor(e))
            msg = F.relu(self.msg(torch.cat(
                [self.mem[s], self.mem[d], ee], dim=-1)))
            new_mem = self.mem.clone()
            new_mem[s] = self.gru(msg, self.mem[s])
            new_mem[d] = self.gru(msg, self.mem[d])
            self.mem = new_mem
            self.last_update[s] = self.last_update[d] = float(t)


def sample_neg(s, d, n_nodes):
    """采样一个 ≠ s 且 ≠ d 的随机节点当负样本目标。"""
    while True:
        x = random.randrange(n_nodes)
        if x != s and x != d:
            return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--split", type=float, default=0.6, help="训练事件占比")
    ap.add_argument("--negs", type=int, default=5, help="测试期每事件负样本数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("1. 建图 → 有向事件流（按时间升序）")
    print("=" * 72)
    nodes, events = load_event_stream(args.log, TAGS)
    n_nodes, n_evts = len(nodes), len(events)
    etype_cnt = Counter(e for _t, _s, _d, e in events)
    print(f"  {n_nodes} 节点, {n_evts} 条事件")
    print(f"  事件类型分布: {dict(sorted(etype_cnt.items(), key=lambda x: -x[1]))}")

    cut = int(n_evts * args.split)
    tr_evts = events[:cut]
    te_evts = events[cut:]
    print(f"  训练段: 前 {len(tr_evts)} 条 (t<={events[cut-1][0]})")
    print(f"  测试段: 后 {len(te_evts)} 条 (t>={events[cut][0]})")
    print(f"  [关键] 节点初始记忆全 0，特征零输入 —— 信息只来自事件流因果历史")

    print("\n" + "=" * 72)
    print(f"2. 训练（epochs={args.epochs}，每轮重置记忆、按时间顺序在线更新）")
    print("=" * 72)
    model = TGN(n_nodes, args.hidden, len(ETYPES))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    first_t = events[0][0]

    for ep in range(1, args.epochs + 1):
        model.reset_memory(first_t)
        total, cnt = 0.0, 0
        for t, s, d, e in tr_evts:
            neg = sample_neg(s, d, n_nodes)
            pos = model.score_edge(s, d, e, t)
            neg_l = model.score_edge(s, neg, e, t)
            loss = F.binary_cross_entropy_with_logits(
                pos - neg_l, torch.ones_like(pos))
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_mem(s, d, e, t)
            total += float(loss)
            cnt += 1
        print(f"  epoch {ep}: 平均 loss = {total / cnt:.4f}")

    print("\n" + "=" * 72)
    print(f"3. 测试（部署模式：记忆贯穿全部事件滚动更新，只对测试段打分）")
    print("=" * 72)
    model.reset_memory(first_t)
    pos_scores, neg_scores = [], []
    for i, (t, s, d, e) in enumerate(events):
        if i >= cut:                          # 测试段事件 → 打分
            pos = float(model.score_edge(s, d, e, t))
            pos_scores.append(pos)
            ns = [float(model.score_edge(s, sample_neg(s, d, n_nodes), e, t))
                  for _ in range(args.negs)]
            neg_scores.append(ns)
        model.update_mem(s, d, e, t)          # 真实事件一律更新记忆

    # AUC：正样本分数 > 每个负样本分数的比例
    ok = total_pairs = 0
    for p, nsl in zip(pos_scores, neg_scores):
        for ns in nsl:
            ok += 1 if p > ns else 0
            total_pairs += 1
    auc = ok / total_pairs if total_pairs else float("nan")
    print(f"  测试事件 {len(pos_scores)} 条 × {args.negs} 负样本 = "
          f"{total_pairs} 个正负对")
    print(f"  链接预测 AUC = {auc:.4f}   （随机基线 0.5000）")

    print("""
  判读要点（诚实边界）：
  · AUC > 0.5 说明模型学会了"正常时序模式"（谁的下一跳通常是谁），
    但它只见过正常日志 —— 就像 GraphMAE 能重建不等于能检测，
    TGN 能预测正常流也不等于能抓攻击。
  · 本探针的意义是机制性的：TGN 用【记忆 + 时间】做严格因果的
    逐事件预测，这是静态聚合图（前三个探针）做不到的实时能力；
    代价是要在线滚动，记忆无法跨图迁移。
  · 想让它变检测器，还缺：正常/异常分界（阈值）、攻击先验
    （攻击的边在"正常流模型"下应得低分，需注入攻击样本验证）。
    """)
    out = os.path.join(ROOT, "04_graph_learning", "tgn_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"epochs={args.epochs} hidden={args.hidden} split={args.split} "
                f"negs={args.negs} seed={args.seed}\n")
        f.write(f"AUC={auc:.4f}\n")
    print(f"  结果已存: {out}")


if __name__ == "__main__":
    main()
