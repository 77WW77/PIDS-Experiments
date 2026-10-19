"""
阶段 5 探针 ③：GraphMAE（Masked Graph Autoencoder）—— 自监督掩码图自编码器

────────────────────────────────────────────────────────────────
与探针 ①/② 的关系
────────────────────────────────────────────────────────────────
  探针 ①（SAGE/GCN）②（GAT）都是【端到端监督】：直接用节点类型标签
  训练图卷积 + 分类头。
  GraphMAE 是【自监督】：训练时完全不碰标签，只做
      随机 mask 节点特征 → 编码器 → 解码器 → 重建被 mask 的特征
  学到表示后，再在【训练图】上拟合一个线性分类头（linear probe），
  到测试图上评测 —— 这一步才用标签。

  所以本探针 = 两段式：
    ① 无监督：GraphMAE 在 exp3_2 上预训练（loss 只对被 mask 节点算 MSE）
    ② 有监督：在训练图 embedding 上训 Linear(32→3)，再在 exp3_3 上推理

────────────────────────────────────────────────────────────────
评测口径（与探针 ①/② 完全对齐，保证可比）
────────────────────────────────────────────────────────────────
  训练图：exp3_2（93 process + 229 file，密集进程树，无网络事件）
  测试图：exp3_3（5 process + 271 file + 3 network）
  报告：全节点准确率 + 训练见过的类(process/file)子集准确率。
        network 类训练时从未见过，属 OOD，不参与公平比较。
  特征：与探针 ①/② 完全相同的 16 维结构特征（无 ntype one-hot），
        标准化只用训练图统计量。
────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────
    .\.venv\Scripts\python.exe 04_graph_learning\graphmae_probe.py --seeds 0,1,2,3,4,5,6,7
    .\.venv\Scripts\python.exe 04_graph_learning\graphmae_probe.py --mask-ratio 0.75 --seeds 0,1,2,3
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "04_graph_learning"))
sys.path.insert(0, os.path.join(ROOT, "03_graph_builder"))
import build_graph as BG                                    # noqa: E402
from graphsage_probe import graph_to_pyg, report_matrix     # noqa: E402 复用建特征/评测

LOG_TRAIN = os.path.join(ROOT, "01_kellect_logs", "exp3_2", "log.json")
LOG_TEST = os.path.join(ROOT, "01_kellect_logs", "exp3_3", "read.json")
TRAIN_TAGS = ("parent_child.py",)
TEST_TAGS = BG.SCRIPT_TAGS
NTYPES = ["process", "file", "network"]


# ============================================================ GraphMAE 模型
class GraphMAE(nn.Module):
    """掩码图自编码器（论文结构：GIN 编码器 ×2 + GIN 解码器 ×1）。

    forward 返回重建输出 out 与被 mask 的掩码；embedding 由 encode 给出。
    """

    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.enc1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.enc2 = GINConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.dec = GINConv(nn.Sequential(
            nn.Linear(hidden, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)))

    def mask_and_forward(self, x, edge_index, mask_ratio, rng):
        """随机 mask 特征行 → 编码 → 解码 → 返回 (重建输出, 掩码)。"""
        mask = (torch.rand(x.size(0), generator=rng, device=x.device)
                < mask_ratio)
        x_in = x.clone()
        x_in[mask] = 0.0
        h = F.relu(self.enc1(x_in, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        out = self.dec(h, edge_index)
        return out, mask

    def encode(self, x, edge_index):
        """全图前向（不 mask），输出 32 维 embedding，供 linear probe 使用。"""
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        return h


# ============================================================ 无监督预训练
def train_mae(model, x, ei, epochs, lr, seed, mask_ratio):
    """只做 mask → 重建，完全不碰标签。loss = 被 mask 特征的 MSE。"""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out, mask = model.mask_and_forward(x, ei, mask_ratio, rng)
        loss = F.mse_loss(out[mask], x[mask])
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


# ============================================================ linear probe
def train_probe(model, x_tr, ei_tr, y_tr, epochs, lr, seed, hidden):
    """在训练图 embedding 上拟合线性分类头（这步才用标签）。"""
    torch.manual_seed(seed)
    with torch.no_grad():
        h_tr = model.encode(x_tr, ei_tr)
    probe = nn.Linear(hidden, len(NTYPES))
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        probe.train()
        opt.zero_grad()
        loss = lossf(probe(h_tr), y_tr)
        loss.backward()
        opt.step()
    return probe


# ============================================================ 评测（同探针 ①/②）
@torch.no_grad()
def eval_probe(model, probe, x_te, ei_te, y_te, seen_classes=(0, 1)):
    model.eval()
    h_te = model.encode(x_te, ei_te)
    logits = probe(h_te)
    pred = logits.argmax(-1)
    acc = float((pred == y_te).float().mean())
    cm = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(y_te.tolist(), pred.tolist()):
        cm[t, p] += 1
    mask = torch.isin(y_te, torch.tensor(seen_classes))
    acc_seen = (float((pred[mask] == y_te[mask]).float().mean())
                if bool(mask.any()) else float("nan"))
    return acc, acc_seen, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default=LOG_TRAIN)
    ap.add_argument("--test-log", default=LOG_TEST)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--seeds", default="0", help="逗号分隔，多种子看稳定性")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 72)
    print("1. 建图：exp3_2（训练）/ exp3_3（测试）  [方案 B，与探针 ①/② 同口径]")
    print("=" * 72)
    graphs = {}
    for name, path, tags in (("train", args.train_log, TRAIN_TAGS),
                             ("test", args.test_log, TEST_TAGS)):
        if not os.path.exists(path):
            print(f"[X] 找不到日志: {path}")
            raise SystemExit(1)
        records, bad = BG.load_records(path)
        keep = BG.focus_pids(records, tags=tags)
        records = [r for r in records if r["PID"] in keep]
        G = BG.build_graph(records, include_open=True)
        graphs[name] = G
        print(f"  {name}: {path.split(os.sep)[-1]}  tags={tags}  "
              f"{G.number_of_nodes()} 节点 {G.number_of_edges()} 边  "
              f"类型 {Counter(d['ntype'] for _n, d in G.nodes(data=True))}")

    print("\n" + "=" * 72)
    print("2. 转 PyG Data + 标准化（与探针 ①/② 相同 16 维结构特征）")
    print("=" * 72)
    x_tr, ei_tr, y_tr, _ = graph_to_pyg(graphs["train"])
    x_te, ei_te, y_te, _ = graph_to_pyg(graphs["test"])
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    print(f"  训练: {x_tr.shape[0]} 节点 标签 {Counter(y_tr.tolist())}")
    print(f"  测试: {x_te.shape[0]} 节点 标签 {Counter(y_te.tolist())}")

    print("\n" + "=" * 72)
    print(f"3. GraphMAE 无监督预训练（mask={args.mask_ratio}，不用标签）"
          f" + linear probe（用训练图标签）")
    print("=" * 72)
    in_dim = x_tr.shape[1]
    rows = []
    for seed in seeds:
        model = GraphMAE(in_dim, args.hidden)
        losses = train_mae(model, x_tr, ei_tr, args.epochs, args.lr, seed,
                           args.mask_ratio)
        probe = train_probe(model, x_tr, ei_tr, y_tr, args.probe_epochs,
                            args.lr, seed, args.hidden)
        acc, acc_seen, cm = eval_probe(model, probe, x_te, ei_te, y_te)
        rows.append((seed, losses[0], losses[-1], acc, acc_seen))
        print(f"  seed {seed:>2}: 重建 loss {losses[0]:.4f}→{losses[-1]:.4f}  "
              f"全节点 {acc:.4f}  见过类子集 {acc_seen:.4f}")
        if seed == seeds[0]:
            report_matrix("GraphMAE + probe", acc, acc_seen, cm)

    print("\n" + "=" * 72)
    print("4. 多种子稳定性汇总")
    print("=" * 72)
    accs = np.array([r[3] for r in rows])
    accs_seen = np.array([r[4] for r in rows])
    print(f"  种子数 {len(seeds)}")
    print(f"  全节点 acc:       mean {accs.mean():.4f}  std {accs.std():.4f}  "
          f"min {accs.min():.4f}  max {accs.max():.4f}")
    print(f"  见过类子集 acc:   mean {accs_seen.mean():.4f}  std {accs_seen.std():.4f}  "
          f"min {accs_seen.min():.4f}  max {accs_seen.max():.4f}")
    print(f"  逐种子见过类子集: {', '.join(f'{r[4]:.4f}' for r in rows)}")
    print("""
  判读要点（诚实边界）：
  · GraphMAE 预训练不碰标签，学到的是"结构→特征"的重建能力；
    linear probe 之后才能评测分类。与探针 ①/② 端到端监督相比，
    这是【任务难度不同】的两条路，acc 差距不能简单归因于模型好坏。
  · 警惕 GAT 式"双峰"：若某些种子 acc 塌缩，说明无监督预训练
    落入坏盆地（表示退化成常数/同质化），需如实记录。
  · network 类(2)训练未见过，OOD，公平对比只看 process/file 子集。
    """)
    # 保存逐种子明细，供归档
    out = os.path.join(ROOT, "04_graph_learning", "graphmae_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"mask_ratio={args.mask_ratio} epochs={args.epochs} "
                f"probe_epochs={args.probe_epochs} lr={args.lr} "
                f"hidden={args.hidden}\n")
        for seed, l0, l1, acc, acc_seen in rows:
            f.write(f"seed {seed}: loss {l0:.4f}->{l1:.4f}  acc {acc:.4f}  "
                    f"acc_seen {acc_seen:.4f}\n")
    print(f"\n  逐种子明细已存: {out}")


if __name__ == "__main__":
    main()
