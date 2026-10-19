"""
阶段 5 探针 ①：GraphSAGE / GAT / GCN 的跨图泛化（inductive vs transductive）

────────────────────────────────────────────────────────────────
实验设计（方案 B：跨场景泛化）
────────────────────────────────────────────────────────────────
  训练图：exp3_2 父子进程场景（parent_child.py）—— 322 节点
          （93 process + 229 file），进程树密集 + 大量文件并行边，
          完全没有网络事件。
  测试图：exp3_3 网络场景（read.json）—— 279 节点
          （5 process + 271 file + 3 network），稀疏文件操作 + 本地 HTTP。

  两图关系（实测）：节点数接近（322 vs 279），但结构与密度完全不同；
  且测试图的 network 类(2)在训练图上【从未出现过】。
  这是比方案 A（read vs fixed 仅差一处文件层）强得多的跨图泛化考验。

  注：exp3_1 日志实测不含被测程序（file_ops.py 的 ProcessStart 缺失，
  focus_pids 筛出 0 个 PID，图为空），无法建出目标图，故训练图改用
  exp3_2 —— 与 exp3_3 同样是"跨场景"差异，符合方案 B 的设计意图。

────────────────────────────────────────────────────────────────
任务与评测
────────────────────────────────────────────────────────────────
  任务：节点类型分类（process / file / network）。
  流程：在 exp3_2 图上训练分类器 → 直接在 exp3_3 图上推理打分。

  诚实边界：
  1) 特征【不含】ntype one-hot，只给 16 维结构特征，逼模型从邻居结构
     推断节点类型。
  2) 标准化只用训练图的 mu/sd，测试图不得借用。
  3) 测试图有 3 个 network 节点，训练图完全没有该类样本 —— 分类头虽
     有 3 个输出但对类 2 从未训练过，network 行必然差，那是"未见过
     的类"的分布外测试，不是 bug。因此报告同时给【全节点准确率】和
     【训练见过的类(process/file)子集准确率】，后者才是两模型的公平对比。
────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────
    .\.venv\Scripts\python.exe 04_graph_learning\graphsage_probe.py
    .\.venv\Scripts\python.exe 04_graph_learning\graphsage_probe.py --epochs 300
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_graph_builder"))
import build_graph as BG          # noqa: E402  复用 4.3 建图

LOG_TRAIN = os.path.join(ROOT, "01_kellect_logs", "exp3_2", "log.json")
LOG_TEST = os.path.join(ROOT, "01_kellect_logs", "exp3_3", "read.json")
# 每个日志各自的目标程序名（focus_pids 的 tags）——方案 B 跨实验，必须分开指定。
# exp3_2 是父子进程场景（parent_child.py）；exp3_3 是网络场景（http 客户端/服务端）。
TRAIN_TAGS = ("parent_child.py",)
TEST_TAGS = BG.SCRIPT_TAGS          # ("http_client.py", "local_http_server.py")

ETYPES = ["Create", "Read", "Write", "CreateFile", "Open", "Delete",
          "Connect", "Accept", "Send", "Recv", "Disconnect"]
NTYPES = ["process", "file", "network"]


# ============================================================ 图 → PyG Data
def graph_to_pyg(G):
    """MultiDiGraph → (x, edge_index, y, nodes)。

    x：16 维纯结构特征（不含 ntype one-hot）——
        出入度 log1p(2) + 出入邻居数 log1p(2) + 出边类型分布(11) + 兜底标记(1)
    edge_index：无向化（双向）去重后的 (2, E) long 张量
    y：节点类型标签 0=process 1=file 2=network（两张图共享的标签）
    nodes：节点列表，索引即 PyG 里的 id，供报告打印
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    out_cnt, in_cnt = Counter(), Counter()
    out_nb, in_nb = defaultdict(set), defaultdict(set)
    out_etype = defaultdict(Counter)
    undir = set()
    for u, v, _k, d in G.edges(keys=True, data=True):
        out_cnt[u] += 1
        in_cnt[v] += 1
        out_nb[u].add(v)
        in_nb[v].add(u)
        out_etype[u][d["etype"]] += 1
        a, b = idx[u], idx[v]
        undir.add((a, b) if a < b else (b, a))   # 无向化：去重时排序

    x = np.zeros((n, 16), dtype=np.float32)
    for node, i in idx.items():
        d = G.nodes[node]
        # 出入度 + 出入邻居数（log1p 压长尾）
        x[i, 0] = np.log1p(out_cnt[node])
        x[i, 1] = np.log1p(in_cnt[node])
        x[i, 2] = np.log1p(len(out_nb[node]))
        x[i, 3] = np.log1p(len(in_nb[node]))
        # 出边事件类型归一化分布
        tot = sum(out_etype[node].values())
        if tot:
            for e, c in out_etype[node].items():
                if e in ETYPES:
                    x[i, 4 + ETYPES.index(e)] = c / tot
        # 兜底身份标记（进程窗口前就活着、无 ProcessStart）
        if d.get("ntype") == "process" and not d.get("resolved", True):
            x[i, 15] = 1.0

    # 无向 edge_index：每条边存 (a,b) 和 (b,a) 两个方向
    ei = []
    for a, b in undir:
        ei.append((a, b))
        ei.append((b, a))
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous()

    # y 与 x 共用同一个 nodes 列表与 idx，保证索引顺序严格一致
    y = torch.tensor([NTYPES.index(G.nodes[n]["ntype"]) for n in nodes],
                     dtype=torch.long)

    return torch.tensor(x, dtype=torch.float32), ei, y, nodes


# ============================================================ 模型（两版同构）
class GraphSAGE_Clf(torch.nn.Module):
    """两层 SAGEConv + 线性分类头（node classification）。"""

    def __init__(self, in_dim, hidden=32, n_classes=3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.clf = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.clf(h)


class GCN_Clf(torch.nn.Module):
    """与 GraphSAGE 同构的两层 GCNConv + 线性分类头。"""

    def __init__(self, in_dim, hidden=32, n_classes=3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.clf = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.clf(h)


class GAT_Clf(torch.nn.Module):
    """与 GraphSAGE/GCN 同构的两层 GATConv + 线性分类头。

    第一层多头(4 头、每头 8 维、拼接成 32)，第二层单头 —— GAT 论文的标准
    做法，hidden 总量与另外两个模型对齐。
    """

    def __init__(self, in_dim, hidden=32, n_classes=3, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False)
        self.clf = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.clf(h)


# ============================================================ 训练（仅在训练图上）
def train_clf(model, x_tr, ei_tr, y_tr, epochs, lr, seed, log_every=50):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    lossf = torch.nn.CrossEntropyLoss()
    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x_tr, ei_tr)
        loss = lossf(out, y_tr)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if ep % log_every == 0 or ep == 1:
            print(f"        epoch {ep:>4}  loss = {float(loss):.4f}")
    return losses


# ============================================================ 评测（在测试图上）
@torch.no_grad()
def eval_clf(model, x_te, ei_te, y_te, seen_classes=(0, 1)):
    """推理时【不加】任何自环/无向处理 —— 测试图结构独立给模型。

    seen_classes：训练时见过的类。方案 B 里测试图的 network 类(2)在训练
    图上从未出现，把它从公平对比中排除，另算"见过类的子集准确率"。
    """
    model.eval()
    logits = model(x_te, ei_te)
    pred = logits.argmax(-1)
    acc = float((pred == y_te).float().mean())
    cm = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(y_te.tolist(), pred.tolist()):
        cm[t, p] += 1
    mask = torch.isin(y_te, torch.tensor(seen_classes))
    acc_seen = (float((pred[mask] == y_te[mask]).float().mean())
                if bool(mask.any()) else float("nan"))
    return acc, acc_seen, cm


def report_matrix(name, acc, acc_seen, cm):
    print(f"\n  {name}  测试图全节点准确率 = {acc:.4f}")
    print(f"      训练见过的类(process/file)子集准确率 = {acc_seen:.4f}")
    print(f"    混淆矩阵（行=真实 列=预测，0=process 1=file 2=network）：")
    for i, rname in enumerate(NTYPES):
        row = " ".join(f"{v:>4}" for v in cm[i].tolist())
        print(f"      {rname:<8}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default=LOG_TRAIN)
    ap.add_argument("--test-log", default=LOG_TEST)
    ap.add_argument("--train-tags", default=",".join(TRAIN_TAGS))
    ap.add_argument("--test-tags", default=",".join(TEST_TAGS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_tags = tuple(t.strip() for t in args.train_tags.split(",") if t.strip())
    test_tags = tuple(t.strip() for t in args.test_tags.split(",") if t.strip())

    print("=" * 72)
    print("1. 建图：exp3_2 进程树图（训练）/ exp3_3 网络图（测试）  [方案 B]")
    print("=" * 72)
    for tag, path, tags in (("train(训练)", args.train_log, train_tags),
                            ("test(测试)", args.test_log, test_tags)):
        if not os.path.exists(path):
            print(f"[X] 找不到日志: {path}")
            raise SystemExit(1)
        records, bad = BG.load_records(path)
        keep = BG.focus_pids(records, tags=tags)
        records = [r for r in records if r["PID"] in keep]
        G = BG.build_graph(records, include_open=True)
        print(f"  {tag}: {path.split(os.sep)[-1]}  tags={tags}  保留 PID={len(keep)}"
              f"  {G.number_of_nodes()} 节点 {G.number_of_edges()} 边")
        print(f"      节点类型: {Counter(d['ntype'] for _n, d in G.nodes(data=True))}")
        if tag.startswith("train"):
            G_tr = G
        else:
            G_te = G

    print("\n" + "=" * 72)
    print("2. 转 PyG Data（结构特征 16 维，无 ntype one-hot）")
    print("=" * 72)
    x_tr, ei_tr, y_tr, nodes_tr = graph_to_pyg(G_tr)
    x_te, ei_te, y_te, nodes_te = graph_to_pyg(G_te)
    print(f"  训练图: {x_tr.shape[0]} 节点, 无向边 {ei_tr.shape[1] // 2}, 标签分布 "
          f"{Counter(y_tr.tolist())}")
    print(f"  测试图: {x_te.shape[0]} 节点, 无向边 {ei_te.shape[1] // 2}, 标签分布 "
          f"{Counter(y_te.tolist())}")

    # 标准化：只用训练图统计量；测试图不得借用（避免测试集信息泄漏）
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd
    print("  标准化: mu/sd 来自训练图，测试图套用同一组（无泄漏）")

    print("\n" + "=" * 72)
    print(f"3. 在训练图上分别训练两类模型（epochs={args.epochs}  "
          f"hidden={args.hidden}  lr={args.lr}  seed={args.seed}）")
    print("=" * 72)
    in_dim = x_tr.shape[1]

    models = {
        "GraphSAGE (SAGEConv)": GraphSAGE_Clf(in_dim, args.hidden),
        "GAT (GATConv)": GAT_Clf(in_dim, args.hidden),
        "GCN (GCNConv)": GCN_Clf(in_dim, args.hidden),
    }
    results = {}
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        losses = train_clf(model, x_tr, ei_tr, y_tr, args.epochs, args.lr,
                           args.seed)
        print(f"      loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        acc, acc_seen, cm = eval_clf(model, x_te, ei_te, y_te)
        report_matrix(name, acc, acc_seen, cm)
        results[name] = (acc, acc_seen, cm)

    print("\n" + "=" * 72)
    print("4. 对比与判读")
    print("=" * 72)
    for name, (acc, acc_seen, _cm) in results.items():
        print(f"  {name}: 全节点 {acc:.4f}   见过类子集 {acc_seen:.4f}")
    gcn = results.get("GCN (GCNConv)")
    if gcn is not None:
        for name, (acc, acc_seen, _cm) in results.items():
            if name == "GCN (GCNConv)":
                continue
            print(f"  vs GCN 差值(全节点): {acc - gcn[0]:+.4f}   "
                  f"差值(见过类子集): {acc_seen - gcn[1]:+.4f}")
    print("""
  判读要点（诚实边界）：
  · 跨图泛化考验比方案 A 强得多：训练图是密集进程树（无网络事件），
    测试图是稀疏文件操作 + 网络层，且 network 类(2)训练时从未见过。
  · network 行两模型必然都差 —— 那是"未见过的类"的 OOD，不是比较项。
    公平对比看"见过类子集(process/file)"准确率。
  · 若 GraphSAGE 在子集上显著优于 GCN → inductive 优势在结构差异足够
    大时可见；若仍打平，说明该任务下两者等价（如实记录）。
    """)


if __name__ == "__main__":
    main()
