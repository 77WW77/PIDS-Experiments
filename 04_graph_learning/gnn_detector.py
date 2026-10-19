"""
阶段 5 预热：把 4.3 建好的 provenance graph 喂给一个最简单的 GNN，跑通
"图 → 表征 → 异常分" 这条管道。

────────────────────────────────────────────────────────────────
这次要回答的问题只有一个
────────────────────────────────────────────────────────────────
  "读了 secret.txt 又往外连网"的那个进程，能不能在无监督异常分里排得靠前？

────────────────────────────────────────────────────────────────
必须提前说清的三个诚实边界（跑完看结果时要记着）
────────────────────────────────────────────────────────────────
  1) 没有攻击标签。这份日志里只有两台自己写的小程序在自娱自乐，
     不存在 ground truth 的"攻击进程"。因此【算不出 Recall / F1 / AUC】，
     这里只能做 sanity check：看分数有没有落在"人看出来可疑"的地方。
  2) 单图训练 + 单图打分。训练集和测试集是同一张图，异常样本也在
     训练集里，模型完全可能把"读 secret.txt"也当成正常结构学会，
     于是它反而不显眼。这是单图的小样本固有局限，不是代码 bug。
  3) 这类 GAE（内积解码器）在无特征图上的主要学习目标是"度数/连通结构"。
     所以结果里附了一条【度数基线】做对照：如果异常分和度数基线的
     Spearman 相关系数接近 1.0，说明 GNN 并没有比"数一数邻居个数"
     多说任何东西 —— 那就要如实承认，别包装成"深度模型的效果"。

────────────────────────────────────────────────────────────────
模型（刻意选最小的那一档）
────────────────────────────────────────────────────────────────
  编码器  GCNConv(19, 32) → ReLU → GCNConv(32, 16)
  解码器  内积  sigmoid(z_u · z_v)      ← 逐边二分类
  损失    BCEWithLogitsLoss，正边 : 负边 = 1 : 1
  异常分  节点分 = 它与所有正邻居的边被"重构失败"的平均程度 = mean(1 - p)
          （正邻居的定义见 build 部分：图里真实存在边的节点对）

  编码用了有向边（保留方向信息），打分用了对称化后的节点对。
  这两者不矛盾：4-C 第 5 问禁止的是"把溯源图的知识表示改成无向图"，
  而这里只是解码器要求对称，方向信息已经通过 in/out 度特征留在 x 里。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────
    .\\.venv\\Scripts\\python.exe 04_graph_learning\\gnn_detector.py
    .\\.venv\\Scripts\\python.exe 04_graph_learning\\gnn_detector.py --epochs 300
    .\\.venv\\Scripts\\python.exe 04_graph_learning\\gnn_detector.py --all
    .\\.venv\\Scripts\\python.exe 04_graph_learning\\gnn_detector.py --no-open
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_graph_builder"))
import build_graph as BG          # noqa: E402  复用 4.3 的建图逻辑，不重写一遍

# 边型顺序必须固定：它决定 x 里那 11 维的语义，换了顺序特征就没意义了
ETYPES = ["Create", "Read", "Write", "CreateFile", "Open", "Delete",
          "Connect", "Accept", "Send", "Recv", "Disconnect"]
NTYPES = ["process", "file", "network"]


# ============================================================ 图 → 特征矩阵
def graph_to_arrays(G):
    """把 MultiDiGraph 摊平成一个无向点对集合 + 每节点的特征向量。

    返回：
      nodes  : 节点列表（索引 = 在 PyG 里的 id）
      x      : (N, 19) float32 特征
      dir_ei : (2, E_unique) 去重后的【有向】边，供 GCN 传播用
      pos    : 正样本点对集合（无向、去重），供解码器算 loss / 打分
      info   : 每个节点的原始统计，用于最后报告
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # ——— 逐条边统计：度数用"边的条数"（multi），不重复计数才是邻居数 ———
    out_cnt, in_cnt = Counter(), Counter()
    out_nb, in_nb = defaultdict(set), defaultdict(set)
    out_etype = defaultdict(Counter)
    dir_pairs = set()
    for u, v, _k, d in G.edges(keys=True, data=True):
        out_cnt[u] += 1
        in_cnt[v] += 1
        out_nb[u].add(v)
        in_nb[v].add(u)
        out_etype[u][d["etype"]] += 1
        dir_pairs.add((idx[u], idx[v]))

    # 正样本 = "这两点之间有边"，不管它在原图里是 u→v 还是 v→u，都算一对
    pos = set()
    for a, b in dir_pairs:
        pos.add((a, b) if a < b else (b, a))

    # ——— 19 维特征 ———
    x = np.zeros((n, 19), dtype=np.float32)
    for node, i in idx.items():
        d = G.nodes[node]
        base = 0
        # 1) 节点类型 one-hot (3)
        if d["ntype"] in NTYPES:
            x[i, NTYPES.index(d["ntype"])] = 1.0
        base = 3
        # 2) 出入度 + 出入邻居数（都做 log1p，度数长尾太狠）(4)
        x[i, base + 0] = np.log1p(out_cnt[node])
        x[i, base + 1] = np.log1p(in_cnt[node])
        x[i, base + 2] = np.log1p(len(out_nb[node]))
        x[i, base + 3] = np.log1p(len(in_nb[node]))
        base += 4
        # 3) 出边事件类型的归一化分布 (11)
        tot = sum(out_etype[node].values())
        if tot:
            for e, c in out_etype[node].items():
                if e in ETYPES:
                    x[i, base + ETYPES.index(e)] = c / tot
        base += len(ETYPES)
        # 4) 是否走了兜底身份（窗口开始前就活着、没有 ProcessStart 的进程）(1)
        if d.get("ntype") == "process" and not d.get("resolved", True):
            x[i, base] = 1.0

    # z-score 标准化：不同量纲的列放到同一尺度，否则 GCN 会被大数值列带跑
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    x = (x - mu) / sd

    info = {
        "out_cnt": out_cnt, "in_cnt": in_cnt,
        "deg": {node: len(out_nb[node]) + len(in_nb[node]) for node in nodes},
    }
    return nodes, idx, x, dir_pairs, pos, info


# ============================================================ 模型
class GAE(torch.nn.Module):
    """图自编码器：GCN 编码 + 内积解码。"""

    def __init__(self, in_dim, hidden=32, out_dim=16):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hidden)
        self.gc2 = GCNConv(hidden, out_dim)

    def encode(self, x, edge_index):
        h = F.relu(self.gc1(x, edge_index))
        return self.gc2(h, edge_index)

    def decode(self, z, edge_index):
        """逐边打分：两个端点表征的内积（越大越像"真有边"）。"""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(-1)


def sample_negatives(pos_set, n_nodes, n_samples, rng):
    """在"本来没有边"的点对里随机取负样本，和正样本 1:1。

    采样时跳过正样本对 —— 否则会拿真边当负例教模型，直接把学到的东西污染掉。
    """
    neg = []
    while len(neg) < n_samples:
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in pos_set:
            continue
        neg.append(key)
    return neg


def train_gae(model, x, edge_index, pos_set, epochs, lr, rng, log_every=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos_list = list(pos_set)

    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(x, edge_index)

        pi = torch.tensor(pos_list, dtype=torch.long).t()
        pos_logit = model.decode(z, pi)

        neg = sample_negatives(pos_set, x.size(0), len(pos_list), rng)
        ni = torch.tensor(neg, dtype=torch.long).t()
        neg_logit = model.decode(z, ni)

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        opt.step()
        losses.append(float(loss))

        if ep % log_every == 0 or ep == 1:
            print(f"    epoch {ep:>4}   loss = {float(loss):.4f}")
    return losses


@torch.no_grad()
def score_nodes(model, x, edge_index, pos_set, n_nodes):
    """节点异常分 = 它的正邻居里"边没被重构出来"的平均程度。

    等价说法：模型认为"这些边不该存在"的程度。一个节点如果它的邻居关系
    在训练中始终没被压进低维表征，说明它的连接模式偏离主体分布。
    """
    model.eval()
    z = model.encode(x, edge_index)

    partners = defaultdict(list)
    for a, b in pos_set:
        partners[a].append(b)
        partners[b].append(a)

    scores = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        ps = partners.get(i)
        if not ps:
            continue          # 孤立点（没有边）不给分，见下面的统计
        zi = z[i].unsqueeze(0)
        zj = z[torch.tensor(ps, dtype=torch.long)]
        prob = torch.sigmoid((zi * zj).sum(-1))
        scores[i] = float((1.0 - prob).mean())
    return scores, partners


# ============================================================ 报告
def label_of(G, node):
    d = G.nodes[node]
    if d["ntype"] == "process":
        extra = "" if d.get("resolved", True) else "  [兜底身份]"
        return f"process  {d['label']} (PID={d['pid']}){extra}"
    return f"{d['ntype']:<8} {d['label']}"


def fit_and_score(x_t, ei, pos_set, n_nodes, seed, hidden, epochs, lr, log_every=50):
    """一次完整的训练 + 打分。抽成函数是为了能换种子重复跑（见 --repeat）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GAE(x_t.shape[1], hidden=hidden)
    losses = train_gae(model, x_t, ei, pos_set, epochs, lr, rng, log_every=log_every)
    scores, partners = score_nodes(model, x_t, ei, pos_set, n_nodes)
    return scores, partners, losses


def collect_targets(G):
    """挑出"人眼认为可疑"的节点，作为 sanity check 的靶子。

    注意这是【我们自己根据实验设计写死的靶子】，不是攻击标签。
    """
    out = []
    secret = next((n for n, d in G.nodes(data=True)
                   if d["ntype"] == "file" and "secret" in (d.get("path") or "")), None)
    if secret is not None:
        out.append(("secret.txt 文件节点", secret))
        # MultiDiGraph 的 in_edges 每条平行边返回一次 → 必须去重
        for p in sorted({u for u, _v, d in G.in_edges(secret, data=True)
                         if d["etype"] == "Read"}):
            out.append((f"读 secret 的进程 {G.nodes[p]['label']}(PID={G.nodes[p]['pid']})", p))
    for p in sorted({u for u, _v, d in G.edges(data=True)
                     if d["etype"] in ("Connect", "Send", "Recv")
                     and G.nodes[u]["ntype"] == "process"}):
        out.append((f"有网络动作的进程 {G.nodes[p]['label']}(PID={G.nodes[p]['pid']})", p))
    return out


def stability_report(args, nodes, idx, x_t, ei, pos_set, targets):
    """换 N 个随机种子重训，看目标节点的名次会不会跳。

    为什么必须做这件事：单图 + 无标签时，"Top-1 是 secret.txt"这种结果
    无法区分两种情况 —— 模型真学到了异常，还是这次随机初始化恰好撞上。
    唯一低成本的判别办法就是重复抽样看名次稳不稳。
    """
    n = len(nodes)
    print("=" * 72)
    print(f"稳定性检查：换 {args.repeat} 个随机种子重训（epochs={args.epochs}），看名次怎么跳")
    print("=" * 72)

    ranks = defaultdict(list)
    for s in range(args.repeat):
        scores, _p, losses = fit_and_score(x_t, ei, pos_set, n, s,
                                           args.hidden, args.epochs, args.lr,
                                           log_every=10 ** 9)
        order = sorted(range(n), key=lambda i: -scores[i])
        rk = {nodes[i]: r + 1 for r, i in enumerate(order)}
        for name, node in targets:
            ranks[name].append(rk[node])
        print(f"  seed {s}: loss {losses[0]:6.2f} → {losses[-1]:.4f}")

    print(f"\n  节点总数 {n}")
    print(f"  {'目标节点':<46}{'min/中位/max':<16}每次的名次")
    print("  " + "-" * 100)
    for name, _node in targets:
        rs = ranks[name]
        print(f"  {name[:44]:<46}{min(rs):>3} / {int(np.median(rs)):>3} / {max(rs):>3}     {rs}")

    print("\n  判读：")
    print("    · 名次始终靠前且稳定  → 分数里可能有真实结构信号（但也可能只是度数这类平凡特征）")
    print("    · 名次在多次重训间大幅跳动 → 这份分数主要来自随机初始化，不能当检测信号用")
    print("    · 无论哪种，都【不能】推出检测率：这里没有攻击标签，靶子是我们自己钉的")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=BG.DEFAULT_LOG)
    ap.add_argument("--all", action="store_true", help="不过滤，把整份日志都建进去")
    ap.add_argument("--no-open", action="store_true", help="丢掉 FileIOCreate")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--repeat", type=int, default=1,
                    help=">1 时改为换种子重训 N 次，只报目标节点的名次稳定性")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------- 1. 建图
    print("=" * 72)
    print("1. 建图（复用 03_graph_builder/build_graph.py）")
    print("=" * 72)
    if not os.path.exists(args.log):
        print(f"[X] 找不到日志: {args.log}")
        raise SystemExit(1)
    records, bad = BG.load_records(args.log)
    print(f"读到 {len(records)} 条事件（坏行 {bad}）")
    if not args.all:
        keep = BG.focus_pids(records)
        records = [r for r in records if r["PID"] in keep]
        print(f"过滤到实验进程子树 {sorted(keep)} 后剩 {len(records)} 条")
    G = BG.build_graph(records, include_open=not args.no_open)
    BG.report(G)

    # ---------------------------------------------------------- 2. 转特征
    print("\n" + "=" * 72)
    print("2. 图 → 张量")
    print("=" * 72)
    nodes, idx, x, dir_pairs, pos_set, info = graph_to_arrays(G)
    print(f"  节点 {len(nodes)}   有向边(去重) {len(dir_pairs)}   正样本点对 {len(pos_set)}")
    print(f"  特征维度 {x.shape[1]} = 类型 one-hot 3 + 出入度/邻居 4 + 出边类型分布 11 + 兜底标记 1")

    x_t = torch.tensor(x, dtype=torch.float32)
    ei = torch.tensor(sorted(dir_pairs), dtype=torch.long).t().contiguous()

    # 人眼钉的靶子（不是攻击标签）：secret.txt 文件节点、读它的进程、有网络动作的进程
    targets = collect_targets(G)

    # ---------------------------------------------------------- 3. 训练
    if args.repeat > 1:
        stability_report(args, nodes, idx, x_t, ei, pos_set, targets)
        return

    print("\n" + "=" * 72)
    print(f"3. 训练 GAE（epochs={args.epochs}  hidden={args.hidden}  lr={args.lr}  seed={args.seed}）")
    print("=" * 72)
    model = GAE(x.shape[1], hidden=args.hidden)
    losses = train_gae(model, x_t, ei, pos_set, args.epochs, args.lr, rng)
    print(f"    loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ---------------------------------------------------------- 4. 打分
    scores, partners = score_nodes(model, x_t, ei, pos_set, len(nodes))
    iso = [nodes[i] for i in range(len(nodes)) if not partners.get(i)]

    print("\n" + "=" * 72)
    print(f"4. 节点异常分 Top-{args.top}（分越高 = 越不像图里的主体模式）")
    print("=" * 72)
    order = sorted(range(len(nodes)), key=lambda i: -scores[i])
    rank = {nodes[i]: r + 1 for r, i in enumerate(order)}
    for r, i in enumerate(order[:args.top], 1):
        node = nodes[i]
        print(f"  {r:>3}. {scores[i]:.4f}  {label_of(G, node)}"
              f"   度={info['deg'][node]}  出边={info['out_cnt'][node]} 入边={info['in_cnt'][node]}")
    if iso:
        print(f"  （另有 {len(iso)} 个孤立节点没有边，不参与打分）")

    # ---------------------------------------------------------- 5. 对照检查
    print("\n" + "=" * 72)
    print("5. Sanity check：人眼看可疑的节点，排在第几？")
    print("=" * 72)

    secret = next((n for n, d in G.nodes(data=True)
                   if d["ntype"] == "file" and "secret" in (d.get("path") or "")), None)
    if secret is None:
        print("  图里没有 secret.txt 节点（fixed 模式本就如此）→ 无法做这一项对照")
    else:
        # 注意：MultiDiGraph 的 in_edges 会为每条平行边各返回一次，
        # 6 条 Read 会把同一个进程打印 6 遍 → 必须 set 去重
        readers = sorted({u for u, _v, d in G.in_edges(secret, data=True)
                          if d["etype"] == "Read"})
        print(f"  secret.txt 节点本身：分 {scores[idx[secret]]:.4f}  排名 {rank[secret]}/{len(nodes)}")

        print(f"\n  读过 secret.txt 的进程（共 {len(readers)} 个）:")
        for p in sorted(readers, key=lambda n: -scores[idx[n]]):
            has_net = any(d["etype"] in ("Connect", "Send", "Recv")
                          for _u, _v, d in G.out_edges(p, data=True))
            flag = "  ← 既读文件又连网" if has_net else ""
            print(f"    分 {scores[idx[p]]:.4f}  排名 {rank[p]:>3}/{len(nodes)}  "
                  f"{label_of(G, p)}{flag}")

    # 兜底身份的进程：身份不确定，看它们的分数分布有没有整体偏高
    unres = [n for n, d in G.nodes(data=True)
             if d["ntype"] == "process" and not d.get("resolved", True)]
    if unres:
        us = np.array([scores[idx[n]] for n in unres])
        all_p = np.array([scores[idx[n]] for n, d in G.nodes(data=True) if d["ntype"] == "process"])
        print(f"\n  兜底身份进程 {len(unres)} 个：分均值 {us.mean():.4f}"
              f"（全部进程均值 {all_p.mean():.4f}，进程数 {len(all_p)}）")
        for n in sorted(unres, key=lambda n: -scores[idx[n]])[:5]:
            print(f"    分 {scores[idx[n]]:.4f}  排名 {rank[n]:>3}  {label_of(G, n)}")

    # ---------------------------------------------------------- 6. 能不能比度数说得更多
    print("\n" + "=" * 72)
    print("6. 关键对照：GNN 分 vs 度数基线")
    print("=" * 72)
    active = [i for i in range(len(nodes)) if partners.get(i)]
    gn = np.array([scores[i] for i in active])
    bl = np.array([info["deg"][nodes[i]] for i in active], dtype=float)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(gn, bl)
        print(f"  Spearman(GNN 分, 度数) = {rho:+.4f}   (p = {pval:.2e}, n = {len(active)})")
        if abs(rho) > 0.85:
            print("  → 接近 ±1：这个 GNN 的排序基本等价于'数邻居个数'，")
            print("    深度部分没有额外贡献。别把它的名次当'模型检测能力'来讲。")
        elif abs(rho) > 0.5:
            print("  → 中高度相关：GNN 有一部分结构信息，但很大成分仍是度数。")
        else:
            print("  → 相关性不高：GNN 分确实带了度数之外的信息（也可能只是噪声）。")
    except ImportError:
        print("  （未安装 scipy，跳过相关性计算）")

    # 用度数基线自己排一次，看两者 Top-15 重合多少
    by_deg = sorted(active, key=lambda i: -info["deg"][nodes[i]])[:args.top]
    overlap = len(set(order[:args.top]) & set(by_deg))
    print(f"  Top-{args.top} 重合度：GNN 与度数基线共有 {overlap} 个节点")

    # ---------------------------------------------------------- 7. 结论
    print("\n" + "=" * 72)
    print("7. 本次能说 / 不能说的话")
    print("=" * 72)
    print("  能说：")
    print("    · 管道跑通了：MultiDiGraph → 19 维特征 → GCN 编码 → 内积解码 → 节点异常分")
    print("    · 图结构确实能变成可训练的数值，且 loss 单调下降（见第 3 节）")
    print("    · 分数集中在哪类节点上 —— 见第 4 节，自己判断是否符合直觉")
    print("  必须补做的一步：")
    print("    · 用 --repeat 10 换种子重训，看第 5 节那些靶子的名次稳不稳。")
    print("      单图无标签时，'Top-1 正好是 secret.txt' 无法区分'真学到'和'初始化撞上'，")
    print("      名次抖动幅度就是这一点的直接证据。")
    print("  不能说：")
    print("    · 任何'检测率/准确率'。没有攻击标签，算不出 Recall/F1/AUC")
    print("    · '模型找到了攻击'。这颗 secret.txt 是自编实验，不是真实攻击")
    print("    · 'GNN 比传统方法强'。单图、单次运行、无对照基线，说不了这件事")
    print("  下一步才有意义的做法（阶段 5 正式内容）：")
    print("    · 找带 ATT&CK 标签的数据集（Kellect4APT / DARPA 溯源数据集）分训练/测试")
    print("    · 换成能处理异构性/时序的模型（HAN、TGN），并加 IsolationForest 等基线对照")


if __name__ == "__main__":
    main()