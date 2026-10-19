"""
阶段 4.3：用 NetworkX.MultiDiGraph 把 Kellect 日志建成 provenance graph

对应档案 PDF 第 6 页 4.3：
  · 每个节点至少保留：entity id、entity type、主要属性
  · 每条边至少保留：event type、timestamp、event id、raw log index
  · 必须用 MultiDiGraph：同一对节点之间会多次 Read/Write，简单图会覆盖

────────────────────────────────────────────────────────────────
建图规则（我们自己定的，按原文"必须记录"的要求写在这里）
────────────────────────────────────────────────────────────────
★ 方向统一为【控制流 control flow】：每条边 = 动作主体(进程) → 被操作的对象
      Create      Parent → Child
      Read        P → File
      Write       P → File
      CreateFile  P → File
      Open        P → File          （FileIOCreate = 打开文件对象，≠ 新建文件）
      Delete      P → File
      Connect/Accept/Send/Recv/Disconnect   P → Endpoint

  理由（4.2 得出的结论）：Connect 只能表达"谁连了谁"。若 Read 用信息流(F→P)、
  Connect 用控制流，图上会凭空出现 F → P → Endpoint 的通路，看起来像
  "文件内容被发出去了" —— 而这是错的。统一方向后，同一进程的几条边互不接力，
  图只肯说它真知道的事。

★ 实体身份（Entity Resolution）
  进程  ("process", pid, start_ts)  start_ts = 该进程 ProcessStart 的 TimeStamp
        兜底 ("process", pid, None)  窗口开始前就活着、日志里没有它的 ProcessStart
  文件  ("file", 规范化路径)         所有文件事件都能拿到路径
        附带把 FileKey 累积成节点属性（FileIOCreate 类没有 FileKey，只有 OpenPath）
  端点  ("network", 排序后的 addr:port, addr:port)
        故意排序：让一条 TCP 连接在"客户端视角"和"服务端视角"下落到同一个节点，
        否则图会被切成互不相连的两半（实测：客户端 57954→18080，服务端 18080→57954）

⚠ 本方案已知的代价（留给 4.4「图构建的信息损失」讨论）
  1) 文件用路径当主键 → 同一路径"删除后重建"的两个文件会被并成一个节点
  2) FileIOCreate/FileIOCleanup/FileIOClose 等句柄级事件被丢弃
  3) 90% 的 PID 没有 ProcessStart（--all 时）→ 身份是兜底的，不是证实的
────────────────────────────────────────────────────────────────

用法：
    .\\.venv\\Scripts\\python.exe 03_graph_builder\\build_graph.py
    .\\.venv\\Scripts\\python.exe 03_graph_builder\\build_graph.py 01_kellect_logs\\exp3_3\\fixed.json
    .\\.venv\\Scripts\\python.exe 03_graph_builder\\build_graph.py --all      # 不过滤，看噪声
"""

import argparse
import json
import os
import socket
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import networkx as nx

CST = timezone(timedelta(hours=8))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG = os.path.join(ROOT, "01_kellect_logs", "exp3_3", "read.json")

# 被测程序的特征串：把实验自身的进程从系统噪声里挑出来
SCRIPT_TAGS = ("http_client.py", "local_http_server.py")

# Windows FILETIME 起点(1601-01-01)与 Unix 起点(1970-01-01)相差的秒数
FILETIME_EPOCH_DELTA = 11644473600

# 事件 → 边类型 的两张映射表（动作主体一律落地为边的 source）
FILE_EDGE = {
    "FileIORead":       "Read",
    "FileIOWrite":      "Write",
    "FileIOFileCreate": "CreateFile",
    "FileIOCreate":     "Open",
    "FileIODelete":     "Delete",
}
NET_EDGE = {
    "TcpIpConnectIPV4":    "Connect",
    "TcpIpAcceptIPV4":     "Accept",
    "TcpIpSendIPV4":       "Send",
    "TcpIpRecvIPV4":       "Recv",
    "TcpIpDisconnectIPV4": "Disconnect",
}


# ============================================================ 管道层
def parse_kellect_line(line):
    """Kellect 输出不是合法 JSON（路径里的反斜杠没转义），先转义再解析。"""
    return json.loads(line.replace("\\", "\\\\"))


def ts(f):
    """Windows FILETIME(单位 100ns，自 1601 起算) → 可读时间字符串。"""
    if not f:
        return "-"
    return datetime.fromtimestamp(f / 1e7 - FILETIME_EPOCH_DELTA, CST).strftime("%H:%M:%S.%f")[:-3]


def args_of(record):
    """实测键名是小写 args（README 里写的 Args 是错的）。"""
    return record.get("args") or {}


def nport(x):
    """sport/dport 是【网络字节序】，ntohs 之后才是人认识的端口号。

    实测证据：监听 18080(0x46A0) 的端口在日志里写作 41030(0xA046)。
    不做转换会得出"日志里没有这个端口"的错误结论。
    范围保护顺带挡住 Kellect 偶尔给出的 -1（网络事件的 TID 就是 -1）。
    """
    return socket.ntohs(x) if isinstance(x, int) and 0 <= x <= 0xFFFF else x


def load_records(path):
    """读日志 → 按时间升序排序。返回 (records, 坏行数)。"""
    records, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(parse_kellect_line(line))
            except json.JSONDecodeError:
                bad += 1        # 采集被强杀时最后一行常没写完，属正常
    records.sort(key=lambda r: r.get("TimeStamp", 0))
    return records, bad


def focus_pids(records, tags=SCRIPT_TAGS):
    """返回"被测程序及其全部后代"的 PID 集合，把图缩小到实验范围。

    ⚠ 3.3 实测：venv 重定向器让一次 Popen 产生两层 python.exe，
      进程树比你以为的多一层 —— 只收一层儿子会漏掉真正干活的进程。
    """
    roots = set()
    for r in records:
        if r.get("Event") != "ProcessStart":
            continue
        cmd = args_of(r).get("CommandLine") or ""
        if any(t in cmd for t in tags):
            roots.add(r["PID"])

    kids = defaultdict(set)
    for r in records:
        if r.get("Event") == "ProcessStart":
            kids[r["PPID"]].add(r["PID"])

    seen, stack = set(), list(roots)
    while stack:                       # 深度优先，沿 PPID→PID 收全部后代
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(kids.get(p, ()))
    return seen


# ============================================================ 实体识别层
def build_process_table(records):
    """预扫：PID → [(start_ts, UniqueProcessKey), ...]，按 start_ts 升序。

    为什么必须预扫：一条事件只带 PID，而"这个 PID 是哪个进程实例"的信息
    只在它自己的 ProcessStart 里。不先扫一遍就没法给事件定身份。

    为什么是"列表"而不是"单个值"：PID 会被系统回收复用，
    同一 PID 在窗口内可能先后属于多个进程实例。
    """
    table = defaultdict(list)
    for r in records:
        if r.get("Event") != "ProcessStart":
            continue
        table[r["PID"]].append((r.get("TimeStamp", 0),
                                args_of(r).get("UniqueProcessKey")))
    for pid in table:
        table[pid].sort(key=lambda x: x[0])      # 升序：后面好做区间查找
    return table


def process_key(pid, t, table):
    """把 (PID, 事件时刻) 解析成进程实体 key → (key, resolved)。

    做法：在该 PID 的所有 ProcessStart 里，找最后一个 start_ts <= t 的实例。
    - 找到 → ("process", pid, start_ts)，resolved=True
    - 找不到（事件早于该 PID 的第一个 ProcessStart，或该 PID 压根没有
      ProcessStart）→ 说明它在我们开始采集前就活着 → 兜底 ("process", pid, None)

    兜底的代价：同一 PID 若在窗口内被复用，兜底实例与前一个实例只能靠
    start_ts 是否为 None 区分，粒度较粗。所以 resolved=False 必须标出来。
    """
    cands = table.get(pid)
    if cands:
        found = None
        for start_ts, _upk in cands:            # cands 已升序
            if start_ts <= t:
                found = start_ts
            else:
                break
        if found is not None:
            return ("process", pid, found), True
    return ("process", pid, None), False


def norm_path(p):
    """路径规范化（统一分隔符 + 小写），作为文件实体的主键。

    代价：同一路径下"删除后重建"的文件会被并成一个节点。
    这是拿"覆盖率"换"版本区分度"的取舍 —— 若改成 FileKey 当主键，
    FileIOCreate 类事件（实测占子树 31%）会因为压根没有 FileKey 而全部丢掉。
    """
    if not p:
        return None
    return p.replace("/", "\\").rstrip("\\").lower()


def file_path(record):
    """取文件路径。实测字段分布：
       FileIORead/FileIOWrite  → FileName
       FileIOCreate            → OpenPath
       FileIOFileCreate        → FileName
    """
    a = args_of(record)
    return norm_path(a.get("FileName") or a.get("OpenPath"))


def net_endpoint(record):
    """网络端点 key。两个 addr:port 排序后拼成对称的元组。

    为什么排序：同一条 TCP 连接，客户端日志记的是 (57954 → 18080)，
    服务端日志记的是 (18080 → 57954)。不排序就是两个节点，图会被切成两半；
    排序后两端落在同一个节点上，才可能从客户端"走"到服务端。
    """
    a = args_of(record)
    sa, sp = a.get("saddr"), nport(a.get("sport"))
    da, dp = a.get("daddr"), nport(a.get("dport"))
    if sa is None or da is None:
        return None
    pair = sorted([f"{sa}:{sp}", f"{da}:{dp}"])
    return ("network", pair[0], pair[1])


# ============================================================ 建图层
def _proc_node(G, key, name, resolved, **extra):
    """加/补进程节点。已存在就只补空缺属性，不覆盖已有值。"""
    if key not in G:
        G.add_node(key, ntype="process", label=(name or "?").split("\\")[-1],
                   pid=key[1], start_ts=key[2], resolved=resolved)
    for k, v in extra.items():
        if v is not None and G.nodes[key].get(k) is None:
            G.nodes[key][k] = v


def _edge(G, u, v, etype, t, idx, stats):
    """加边。key 用 "边类型#日志行号" 保证确定性且唯一。

    ⚠ 这是 MultiDiGraph 而非 DiGraph 的关键用途：
      同一对 (P, File) 之间 6 条 Read + 1 条 Write 会变成 7 条独立边，
      各自带自己的 timestamp。换成 DiGraph 会塌成 1 条，类型还会被后者覆盖。
    """
    G.add_edge(u, v, key=f"{etype}#{idx}", etype=etype, ts=t, raw_index=idx)
    stats[f"edge:{etype}"] += 1


def build_graph(records, include_open=True):
    """事件流 → MultiDiGraph。"""
    G = nx.MultiDiGraph()
    G.graph["records"] = records      # 边 → 原始日志 的回溯靠它（见 q5）

    ptab = build_process_table(records)
    stats = Counter()
    file_keys = defaultdict(set)      # 路径 → 该路径上见过的 FileKey 集合
    unresolved = set()                # 走了兜底身份的 PID

    for idx, r in enumerate(records):
        ev = r.get("Event")
        t = r.get("TimeStamp", 0)
        pid = r.get("PID")
        src, ok = process_key(pid, t, ptab)
        if not ok:
            unresolved.add(pid)
        pname = r.get("PName") or "?"

        # ① 进程创建：本条记录的 PID = 子进程，PPID = 父进程
        if ev == "ProcessStart":
            a = args_of(r)
            _proc_node(G, src, a.get("ImageFileName") or pname, ok,
                       commandline=a.get("CommandLine"),
                       usersid=a.get("UserSID"),
                       unique_key=a.get("UniqueProcessKey"))
            parent, pok = process_key(r.get("PPID"), t, ptab)
            if not pok:
                unresolved.add(r.get("PPID"))
            _proc_node(G, parent, r.get("PPName"), pok)
            _edge(G, parent, src, "Create", t, idx, stats)
            continue

        # ② 文件事件：进程 → 文件
        if ev in FILE_EDGE:
            if ev == "FileIOCreate" and not include_open:
                stats["skip:FileIOCreate(按需关闭)"] += 1
                continue
            path = file_path(r)
            if not path:
                stats["skip:文件事件无路径"] += 1
                continue
            fkey = ("file", path)
            fk = args_of(r).get("FileKey")
            if fk:
                file_keys[path].add(fk)
            if fkey not in G:
                G.add_node(fkey, ntype="file", label=os.path.basename(path) or path,
                           path=path)
            _proc_node(G, src, pname, ok)
            _edge(G, src, fkey, FILE_EDGE[ev], t, idx, stats)
            continue

        # ③ 网络事件：进程 → 端点
        if ev in NET_EDGE:
            ep = net_endpoint(r)
            if ep is None:
                stats["skip:网络事件无端点"] += 1
                continue
            if ep not in G:
                G.add_node(ep, ntype="network", label=f"{ep[1]} | {ep[2]}", pair=list(ep[1:]))
            _proc_node(G, src, pname, ok)
            _edge(G, src, ep, NET_EDGE[ev], t, idx, stats)
            continue

        # ④ 其余事件不静默吞掉 —— 数出来，让信息损失可见
        stats[f"skip:{ev}"] += 1

    # 把 FileKey 挂到文件节点上，并把"一个路径多个 FileKey"这种冲突挑出来
    conflicts = {}
    for path, keys in file_keys.items():
        node = ("file", path)
        if node in G:
            G.nodes[node]["file_keys"] = sorted(keys)
            if len(keys) > 1:
                conflicts[path] = sorted(keys)

    G.graph["stats"] = stats
    G.graph["unresolved_pids"] = sorted(unresolved)
    G.graph["multi_filekey_paths"] = conflicts
    return G


# ============================================================ 查询层（4.3 必做）
def q1_files_read_by_process(G, proc):
    """某进程读取了哪些文件 → {文件节点: 次数}"""
    hit = Counter()
    for _u, v, d in G.out_edges(proc, data=True):
        if d["etype"] == "Read":
            hit[v] += 1
    return hit


def q2_processes_writing_file(G, file_):
    """某文件被哪些进程写入 → {进程节点: 次数}"""
    hit = Counter()
    for u, _v, d in G.in_edges(file_, data=True):
        if d["etype"] == "Write":
            hit[u] += 1
    return hit


def q3_parent_of(G, proc):
    """某进程的父进程。

    ⚠ 实测：venv 重定向器会让答案指向重定向器，而不是"真实"父进程。
    """
    return [u for u, _v, d in G.in_edges(proc, data=True) if d["etype"] == "Create"]


def q4_k_hop_neighbors(G, node, k):
    """k 跳邻居（无向意义上遍历）。

    注意 to_undirected() 返回的是副本，不会把原图改成无向图 ——
    原图的方向必须保住（4.2 的教训）。
    """
    U = G.to_undirected()
    dist = nx.single_source_shortest_path_length(U, node, cutoff=k)
    return [n for n, d in dist.items() if 0 < d <= k]


def q5_edge_to_raw_log(G, u, v, key=None):
    """边 → 原始 Kellect 记录（双向追溯的另一半）。

    key=None 时返回该对节点间的全部边 [(边key, 原始记录), ...]。
    原文第八节要求的链路 `Anomaly Score → Node/Event → Entity → Raw Log`
    能不能走通，全靠这里的 raw_index。
    """
    data = G.get_edge_data(u, v) or {}
    if key is None:
        return [(k, G.graph["records"][d["raw_index"]]) for k, d in data.items()]
    return G.graph["records"][data[key]["raw_index"]]


# ============================================================ 输出
def report(G):
    s = G.graph["stats"]
    print("\n===== 图规模 =====")
    print(f"  节点 {G.number_of_nodes()}    边 {G.number_of_edges()}")
    print(f"  节点类型: {dict(Counter(d['ntype'] for _, d in G.nodes(data=True)))}")
    print(f"  边类型:   {dict(Counter(d['etype'] for _, _, d in G.edges(data=True)))}")

    skips = {k.split(":", 1)[1]: v for k, v in s.items() if k.startswith("skip:")}
    if skips:
        print(f"  跳过的事件(信息损失，共 {sum(skips.values())} 条): {dict(sorted(skips.items(), key=lambda x: -x[1]))}")

    up = G.graph["unresolved_pids"]
    if up:
        print(f"  ⚠ 兜底身份的 PID: {len(up)} 个，例 {up[:8]}")
    mf = G.graph["multi_filekey_paths"]
    if mf:
        print(f"  ⚠ 同一路径多个 FileKey（文件被重建/复用）: {mf}")


def demo(G):
    """演示 4.3 要求的 5 个查询。"""
    secret = next((n for n, d in G.nodes(data=True)
                   if d["ntype"] == "file" and "secret" in (d.get("path") or "")), None)
    if secret is None:
        print("\n（图里没有 secret.txt 节点 —— fixed 模式本来就该如此）")
        return

    readers = [u for u, _v, d in G.in_edges(secret, data=True) if d["etype"] == "Read"]
    if not readers:
        print("\n（没有任何进程读过 secret.txt）")
        return
    proc = readers[0]

    print(f"\n===== 5 个查询演示 =====")
    print(f"  焦点进程: {G.nodes[proc]['label']} {proc[1]}")

    r = q1_files_read_by_process(G, proc)
    print(f"\n  q1 它读过的文件（共 {len(r)} 个，前 5）:")
    for f, c in r.most_common(5):
        print(f"      {c:>3} 次  {G.nodes[f]['label']}")

    w = q2_processes_writing_file(G, secret)
    print(f"\n  q2 谁写过 {G.nodes[secret]['label']}: "
          f"{[(G.nodes[p]['label'], p[1], c) for p, c in w.items()]}")

    print(f"\n  q3 {proc[1]} 的父进程: {[p[1] for p in q3_parent_of(G, proc)]}")

    hop2 = q4_k_hop_neighbors(G, secret, 2)
    print(f"\n  q4 从 {G.nodes[secret]['label']} 出发 2 跳邻居 {len(hop2)} 个:")
    for n in hop2[:6]:
        print(f"      {G.nodes[n]['ntype']:<8} {G.nodes[n]['label']}")

    ev = q5_edge_to_raw_log(G, proc, secret)
    print(f"\n  q5 从边回到原始日志（该对节点间 {len(ev)} 条边，看第一条）:")
    k, rec = ev[0]
    print(f"      边key={k}")
    print(f"      → Event={rec['Event']}  PName={rec['PName']}  "
          f"PID={rec['PID']}  IoSize={args_of(rec).get('IoSize')}  "
          f"时间={ts(rec.get('TimeStamp'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=DEFAULT_LOG, help="Kellect 日志路径")
    ap.add_argument("--all", action="store_true",
                    help="不做范围过滤，把整份日志都建进去（用于 4.4 看噪声）")
    ap.add_argument("--no-open", action="store_true",
                    help="丢掉 FileIOCreate（句柄级打开事件），用于对比聚合前后")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"[X] 找不到日志: {args.log}")
        raise SystemExit(1)

    records, bad = load_records(args.log)
    print(f"读到 {len(records)} 条事件，跳过 {bad} 条坏行")

    if not args.all:
        keep = focus_pids(records)
        records = [r for r in records if r["PID"] in keep]
        print(f"过滤到被测程序进程子树 PID {sorted(keep)} 后剩 {len(records)} 条")

    G = build_graph(records, include_open=not args.no_open)
    report(G)
    demo(G)


if __name__ == "__main__":
    main()