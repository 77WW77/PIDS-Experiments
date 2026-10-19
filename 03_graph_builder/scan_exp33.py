"""
阶段 3.3 分析脚本：从 Kellect 日志里定位一次本地 HTTP 请求

它按档案第 5 页的要求，逐项定位：
    客户端进程 / 服务端进程 / IP / 端口 / 连接事件 / 服务器读取测试文件的行为

用法（管理员窗口跑 run_collect.ps1 采完日志后，普通窗口跑本脚本）：

    .\run_collect.ps1 -Exp 3_3 -Script 02_experiments\http_client.py -Mask 0x89 -ScriptArgs "--spawn-server read"
    .\.venv\Scripts\python.exe 03_graph_builder\scan_exp33.py

把 read 换成 fixed 再采一份，然后用第二份日志做对照：

    .\.venv\Scripts\python.exe 03_graph_builder\scan_exp33.py 01_kellect_logs\exp3_3\fixed.json
"""

import json
import os
import socket
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "01_kellect_logs", "exp3_3", "log.json")
if len(sys.argv) > 1:
    LOG = sys.argv[1]

PORT = 18080
TAG = "exp3_3_work"          # 工作目录名，用来把实验自身的文件访问从噪声里挑出来


def parse_kellect_line(line: str):
    """把 Kellect 的准 JSON 修成合法 JSON 再解析（它不转义路径里的反斜杠）。"""
    return json.loads(line.replace("\\", "\\\\"))


def ts(f):
    return datetime.fromtimestamp(f / 1e7 - 11644473600, CST).strftime("%H:%M:%S.%f")[:-3]


def args_of(rec):
    return rec.get("args") or rec.get("Args") or {}


def is_net(name: str) -> bool:
    n = name.lower()
    return "tcp" in n or "udp" in n


def nport(x):
    """Kellect 的 sport/dport 是【网络字节序】，ntohs 之后才是人认识的端口号。
    例：18080(0x46A0) 在日志里写作 41030(0xA046)。"""
    return socket.ntohs(x) if isinstance(x, int) else x


def main():
    if not os.path.exists(LOG) or os.path.getsize(LOG) == 0:
        print(f"[X] 找不到或为空: {LOG}")
        print("    先以【管理员身份】运行采集（注意 -Mask 必须含 0x80 才有网络事件）：")
        print(r'    .\run_collect.ps1 -Exp 3_3 -Script 02_experiments\http_client.py '
              r'-Mask 0x89 -ScriptArgs "--spawn-server read"')
        raise SystemExit(1)

    records, bad = [], 0
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(parse_kellect_line(line))
            except json.JSONDecodeError:
                bad += 1

    print(f"读到 {len(records)} 条事件，跳过 {bad} 条")
    times = [r["TimeStamp"] for r in records if "TimeStamp" in r]
    if times:
        print(f"时间范围 {ts(min(times))} → {ts(max(times))}")

    # ---------------------------------------------------- 1. 客户端 / 服务端进程
    print("\n===== 1. 进程事件（只列与本次实验有关的）=====")
    print(f"{'时间':<14}{'PID':<8}{'PPID':<8}{'PName':<14}角色 / CommandLine")
    print("-" * 100)
    procs = []
    for r in records:
        if r.get("Event") != "ProcessStart":
            continue
        a = args_of(r)
        cl = str(a.get("CommandLine") or "")
        if "http_client.py" not in cl and "local_http_server.py" not in cl:
            continue
        role = "服务端" if "local_http_server.py" in cl else "客户端"
        hop = " [venv 重定向器]" if ".venv" in cl.lower() else ""
        procs.append((r.get("PID"), role, hop, cl))
        print(f"{ts(r['TimeStamp']):<14}{r.get('PID', ''):<8}{r.get('PPID', ''):<8}"
              f"{str(r.get('PName', '')):<14}{role}{hop}  {cl[:60]}")
    if not procs:
        print("  (无) —— 检查 -Script / -ScriptArgs 是否写对")

    server_pids = {pid for pid, role, hop, _ in procs if role == "服务端"}
    client_pids = {pid for pid, role, hop, _ in procs if role == "客户端"}

    # ---------------------------------------------------- 2. 网络事件类型分布
    print("\n===== 2. 网络事件类型分布 =====")
    net = [r for r in records if is_net(str(r.get("Event", "")))]
    net.sort(key=lambda r: r.get("TimeStamp", 0))       # 按时间排，便于读时序
    if not net:
        print("  (无网络事件) —— -e 掩码是否含 TCPIP(0x80)？3.3 要用 -Mask 0x89")
    else:
        for ev, n in Counter(r["Event"] for r in net).most_common():
            print(f"  {ev:<28}{n:>6}")
        a = args_of(net[0])
        print(f"  字段样例（{net[0]['Event']}）: {sorted(a.keys())}")

    # ---------------------------------------------------- 3. 与 PORT 相关的网络事件
    print(f"\n===== 3. 涉及端口 {PORT} 的网络事件 =====")
    print("  ⚠ Kellect 的 sport/dport 是【网络字节序】，下表已 ntohs 转成主机序")
    print(f"{'时间':<14}{'Event':<24}{'PID':<8}{'PName':<14}关键字段（端口已转主机序）")
    print("-" * 100)
    hit = 0
    for r in net:
        a = args_of(r)
        sp, dp = nport(a.get("sport")), nport(a.get("dport"))
        if PORT not in (sp, dp):
            continue
        hit += 1
        key = {"saddr": a.get("saddr"), "sport": sp,
               "daddr": a.get("daddr"), "dport": dp,
               "size": a.get("size")}
        print(f"{ts(r['TimeStamp']):<14}{r['Event']:<24}{r.get('PID', ''):<8}"
              f"{str(r.get('PName', '')):<14}{key}")
    if hit == 0:
        print(f"  (无) —— 端口 {PORT} 未出现在网络事件里")
    else:
        print(f"  共 {hit} 条；其中客户端 PID {sorted(client_pids)}、服务端 PID {sorted(server_pids)}")

    # ---------------------------------------------------- 4. 服务端是否读了 File A
    print("\n===== 4. 服务端进程的文件行为（重点：有没有读 secret.txt）=====")
    print(f"{'时间':<14}{'Event':<20}{'PID':<8}{'文件':<22}{'FileKey':<13}IO")
    print("-" * 100)
    dir_only = 0        # 只碰了工作目录、没碰 secret.txt 的（含启动时建目录）
    secret_hits = 0     # 碰了 secret.txt
    secret_reads = 0    # 其中是 Read（≠ 仅 Create/Close）
    for r in records:
        if r.get("PID") not in server_pids:
            continue
        a = args_of(r)
        name = str(a.get("FileName") or a.get("OpenPath") or "")
        hit_secret = "secret" in name.lower()
        if not (TAG in name or hit_secret):
            continue
        if hit_secret:
            secret_hits += 1
            if r.get("Event") == "FileIORead":
                secret_reads += 1
        else:
            dir_only += 1
        io = ""
        if a.get("IoSize") is not None:
            io = f"IoSize={a.get('IoSize')} Offset={a.get('Offset')}"
        mark = "秘密文件" if hit_secret else "仅工作目录"
        print(f"{ts(r['TimeStamp']):<14}{str(r.get('Event', '')):<20}{r.get('PID', ''):<8}"
              f"{name[-21:]:<22}{str(a.get('FileKey', '')):<13}{io}  [{mark}]")
    print(f"  → 只碰【工作目录】的事件数   : {dir_only}")
    print(f"  → 碰【secret.txt】的事件数   : {secret_hits}（其中 FileIORead {secret_reads} 条）")

    # ---------------------------------------------------- 5. 判定
    print("\n===== 5. 本份日志的判定 =====")
    print(f"  服务端是否真的读了 secret.txt : {'是' if secret_reads else '否'}")
    print("      （判定依据只看 secret.txt 的 FileIORead，不能拿'碰了工作目录'顶替）")
    sizes = sorted({args_of(r).get("size", 0) for r in net
                    if r.get("Event") == "TcpIpSendIPV4" and nport(args_of(r).get("sport")) == PORT})
    print(f"  服务端发出的 IPv4 报文长度（去重）: {sizes}")
    print("      153 是 HTTP 响应头；另一个就是响应体 —— 但它只是【字节数】，没有内容")
    print("  网络事件里能否看出传输内容     : 不能（只有地址/端口/字节数，没有内容）")
    print("  对照要点：与另一份日志（read / fixed）比第 3、4 节 ——")
    print("            第 3 节形态几乎相同（只差响应体字节数）→ 事件类型分不出内容来源")
    print("            第 4 节才见分晓（有/无 secret.txt 的 FileIORead）")


if __name__ == "__main__":
    main()