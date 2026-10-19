"""
阶段 3.2 日志分析：从事件流恢复进程树 + 定位文件行为的归属

要回答的四个问题都落在下面的 5 个小节里：
  Q1 父子都叫 python.exe，凭什么当两个实体  -> 第 1、5 节
  Q2 PID 能否长期作唯一标识                 -> 第 3 节
  Q3 只用 PName 作节点会怎样                -> 第 2 节
  Q4 忽略子进程会漏掉什么                   -> 第 4、5 节
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "01_kellect_logs", "exp3_2", "log.json")
TAG = "exp3_2_work"
CST = timezone(timedelta(hours=8))


def parse_kellect_line(line):
    """Kellect 输出不是合法 JSON（路径反斜杠未转义），一律转义后再解析"""
    return json.loads(line.replace("\\", "\\\\"))


def ts(f):
    """Windows FILETIME -> 本地时间"""
    return datetime.fromtimestamp(f / 1e7 - 11644473600, CST).strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------- 加载
if not os.path.exists(LOG):
    print(f"[X] 找不到日志: {LOG}")
    print("    请先在【管理员 PowerShell】里运行：")
    print(r'      cd C:\Users\ww\Desktop\PIDS')
    print(r'      .\run_collect.ps1 -Exp 3_2 -Script 02_experiments\parent_child.py')
    raise SystemExit(1)

records, bad = [], 0
with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(parse_kellect_line(line))
        except json.JSONDecodeError:
            bad += 1          # 采集被强杀时最后一行可能没写完，属正常

records.sort(key=lambda r: r["TimeStamp"])
print(f"读到 {len(records)} 条事件，跳过 {bad} 条")
if records:
    print(f"时间范围 {ts(records[0]['TimeStamp'])} → {ts(records[-1]['TimeStamp'])}")


def args(r):
    return r.get("args") or {}


# ---------------------------------------------------------------- 1. 进程事件
starts = [r for r in records if r["Event"] == "ProcessStart"]
ends = [r for r in records if r["Event"] == "ProcessEnd"]
print(f"\n===== 1. 进程事件  ProcessStart={len(starts)}  ProcessEnd={len(ends)} =====")
print(f"{'时间':<14}{'PID':<8}{'PPID':<8}{'PName':<16}CommandLine")
print("-" * 100)
for r in starts:
    cmd = (args(r).get("CommandLine") or "")[:52]
    print(f"{ts(r['TimeStamp']):<14}{r['PID']:<8}{r['PPID']:<8}{r['PName']:<16}{cmd}")

# ---------------------------------------------------------------- 2. 只用 PName
print("\n===== 2. 若只用 PName 作节点 =====")
by_name = Counter(r["PName"] for r in starts)
for name, c in by_name.most_common(8):
    print(f"  {name:<24}{c} 个实例")
print(f"  不同进程名 {len(by_name)} 种  →  实例总数 {len(starts)}")
print(f"  ** 节点数会从 {len(starts)} 被压缩到 {len(by_name)} **")

# ---------------------------------------------------------------- 3. PID 重号
print("\n===== 3. PID 重号检查 =====")
pid_times = defaultdict(list)
for r in starts:
    pid_times[r["PID"]].append(r["TimeStamp"])
dup = {p: t for p, t in pid_times.items() if len(t) > 1}
print(f"  出现的唯一 PID 数: {len(pid_times)}  /  ProcessStart 数: {len(starts)}")
if dup:
    for p, t in dup.items():
        print(f"  [重号] PID={p} 出现在 {[ts(x) for x in t]}")
else:
    print("  本窗口内未发现重号")
    print("  提示: 这只说明短时间内没撞上。PID 可复用是 OS 的语义，不是统计结论")

# ---------------------------------------------------------------- 4. 文件行为归属
touched = [r for r in records
           if TAG in json.dumps(args(r), ensure_ascii=False)]
by_pid = Counter((r["PID"], r["PPID"], r["PName"]) for r in touched)
print(f"\n===== 4. 涉及 {TAG} 的事件共 {len(touched)} 条，按进程归属 =====")
print(f"{'PID':<8}{'PPID':<8}{'PName':<16}{'条数':<8}")
print("-" * 44)
for (pid, ppid, name), c in by_pid.most_common():
    print(f"{pid:<8}{ppid:<8}{name:<16}{c:<8}")

# ---------------------------------------------------------------- 5. 时间线
print(f"\n===== 5. {TAG} 相关时间线（带 PID/PPID，用于还原三级链）=====")
print(f"{'时间':<14}{'Event':<18}{'PID':<8}{'PPID':<8}{'文件':<20}IO")
print("-" * 96)
for r in touched:
    a = args(r)
    p = a.get("OpenPath") or a.get("FileName") or ""
    name = p.split("\\")[-1] if p else ""
    io = f"IoSize={a['IoSize']} Offset={a.get('Offset')}" if "IoSize" in a else ""
    print(f"{ts(r['TimeStamp']):<14}{r['Event']:<18}{r['PID']:<8}{r['PPID']:<8}{name:<20}{io}")

# ---------------------------------------------------------------- 附加：进程树
print("\n===== 附加：从 (PID, PPID) 还原的进程树 =====")
children = defaultdict(list)
for r in starts:
    children[r["PPID"]].append((r["PID"], r["PName"]))

for r in starts:
    # 只打印"自己是被测程序拉起来的"那一棵（PPID 也出现在 ProcessStart 里）
    if r["PPID"] in pid_times:
        continue
    stack = [(r["PID"], r["PName"], 0)]
    while stack:
        pid, name, depth = stack.pop()
        print(f"  {'    ' * depth}{name} (PID={pid})")
        for cpid, cname in children.get(pid, []):
            stack.append((cpid, cname, depth + 1))