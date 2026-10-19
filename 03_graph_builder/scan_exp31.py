"""
阶段 3.1 日志扫描：从 Kellect 输出里读回你自己制造的文件行为

用法：
    1. 先跑一次，它会列出候选的 python 进程 PID
    2. 把属于你的那个 PID 填进 TARGET_PID
    3. 再跑一次
"""
"""
阶段 3.1 日志扫描：从 Kellect 输出里读回你自己制造的文件行为
"""

import json
import os
from collections import Counter
from datetime import datetime, timezone, timedelta

TARGET_PID = 4652          # <<< 改成你自己运行 file_ops.py 时打印出来的 PID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "01_kellect_logs", "exp3_1", "file_ops.json")
TAG = "test_a"          # 你的文件名前缀（test_a.text / test_a_copy.text / test_a_renamed.text）
                        # 注意：不能写成 "exp3_1_work" —— 主文件 test_a.text 建在 02_experiments/ 下，
                        #       不在工作目录里，用目录名过滤会漏掉前三步（创建/读取/追加）。
CST = timezone(timedelta(hours=8))

def parse_kellect_line(line: str):
    """Kellect 输出的不是合法 JSON：路径里的反斜杠没转义，
    一律把所有反斜杠转义一遍再解析。"""
    return json.loads(line.replace("\\", "\\\\"))

def to_local_time(filestime: int) -> str:
    """Windows FILETIME（自 1601 年起的 100 纳秒数）→ 本地时间字符串。"""
    return datetime.fromtimestamp(filestime / 1e7 - 11644473600, CST).strftime("%H:%M:%S.%f")[:-3]

# ---------------------------------------------------------- 加载
if not os.path.exists(LOG):
    print("找不到日志文件：")
    print(f"  {LOG}")
    print("\n说明采集还没做。正确顺序是：\n")
    print("  ① 管理员 PowerShell 启动采集器（先等 2~3 秒让它就绪）")
    print(r'     cd "C:\Users\ww\Desktop\PIDS\01_kellect_logs\_tool\KellectAgent-Windows-2.0\release"')
    print(r'     .\kellect.exe -e 0x9 -f "C:\Users\ww\Desktop\PIDS\01_kellect_logs\exp3_1\file_ops.json"')
    print("\n  ② 另开一个 PowerShell 跑被测程序")
    print(r"     cd C:\Users\ww\Desktop\PIDS")
    print(r"     .\.venv\Scripts\python.exe 02_experiments\file_ops.py")
    print("\n  ③ 回采集器窗口按 Ctrl+C")
    print("\n  ④ 再跑本脚本")
    raise SystemExit(1)

records, skipped = [], []
with open(LOG, encoding="utf-8", errors="replace") as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(parse_kellect_line(line))
        except json.JSONDecodeError as e:
            skipped.append((ln, str(e)))      # 被 Ctrl+C 截断的半行属正常

records.sort(key=lambda r: r["TimeStamp"])
print(f"读到 {len(records)} 条事件，跳过 {len(skipped)} 条")
for ln, err in skipped:
    print(f"  [跳过] 第 {ln} 行: {err}")
if records:
    print(f"时间范围 {to_local_time(records[0]['TimeStamp'])} → "
          f"{to_local_time(records[-1]['TimeStamp'])}")

# ---------------------------------------------------------- 1. 一条完整记录
print("\n===== 1. 一条完整记录（确认公共字段）=====")
print(json.dumps(records[0], ensure_ascii=False, indent=2)[:1000])
print('注意：实际字段是 "args"（小写），README 写的是 "Args"')

# ---------------------------------------------------------- 2. 事件类型分布
print("\n===== 2. 事件类型分布（整台机器）=====")
for name, cnt in Counter(r["Event"] for r in records).most_common():
    print(f"  {name:<20}{cnt}")

# ---------------------------------------------------------- 4. 找自己的进程
if not TARGET_PID:
    print("\n===== 4. 候选进程（TARGET_PID 还没填）=====")
    cand = {}
    for r in records:
        if "python" in (r.get("PName") or "").lower():
            cand.setdefault(r["PID"], r.get("PName"))
    for pid, pname in sorted(cand.items()):
        print(f"  PID={pid:<8}{pname}")
    print("\n把属于你的那个 PID 填进 TARGET_PID，然后重跑本脚本")
    raise SystemExit(0)

mine = [r for r in records if r["PID"] == TARGET_PID]
touched = [r for r in mine if TAG in json.dumps(r.get("args") or {}, ensure_ascii=False)]
print(f"\n===== PID={TARGET_PID} 共 {len(mine)} 条事件，"
      f"其中涉及 {TAG} 的 {len(touched)} 条 =====")
print(f"事件类型分布: {dict(Counter(r['Event'] for r in touched))}")

# ---------------------------------------------------------- 3. 时间线
print(f"\n===== 3. 时间线（涉及 {TAG} 的事件；空行 = 步骤分界）=====")
print(f"{'时间':<14}{'Event':<18}{'文件':<24}{'FileKey':<12}{'FileObject':<12}IO")
print("-" * 106)
prev = None
for r in touched:
    if prev is not None and r["TimeStamp"] - prev > 1e7:      # 1 秒 = 1e7
        print()
    a = r.get("args") or {}
    p = a.get("OpenPath") or a.get("FileName") or ""
    name = p.split("\\")[-1] if p else "(无路径)"
    io = f"IoSize={a['IoSize']} Offset={a.get('Offset')}" if "IoSize" in a else ""
    print(f"{to_local_time(r['TimeStamp']):<14}{r['Event']:<18}{name:<24}"
          f"{str(a.get('FileKey')):<12}{str(a.get('FileObject')):<12}{io}")
    prev = r["TimeStamp"]

# ---------------------------------------------------------- 5. 写入明细
print("\n===== 5. 写入相关事件 =====")
writes = [r for r in touched if "Write" in r["Event"]]
print(f"写入事件共 {len(writes)} 条")
for r in writes:
    a = r["args"]
    print(f"  {to_local_time(r['TimeStamp'])}  {r['Event']}  "
          f"{(a.get('FileName') or a.get('OpenPath') or '').split(chr(92))[-1]}  "
          f"IoSize={a.get('IoSize')} Offset={a.get('Offset')} FileKey={a.get('FileKey')}")

# ---------------------------------------------------------- 6. FileKey 覆盖
print("\n===== 6. FileKey / FileObject 覆盖情况（整台机器）=====")
per_event = {}
for r in records:
    a = r.get("args") or {}
    d = per_event.setdefault(r["Event"], [0, 0, 0])
    d[0] += 1
    d[1] += "FileKey" in a
    d[2] += "FileObject" in a
print(f"{'Event':<20}{'总数':<8}{'有FileKey':<12}{'有FileObject':<14}覆盖率")
print("-" * 66)
for ev, (tot, fk, fo) in sorted(per_event.items(), key=lambda kv: -kv[1][0]):
    print(f"{ev:<20}{tot:<8}{fk:<12}{fo:<14}{fk / tot * 100:.0f}%")