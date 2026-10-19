"""
阶段 3.2 被测程序：制造可预期的进程树 + 文件访问归属

设计意图（对应档案 3.2 的四个引导问题）：
  parent  —— 只负责创建进程，自己不碰任何文件
  child   —— 读 File A，并创建 grandchild
  grandchild —— 写 File B
三层都是同名 python.exe，只用 PID/PPID 区分。

另有 40 个短命进程（tick），用来观察 PID 的分配规律。

运行：
    parent 模式（默认，直接跑本文件即可）
    child / grandchild / tick 由 parent 自动拉起，不用手动跑
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "exp3_2_work")
FILE_A = os.path.join(WORK, "file_a.txt")
FILE_B = os.path.join(WORK, "file_b.txt")
PY = sys.executable


def log(role, msg):
    """每一行都打出 role / pid / ppid —— 这三个是恢复进程树的最小信息"""
    print(f"[{time.strftime('%H:%M:%S')}] {role:<10} pid={os.getpid():<7} "
          f"ppid={os.getppid():<7} {msg}", flush=True)


def spawn(role):
    return subprocess.Popen([PY, os.path.abspath(__file__), "--role", role])


# ---------------------------------------------------------------- 三个角色

def run_child():
    """child：读 File A，然后创建 grandchild"""
    log("child", "start")
    with open(FILE_A, encoding="utf-8") as f:
        data = f.read()
    log("child", f"读完 File A（{len(data)} 字符）")

    g = spawn("grandchild")
    log("child", f"已创建 grandchild pid={g.pid}")
    g.wait()
    log("child", "end")


def run_grandchild():
    """grandchild：写 File B"""
    log("grandchild", "start")
    with open(FILE_B, "w", encoding="utf-8") as f:
        f.write("written by grandchild\n")
    log("grandchild", "end")


def run_tick():
    """短命进程：什么都不做，只用于观察 PID 分配"""
    pass


def run_parent():
    os.makedirs(WORK, exist_ok=True)
    with open(FILE_A, "w", encoding="utf-8") as f:
        f.write("content of file A\n")

    log("parent", "start")

    # ① 三级链：parent -> child -> grandchild
    c = spawn("child")
    log("parent", f"已创建 child pid={c.pid}")
    c.wait()
    log("parent", "child 已退出")
    time.sleep(1.5)

    # ② 同名同类角色的多个实例（回答"只用 PName 作节点会怎样"）
    procs = [spawn("grandchild") for _ in range(3)]
    for p in procs:
        log("parent", f"又创建 grandchild 实例 pid={p.pid}")
    for p in procs:
        p.wait()
    time.sleep(1.5)

    # ③ 短命进程序列（观察 PID 是否递增 / 是否重号）
    seen = []
    for _ in range(40):
        p = subprocess.Popen([PY, os.path.abspath(__file__), "--role", "tick"])
        seen.append(p.pid)
        p.wait()

    log("parent", f"40 个 tick 的 PID: {seen}")
    print(f"  PID 是否单调递增? {seen == sorted(seen)}")
    print(f"  PID 是否重号?     {len(seen) != len(set(seen))}"
          f"（重复 {len(seen) - len(set(seen))} 个）")

    log("parent", "end")


# ---------------------------------------------------------------- 入口

if __name__ == "__main__":
    role = "parent"
    if "--role" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1]

    {"parent": run_parent,
     "child": run_child,
     "grandchild": run_grandchild,
     "tick": run_tick}[role]()