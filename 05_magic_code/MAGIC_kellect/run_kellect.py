"""Kellect 端到端：用 MAGIC 官方训练/评估代码跑 Kellect 采集的日志数据。

本文件是「MAGIC 适配 Kellect」的上传展示脚本。它依赖同仓库的
PIDS 主项目内容（build_graph、Kellect 实验日志），因此这段逻辑原本
写在 MAGIC 仓库目录里；此处用相对路径指向 PIDS 根目录，保证跨机可用。

约定路径（相对本仓库根）：
  · MAGIC 官方仓库          05_magic_code/MAGIC
  · Kellect 适配解析器      utils/kellect_parser.py
  · Kellect 实验日志        01_kellect_logs/exp3_3/{fixed,read}.json
    - fixed.json = 良性：服务端从不碰 secret.txt，只回固定串
    - read.json  = 攻击：服务端真实读取 secret.txt 并经网络发出（模拟外泄）

流程：
  1) cd 进入 MAGIC 仓库根目录
  2) 生成 data/kellect 的 train.pkl/test.pkl/malicious.pkl
  3) 调用官方 train.py 训练实体级掩码 GAE
  4) 调用官方 eval.py 用 KNN 做实体级异常检测，结果存入 results/
"""
import os
import subprocess
import sys

PIDS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAGIC_DIR = os.path.join(PIDS_ROOT, "05_magic_code", "MAGIC")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_in_magic(cmd):
    """在 MAGIC 仓库目录里用当前解释器执行命令，返回 stdout。"""
    p = subprocess.run(cmd, cwd=MAGIC_DIR, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out)
    return out


print("===== 1) 生成 data/kellect（复用 kellect_parser）=====")
run_in_magic([sys.executable, "utils", "kellect_parser.py"])

print("\n===== 2) MAGIC 训练（kellect 数据集）=====")
run_in_magic([sys.executable, "train.py", "--dataset", "kellect"])

print("\n===== 3) MAGIC 评估（kellect 数据集）=====")
out = run_in_magic([sys.executable, "eval.py", "--dataset", "kellect"])
with open(os.path.join(RESULTS_DIR, "kellect_eval_raw.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print(f"评估输出已保存到 results/kellect_eval_raw.txt")