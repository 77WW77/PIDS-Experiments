"""
阶段 3.3 被测程序（客户端）：访问本地 HTTP 服务并校验响应

两种用法：

  ① 一条命令跑完整实验（推荐，交给 run_collect.ps1）——
     客户端自己拉起服务端，父子两个进程都落在同一个采集窗口里：

        .\run_collect.ps1 -Exp 3_3 -Script "02_experiments\http_client.py --spawn-server read" -Mask 0x89

     把 read 换成 fixed 就是「证伪实验」的另一半。

  ② 服务端已在另一终端手工跑着，客户端只发请求：

        .\.venv\Scripts\python.exe 02_experiments\http_client.py
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "local_http_server.py")
PY = sys.executable
N = 3           # 请求次数，与 local_http_server.py 的 N_REQUESTS 保持一致
GAP = 1.5       # 两次请求之间的间隔，便于按时间戳切分日志


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] client {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn-server", choices=["read", "fixed"], default=None,
                    help="顺带拉起服务端（read=读 File A，fixed=返回固定串）")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    srv = None
    if args.spawn_server:
        srv = subprocess.Popen(
            [PY, SERVER, "--mode", args.spawn_server, "--port", str(args.port)]
        )
        log(f"已拉起服务端 mode={args.spawn_server} pid={srv.pid}")
        time.sleep(2.0)         # 等服务端 listen 完成

    try:
        for i in range(1, N + 1):
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = resp.read()
            dt = (time.time() - t0) * 1000
            log(f"#{i} status={resp.status} bytes={len(data)} "
                f"sha256[:16]={hashlib.sha256(data).hexdigest()[:16]} {dt:.0f}ms")
            if i < N:
                time.sleep(GAP)
    finally:
        if srv:
            srv.terminate()
            srv.wait()
            log("服务端已停止")
    log("done")


if __name__ == "__main__":
    main()