"""
阶段 3.3 被测程序（服务端）：仅监听 127.0.0.1 的本地 HTTP 服务

对应档案第 5 页的「核心思辨」：
    P → Read → File A   与   P → Connect → Endpoint
    能否推出「File A 的内容通过该连接发送出去了」？

为此提供两种模式 —— 日志形态几乎相同，语义却完全不同：

    --mode read    真正读取 File A，把内容作为响应体返回     ← "确实把文件发出去了"
    --mode fixed   完全不碰 File A，直接返回固定字符串        ← "看起来一样，其实没发文件"

手工运行（另开一个终端跑 http_client.py）：
    .\.venv\Scripts\python.exe 02_experiments\local_http_server.py --mode read
    .\.venv\Scripts\python.exe 02_experiments\local_http_server.py --mode fixed

注意：一般不需要手工跑 —— http_client.py 的 --spawn-server 会自动拉起本脚本，
      这样服务端与客户端都会落在同一个采集窗口里。
"""

import argparse
import hashlib
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "exp3_3_work")
FILE_A = os.path.join(WORK, "secret.txt")
HOST = "127.0.0.1"
PORT = 18080
N_REQUESTS = 3          # 收到这么多次请求后自动退出，避免忘记停止采集

MODE = "read"
_served = 0
_lock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] server {msg}", flush=True)


def read_file_a() -> bytes:
    with open(FILE_A, "rb") as f:
        return f.read()


def fixed_body() -> bytes:
    return b"fixed response, file A was never read\n"


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 = 短连接：一次请求一个 TCP 连接，日志里的连接事件数才好数
    protocol_version = "HTTP/1.0"
    server_version = "PIDSExp33/1.0"

    def _make_body(self) -> bytes:
        body = read_file_a() if MODE == "read" else fixed_body()
        log(f"mode={MODE} body={len(body)}B "
            f"sha256[:16]={hashlib.sha256(body).hexdigest()[:16]}")
        return body

    def do_GET(self):
        global _served
        body = self._make_body()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

        with _lock:
            _served += 1
            done = _served >= N_REQUESTS
        if done:
            log(f"已服务 {_served} 次请求，关闭服务")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, fmt, *args):
        log(f"remote={self.client_address[0]} " + fmt % args)


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["read", "fixed"], default="read")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    MODE = args.mode

    os.makedirs(WORK, exist_ok=True)
    if MODE == "read":
        # 每次 read 模式都重建 File A，保证内容确定、可校验
        with open(FILE_A, "w", encoding="utf-8") as f:
            f.write("content of file A (exp 3.3)\n")

    srv = ThreadingHTTPServer((HOST, args.port), Handler)
    log(f"listening http://{HOST}:{args.port}/  mode={MODE}")
    log(f"file_a={FILE_A}")
    log(f"等客户端 {N_REQUESTS} 次请求；收满自动退出，或 Ctrl+C 手动停")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("手动停止")
    finally:
        srv.server_close()
        log("已退出")


if __name__ == "__main__":
    main()