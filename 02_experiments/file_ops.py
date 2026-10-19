"""
阶段 3.1 被测程序：制造 6 种可预期的文件行为

运行：
    cd C:\\Users\\ww\\Desktop\\PIDS
    .\\.venv\\Scripts\\python.exe 02_experiments\\file_ops.py
"""

import os
import time
import shutil

HEAR = os.path.dirname(os.path.abspath(__file__))    #获取当前脚本地址
WORK = os.path.join(HEAR, "exp3_1_work")    #拼接工作目录路径

SRC = os.path.join(HEAR,"test_a.text")        #主文件
COPY = os.path.join(WORK,"test_a_copy.text")    #复制文件
RENAMED = os.path.join(WORK,"test_a_renamed.text")    #重命名文件

LINE1 = "PIDS experiment 3.1 line one\n"
LINE2 = "PIDS experiment 3.1 line two\n"
LINE3 = "PIDS experiment 3.1 line three (appended)\n"

def mark(tag):
    print(f"[{time.strftime('%H:%M:%S')}] ===={tag}====")

def create_file():
    """创建SRC,写入两行"""
    with open(SRC, "w", encoding = "utf-8") as f:
        f.write(LINE1)
        f.write(LINE2)

def read_file():
    """读取SRC所有行"""
    with open(SRC,"r",encoding = "utf-8") as f:
        content = f.read()
    print((f"读到{len(content)}个字符"))

def append_file():
    """向 SRC 追加一行。"""
    with open (SRC, "a" , encoding= "utf-8" ) as f:
        f.write(LINE3)

def copy_file():
    """复制 SRC 到 COPY"""
    shutil.copy(SRC, COPY)

def rename_file():
    """重命名 COPY 为 RENAMED"""
    os.rename(COPY, RENAMED)

def delete_file():
    """删除 RENAMED"""
    os.remove(RENAMED)

def main():
    os.makedirs(WORK, exist_ok=True)
    print(f"被测进程 PID = {os.getpid()} PPID = {os.getppid()}")
    print(f"工作目录 = {WORK}")
    print("-----------------")

    for tag, fn in [
        ( "1 创建并写入" , create_file),
        ( "2 读取" , read_file),
        ( "3 追加" , append_file),
        ( "4 复制" , copy_file),
        ( "5 重命名" , rename_file),
        ( "6 删除" , delete_file),
    ]:
        mark(tag)
        fn()
        time.sleep(1.5)     #暂停程序执行,拉开间隔
    
    print("全部完成")

if __name__ == "__main__":
    main()

