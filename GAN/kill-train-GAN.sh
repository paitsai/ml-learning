#!/bin/bash

# 获取所有正在运行的 python GAN.py 进程的 PID
pids=$(pgrep -f "python GAN.py")

# 检查是否有进程需要杀死
if [ -z "$pids" ]; then
    echo "没有找到正在运行的 python GAN.py 进程。"
else
    echo "正在杀死以下进程: $pids"
    # 杀死所有找到的进程
    kill -9 $pids
    echo "已成功杀死进程。"
fi