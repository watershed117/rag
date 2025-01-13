import os
import argparse
import ast
# import psutil
import time


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="处理路径参数")

parser.add_argument("--dir", type=str, help="输入路径")

parser.add_argument("--pid", type=str, help="pid")

# 解析参数
args = parser.parse_args()


def delete_directory(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径 {path} 不存在")
        return

    # 遍历目录内容
    for item in os.listdir(path):
        item_path = os.path.join(path, item)  # 获取完整路径

        # 如果是目录，递归删除
        if os.path.isdir(item_path):
            delete_directory(item_path)  # 递归删除子目录
        else:
            # 如果是文件，直接删除
            os.remove(item_path)
            print(f"已删除文件: {item_path}")

    # 删除空目录
    os.rmdir(path)
    print(f"已删除目录: {path}")


# def is_process_alive(pid):
#     try:
#         # 获取进程对象
#         process = psutil.Process(pid)
#         # 检查进程状态
#         return True
#     except psutil.NoSuchProcess:
#         # 如果进程不存在
#         return False

# if args.pid:
#     while True:
#         if not is_process_alive(int(args.pid)):
#             break
#         time.sleep(1)
for path in ast.literal_eval(args.dir):
    delete_directory(path)
