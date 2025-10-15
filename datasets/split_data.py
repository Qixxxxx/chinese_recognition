#!/usr/bin/env python3
"""
从 train/ 中按 20 取 1 的规则抽样子图，目录结构不变，生成 train_1/
用法:  python sample_train.py  [--src ROOT] [--dst ROOT] [--num num] [-j NPROC]
"""
import os
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

def iter_subdirs(root):
    """按数字顺序遍历 00000...07355 共 7356 个子目录"""
    for idx in range(7356):
        yield os.path.join(root, f"{idx:05d}")

def copy_num(subdir_src, dst_root, num=20):
    """
    单个目录的处理函数，返回 (src_dir, dst_dir, copied_files)
    """
    subdir_name = os.path.basename(subdir_src)
    subdir_dst = os.path.join(dst_root, subdir_name)
    os.makedirs(subdir_dst, exist_ok=True)

    # 只拿 png，按文件名排序保证顺序一致
    pngs = sorted(glob(os.path.join(subdir_src, "*.png")))
    selected = pngs[:num]
    for fp in selected:
        shutil.copy2(fp, subdir_dst) # 保留元数据
    return subdir_src, subdir_dst, len(selected)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="train", help="源根目录")
    parser.add_argument("--dst", default="train_1", help="目标根目录")
    parser.add_argument("--num", type=int, default=100, help="复制数量")
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="并行进程数，默认 CPU 核数")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise FileNotFoundError(f"源目录 {args.src} 不存在")

    os.makedirs(args.dst, exist_ok=True)

    # 准备任务列表
    tasks = [(d, args.dst, args.num) for d in iter_subdirs(args.src)]

    # 多进程映射 + 进度条
    with Pool(args.jobs) as pool:
        results = list(tqdm(
            pool.starmap(copy_num, tasks),
            total=len(tasks),
            desc="Sampling"
        ))

    # 简单统计
    total_copied = sum(cnt for _, _, cnt in results)
    print(f"完成！共复制 {total_copied} 张图片到 {args.dst}")

if __name__ == "__main__":
    main()