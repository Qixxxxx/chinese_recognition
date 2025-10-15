import os
import random
import shutil
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

random.seed(0)
split_rate = 0.1      # 10 % 做验证集
data_root = os.getcwd()
origin_data_path = os.path.join(data_root, "HWDB-1-train")
assert os.path.exists(origin_data_path), f"path '{origin_data_path}' does not exist."

train_root = os.path.join(data_root, "train")
val_root   = os.path.join(data_root, "val")


def mk_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy(src, dst_dir):
    """把文件 src 拷贝到目录 dst_dir 下（保持文件名）"""
    shutil.copy(src, dst_dir)


# ---------- 单类别处理函数（子进程跑） ----------
def process_one_class(args):
    cla, origin_path, train_dir, val_dir, split_rate = args
    random.seed(0)          # 每个子进程再设一次，保证可重复

    cla_path = os.path.join(origin_path, cla)
    images   = os.listdir(cla_path)
    num      = len(images)
    eval_imgs = set(random.sample(images, k=int(num * split_rate)))

    for img in images:
        src = os.path.join(cla_path, img)
        if img in eval_imgs:
            dst = val_dir
        else:
            dst = train_dir
        copy(src, dst)
    return cla, num

if __name__ == '__main__':
    # 1. 获取类别
    classes = [c for c in os.listdir(origin_data_path)
               if os.path.isdir(os.path.join(origin_data_path, c))]

    # 2. 建目录
    mk_file(train_root)
    mk_file(val_root)
    for c in classes:
        mk_file(os.path.join(train_root, c))
        mk_file(os.path.join(val_root, c))

    # 3. 组装任务
    tasks = [(c, origin_data_path,
              os.path.join(train_root, c),
              os.path.join(val_root, c),
              split_rate) for c in classes]

    # 4. 多进程
    cpu = cpu_count()
    with Pool(processes=cpu) as pool:
        # tqdm 实时显示进度
        for cla, num in tqdm(pool.imap_unordered(process_one_class, tasks),
                             total=len(tasks), desc="Processing"):
            print(f"\n{cla}: {num} images done")

    print("processing done!")