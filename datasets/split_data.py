import os
from shutil import copy
import random


def mk_file(file_path: str):
    if not os.path.exists(file_path):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(file_path)


def main():
    random.seed(0)   # 随机种子

    split_rate = 0.1   # 将数据集中10%的数据划分到验证集中

    data_root = os.getcwd()  # 获取当前工作目录的路径
    origin_data_path = os.path.join(data_root, "HWDB-1-train")
    assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)

    classes = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in classes:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in classes:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in classes:
        cla_path = os.path.join(origin_data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()