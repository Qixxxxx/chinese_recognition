import os
import sys
import json
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from utils.loss import LossHistory

from model import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    log_dir = "logs/"
    dataset_path = "datasets/really_smoke_with_mask/"
    inputs_size = [256, 256, 3]
    num_classes = 2
    pretrained = False  # 使用预训练权重
    weights_init = False
    Cuda = True

    if not pretrained:
        weights_init = True

    model = AlexNet(num_classes=num_classes, init_weights=weights_init).train()

    # 导入以及训练好的权重
    model_path = r"model_data\3090_result\ninet\81.55.pth"
    print('Loading weights into state dict...')
    model_dict = model.state_dict()  # 按state_dict导入权重
    pretrained_dict = torch.load(model_path)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    loss_history = LossHistory(log_dir)

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
