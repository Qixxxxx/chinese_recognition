import os
import sys
import json
import torch.backends.cudnn as cudnn
from utils.fit_function import fit_one_epoch
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from utils.loss_record import LossHistory

from model.AlexNet import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    log_dir = "./logs/"
    dataset_path = "datasets/"
    inputs_size = [224, 224, 3]
    batch_size = 128
    num_classes = 7356
    total_epochs = 200
    lr = 0.01
    pretrained = False  # 使用预训练权重
    weights_init = False
    Cuda = True

    if not pretrained:
        weights_init = True

    model = AlexNet(num_classes=num_classes, init_weights=weights_init)

    # # 导入以及训练好的权重
    # model_path = r"model_data\3090_result\ninet\81.55.pth"
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()  # 按state_dict导入权重
    # pretrained_dict = torch.load(model_path)
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    loss_history = LossHistory(log_dir)

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.to(device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "val"),
                                            transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)

    batch_num = train_num // batch_size
    batch_num_val = val_num // batch_size

    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    # lf = lambda x: ((1 + math.cos(x * math.pi / Epoch)) / 2) * (1 - 0.01) + 0.01  # cosine
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(total_epochs):
        fit_one_epoch(model, optimizer, epoch, batch_num, batch_num_val, train_loader, validate_loader, total_epochs, Cuda, loss_history)
        lr_scheduler.step()






