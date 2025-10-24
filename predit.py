import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import glob
from model.resnet import resnet18
import json


def build_class_names(json_path):
    """
    读取 json，反转 {k: v} 得到 {v: k}，并去掉 value 中可能出现的 \\u0000 字符。
    返回按类别索引顺序排列的列表 class_names。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)          # 假设 raw 是 {"0":"cat\u0000","1":"dog\u0000",...}

    # 1. 反转并清洗
    swapped = {}
    for k, v in raw.items():
        clean_k = k.replace('\u0000', '')
        swapped[int(v)] = clean_k

    # 2. 按索引排序得到 list
    max_idx = max(swapped.keys())
    class_names = [swapped[i] for i in range(max_idx + 1)]
    return class_names


if __name__ == '__main__':

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path   = 'model_data/best.pth'
    model_def     = 'model.model'   # 对应 model/model.py 中的 Net 类
    image_dir     = 'test'
    class_names   = build_class_names("datasets/char_dict_readable.json")

    model  = resnet18(pretrained=False, num_classes=7356).to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()

    tfm = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor()])

    image_list = glob.glob(os.path.join(image_dir, '*'))
    assert image_list, f'No image found in {image_dir}'

    img_path = image_list[0]
    img = Image.open(img_path).convert('RGB')
    x   = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        score, pred = torch.max(probs, dim=1)

    cls = class_names[pred.item()]
    print(f'Image: {img_path}')
    print(f'Predict: {cls}  (confidence={score.item():.3f})')