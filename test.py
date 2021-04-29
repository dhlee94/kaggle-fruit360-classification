import torch
import torchvision
from adamp import AdamP
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from FruitDataset import FruitDataset
import matplotlib.pyplot as plt
import numpy as np
import albumentations
import albumentations.pytorch
from model import MyModel
from torchsummary import summary as summary_
import pandas as pd
from tqdm import tqdm
Device = "cuda:0" if torch.cuda.is_available else "cpu"

Batch_size = 8
test_transforms = albumentations.Compose([
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2()
])

data_frame = pd.read_csv("./test_data.csv")
dataset = FruitDataset(data_frame=data_frame, transform=test_transforms)
test_data_loader = DataLoader(dataset, batch_size=Batch_size, shuffle=False)

model = MyModel()
models = []

path = []
for i in range(3):
    path.append("./checkpoint_" + str(i) + ".pth")
    models.append(model)

for i in range(3):
    models[i].load_state_dict(torch.load(path[i]))
    models[i].to(Device)
    models[i].eval()

with torch.no_grad():
    test_acc = 0
    outputs = []
    for idx, test_batch in enumerate(tqdm(test_data_loader)):
        acc = 0
        img, label = test_batch
        img = img.float().to(Device)
        label = label.long().to(Device)

        for idx, model in enumerate(models):
            if idx == 0:
                outputs = model(img)
                outputs = torch.reshape(outputs, (1, Batch_size, -1))
            else:
                output = model(img)
                output = torch.reshape(output, (1, Batch_size, -1))
                outputs = torch.cat([outputs, output], dim=0)

        output = torch.mean(outputs, 0)
        output = torch.reshape(output, (Batch_size, -1))
        pred = torch.argmax(output, -1)
        acc += (pred == label).sum().item()
        test_acc += acc / Batch_size

print(f'Test Dataset Accuracy : {test_acc/(len(test_data_loader)):.4f}')