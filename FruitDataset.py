import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from glob import glob
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

class FruitDataset(Dataset):
    def __init__(self,  df, transform=None):
        super(FruitDataset, self).__init__()
        self.df = df.reset_index()
        self.image_path = self.df.images
        self.labels = self.df.label
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = Image.open(self.image_path[item])
        label = self.labels[item]
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return img, int(label)