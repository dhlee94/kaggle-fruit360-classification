import torch
import torch.nn as nn
import timm

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.model = timm.create_model(cfg.values.model_arc, pretrained=True, num_classes=cfg.values.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

