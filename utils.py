import torch
import numpy as np
import os
from FruitDataset import FruitDataset
import yaml
import json
from easydict import EasyDict
import random
from torch.utils.data import Dataset, DataLoader
from importlib import import_module
from adamp import AdamP
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss

class AddParserManager:
    def __init__(self, cfg_file_path, cfg_name, is_json=False):
        super().__init__()
        self.values = EasyDict()
        if cfg_file_path:
            self.config_file_path = cfg_file_path
            self.config_name = cfg_name
            self.reload()
    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, in_dict):
        for (key, value) in in_dict.item():
            if isinstance(value, dict):
                for (key2, value2) in value.item():
                    if isinstance(value2, dict):
                        for (key3, value3) in value2.item():
                            self.values[key][key2][key3] = value3

                    else:
                        self.values[key][key2] = value2

            else:
                self.values[key] = value

    def export(selfself, save_cfg_path):
        if save_cfg_path:
            with open(save_cfg_path, 'w') as f:
                yaml.dump(dict(self.values), f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_train_functions(all_train_parse):
    param, optimizer_name, lr, weight_decay, scheduler_name, loss_fn = all_train_parse

    optimizer_module = getattr(import_module('adamp'), optimizer_name)
    scheduler_module = getattr(import_module('torch.optim.lr_scheduler'), scheduler_name)
    loss_module = getattr(import_module('torch.nn'), loss_fn)

    optimizer = optimizer_module(param, lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_module(optimizer, T_max=50, eta_min=0)

    return optimizer, scheduler, loss_module()

def get_loader(df, batch_size, transform, shuffle=True):
    dataset = FruitDataset(df, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

class ComputeMetric(object):
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric

    def Metric_accuracy(self, output, labels, topk=(1, 5)):
        max_k = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        matches = pred.eq(labels.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            matches_k = matches[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(matches_k.mul_(100.0 / batch_size))

        return result

    def compute(self, output, labels, topk=(1, 5)):
        if self.metric == 'accuracy':
            out = self.Metric_accuracy(output=output, labels=labels, topk=topk)

        return out


class Update_Metric(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count