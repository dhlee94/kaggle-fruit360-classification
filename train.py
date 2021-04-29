import torch
import torchvision
from adamp import AdamP
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from FruitDataset import FruitDataset
import matplotlib.pyplot as plt
import numpy as np
import albumentations
import albumentations.pytorch
from model import MyModel
from torchsummary import summary as summary_
from utils import get_loader, get_train_functions, seed_everything, ComputeMetric, Update_Metric
import pandas as pd
from tqdm import tqdm

Device = 'cuda:0' if torch.cuda.is_available else 'cpu'

def train(cfg):
    DATA_PATH = cfg.values.data_path
    USE_KFOLD = cfg.values.kfold.use_kfold
    INPUT_IMAGES_SIZE = cfg.values.input_images_size
    SEED = cfg.values.seed
    EPOCH_NUM = cfg.values.train_args.num_epochs
    BATCH_SIZE = cfg.values.train_args.batch_size
    METRIC = cfg.values.metric
    WRITE_ITER_NUM = cfg.values.write_iter_num
    seed_everything(SEED)
    data_file = pd.read_csv(DATA_PATH)

    train_transform = albumentations.Compose([
        albumentations.GaussNoise(),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    val_transform = albumentations.Compose([
        albumentations.GaussNoise(),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    evaluation_metric = ComputeMetric(METRIC)

    if USE_KFOLD:
        kfold = StratifiedKFold(n_splits=cfg.values.kfold.n_splits)
        fold = 0

        for train_idx, val_idx in kfold.split(data_file, data_file['label'].values):
            fold += 1
            train_df = data_file.iloc[train_idx]
            val_df = data_file.iloc[val_idx]

            train_loader = get_loader(train_df, batch_size=BATCH_SIZE, transform=train_transform)
            val_loader = get_loader(val_df, batch_size=BATCH_SIZE, transform=val_transform)

            model = MyModel(cfg)
            model.to(Device)

            summary_(model, (3, 100, 100), batch_size=BATCH_SIZE)

            train_functions = [
                model.parameters(),
                cfg.values.train_args.optimizer,
                cfg.values.train_args.lr,
                cfg.values.train_args.weight_decay,
                cfg.values.train_args.scheduler,
                cfg.values.train_args.loss_fn
            ]

            optimizer, scheduler, criterion = get_train_functions(train_functions)

            for epoch in range(EPOCH_NUM):
                model.train()
                loss_val = 0
                matches = 0

                loss_values = Update_Metric()
                top1 = Update_Metric()
                top5 = Update_Metric()

                for idx, train_batch in enumerate(tqdm(train_loader)):
                    img, label = train_batch
                    img = img.float().to(Device)
                    label = label.long().to(Device)

                    output = model(img)
                    loss = criterion(output, label)
                    loss_val += loss.item()
                    pred = torch.argmax(output, -1)
                    matches += (pred == label).sum().item()

                    matches /= BATCH_SIZE

                    top1_err, top5_err = evaluation_metric.compute(output.data, label, topk=(1, 5))

                    loss_values.update(loss.item(), BATCH_SIZE)
                    top1.update(top1_err.item(), BATCH_SIZE)
                    top5.update(top5_err.item(), BATCH_SIZE)

                    if idx % WRITE_ITER_NUM == 0:
                        tqdm.write(f'NUM_KFold : {fold} '
                                   f'Epoch : {epoch + 1}/{EPOCH_NUM} {idx + 1}/{len(train_loader)} '
                                   f'Loss : {loss :.4f} 'f'Loss_val : {loss_values.val :.4f} 'f'Accuracy : {matches:.4f} '
                                   f'top1 Acr : {top1.val:.4f} '
                                   f'top5 Acr : {top5.val:.4f}')

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_val = 0
                    matches = 0


                with torch.no_grad():
                    model.eval()
                    for idx, val_batch in enumerate(val_loader):
                        img, label = val_batch
                        img = img.float().to(Device)
                        label = label.long().to(Device)
                        output = model(img)
                        loss = criterion(output, label)
                        pred = torch.argmax(loss, -1)
                        matches += (pred == label).sum().item()

                        top1_err, top5_err = evaluation_metric.compute(output, label, topk=(1, 5))
                        loss_values.update(loss.item(), BATCH_SIZE)
                        top1.update(top1_err.item(), BATCH_SIZE)
                        top5.update(top5_err.item(), BATCH_SIZE)

                    print(f'Validation Accuracy : {matches / (BATCH_SIZE * len(val_loader)):.4f}'
                          f'top1 Acr : {top1.avg:.4f} '
                          f'top5 Acr : {top5.avg:.4f}')




    else:
        tranin_df, val_df = train_test_split(data_file, test_size=cfg.vaklues.test_size, random_state=SEED)
        train_loader = get_loader(train_df, batch_size=BATCH_SIZE, transform=train_transform)
        val_loader = get_loader(val_df, batch_size=BATCH_SIZE, transform=val_transform)

        model = MyModel(cfg)
        model.to(Device)

        summary_(model, INPUT_IMAGES_SIZE, batch_size=batch_size)

        train_functions = [
            model.parameters(),
            cfg.values.train_args.optimizer,
            cfg.values.train_args.lr,
            cfg.values.train_args.weight_decay,
            cfg.values.train_args.scheduler,
            cfg.values.train_args.loss_fn
        ]

        optimizer, scheduler, loss_module = get_train_functions(train_functions)

        loss_val = 0
        matches = 0

        loss_values = Update_Metric()
        top1 = Update_Metric()
        top5 = Update_Metric()

        for epoch in range(EPOCH_NUM):
            model.train()
            for idx, train_batch in enumerate(train_loader):
                img, label = train_batch
                img = img.float().to(Device)
                label = label.long().to(Device)
                output = model(img)
                loss = criterion(output, label)
                loss_val += loss.item()
                pred = torch.argmax(loss, -1)
                matches += (pred == label).sum().item()

                loss_val /= BATCH_SIZE
                matches /= BATCH_SIZE

                top1_err, top5_err = evaluation_metric.compute(output, label, topk=(1, 5))

                loss_values.update(loss.item(), BATCH_SIZE)
                top1.update(top1_err.item(), BATCH_SIZE)
                top5.update(top5_err.item(), BATCH_SIZE)

                if idx % WRITE_ITER_NUM == 0:
                    tqdm.write(f'NUM_KFold : {fold} '
                               f'Epoch : {epoch + 1}/{EPOCH_NUM} {idx + 1}/{len(train_loader)} '
                               f'Loss : {loss :.4f} 'f'Loss_val : {loss_values.val :.4f} 'f'Accuracy : {matches:.4f} '
                               f'top1 Acr : {top1.val:.4f} '
                               f'top5 Acr : {top5.val:.4f}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = 0
                matches = 0

            scheduler.step()

            with torch.no_grad():
                model.eval()
                for idx, val_batch in enumerate(val_loader):
                    img, label = val_batch
                    img = img.float().to(Device)
                    label = label.long().to(Device)
                    output = model(img)
                    loss = criterion(output, label)
                    pred = torch.argmax(loss, -1)
                    matches += (pred == label).sum().item()

                    top1_err, top5_err = evaluation_metric.compute(output, label, topk=(1, 5))
                    loss_values.update(loss.item(), BATCH_SIZE)
                    top1.update(top1_err.item(), BATCH_SIZE)
                    top5.update(top5_err.item(), BATCH_SIZE)

                print(f'Validation Accuracy : {matches / (BATCH_SIZE * len(val_loader)):.4f}'
                      f'top1 Acr : {top1.avg:.4f} '
                      f'top5 Acr : {top5.avg:.4f}')


