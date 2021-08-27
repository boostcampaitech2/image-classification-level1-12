import pandas as pd
import os, sys, random, tqdm
from torch import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch.nn.functional as F

sys.path.append('/opt/ml/repos/VIT-base/')
from util import dirlister, labeler
from dataloader import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn, AgeDataLoader, AgeDataset
from loss import FocalLoss, MaskLoss
from model import vision_transformer, vision_transformer_for_age


def pretrainer(model, dataloader, batch_size, mode, criterion, optimizer, device):
    total_metric, total_loss= 0, 0
    torch.cuda.empty_cache()
    for idx, (img, label) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)//batch_size + 1, leave=False, desc = f'    {mode}: '):
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)
        label_pred  = model(img)

        if mode == 'train':
            loss_sum = criterion(label_pred, label)
            loss_sum.backward()
            optimizer.step()
            total_loss += loss_sum.cpu().item()

        metric_score = f1_score(torch.argmax(label_pred, dim = -1).detach().cpu().numpy(), label.detach().cpu().numpy(), average="macro")
        total_metric += metric_score

        del img, label
        torch.cuda.empty_cache()

    return model, round(total_loss/idx, 3), round(total_metric/idx, 3)


def validater(model, dataloader, batch_size, mode, criterion, optimizer, device):
    total_metric, total_loss= 0, 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for idx, (img, label) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)//batch_size + 1, leave=False, desc = f'    {mode}: '):
            
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            label_pred  = model(img)


            loss_sum = criterion(label_pred, label)
            total_loss += loss_sum.cpu().item()

            metric_score = f1_score(torch.argmax(label_pred, dim = -1).detach().cpu().numpy(), label.detach().cpu().numpy(), average="macro")
            total_metric += metric_score

        del img, label
        torch.cuda.empty_cache()
        
    return model, round(total_loss/idx, 3), round(total_metric/idx, 3)


def pretrain_session(epoch, batch_size, lr, debug):
    PRETRAIN_DATA_ROOT = '/opt/ml/input/data/outdata/'
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    #dirlister
    over_60, under_60 = [], []
    for (root, dirs, files) in os.walk(PRETRAIN_DATA_ROOT):
        if int(root.split('/')[-1]) > 60:
            for file in files:
                over_60.append(os.path.join(root, file))
        else:
            for file in files:
                under_60.append(os.path.join(root, file))

    over_60_label = [1 for x in range(len(over_60))]
    under_60_label = [0 for x in range(len(under_60))]

    if len(over_60) != len(over_60_label) or len(under_60) != len(under_60_label):
        raise Exception
    
    dir_list = over_60 + under_60
    label_list = over_60_label + under_60_label
    criterion = MaskLoss()
    
    CV = StratifiedKFold(n_splits = 2, shuffle = True)
    train_idx, valid_idx  = next(iter(CV.split(dir_list, label_list)))
    
    train, valid = [dir_list[i] for i in train_idx], [dir_list[i] for i in valid_idx]
    t_label, val_label = [label_list[i] for i in train_idx], [label_list[i] for i in valid_idx]
    train_dataloader = AgeDataLoader(AgeDataset(train, t_label), batch_size, collate_fn=collate_fn)
    valid_dataloader = AgeDataLoader(AgeDataset(valid, val_label), batch_size, collate_fn=collate_fn)

    model = vision_transformer_for_age().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #스케쥴러 추가
    
    for e in range(epoch):
        print(f'--epoch {e + 1}/{epoch}--')
        model.train()
        torch.cuda.empty_cache()
        model, _loss, _metric = pretrainer(model, train_dataloader, batch_size, 'train', criterion, optimizer, device)
        print(f'    train epoch {e + 1} total loss: {_loss}, metric score: {_metric}')

        model.eval()
        torch.cuda.empty_cache()
        model, _loss, _metric = validater(model, valid_dataloader, batch_size, 'vaild', criterion, optimizer, device)
        print(f'    valid epoch {e + 1} total loss: {_loss}, metric score: {_metric}')

    return model