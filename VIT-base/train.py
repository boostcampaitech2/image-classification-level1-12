import pandas as pd
import os, sys, random, tqdm
from torch import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch.nn.functional as F
from statistics import mean

sys.path.append('/opt/ml/repos/VIT-base/')
from util import dirlister, labeler, stratified_CV
from dataloader import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn
from loss import FocalLoss, MaskLoss
from model import vision_transformer

def sub_session(model, epoch, batch_size, debug):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
    sub_meta = pd.read_csv(os.path.join(SUB_DATA_ROOT, 'info.csv'), nrows = 100 if debug else None)
    sub_dir_list = dirlister(SUB_DATA_ROOT, sub_meta, mode = 'sub')
    sub_dataloader = MaskDataLoader(MaskDataset(sub_dir_list, mode = 'sub', shuffle = False), batch_size, collate_fn=sub_collate_fn)

    labels = []
    with torch.no_grad():
        for idx, (img) in tqdm.tqdm(enumerate(sub_dataloader), total = len(sub_dataloader)//batch_size + 1, leave=False, desc = f'    submission: '):
            label = model(img.to(device))
            labels.append(torch.argmax(label, dim = -1).tolist())

    sub_meta.ans = sum(labels, [])
    return sub_meta

def trainer(model, dataloader, batch_size, mode, criterion, optimizer, device):
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

def fit_session(epoch, batch_size, lr, debug):
    print('\nstart fitting')
    TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'), nrows = 100 if debug else None)
    train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta, mode = 'train')
    
    criterion = FocalLoss()
    full_dataset = MaskDataset(train_dir_list, mode = 'train')
    full_dataloader = MaskDataLoader(full_dataset, batch_size, collate_fn=collate_fn)
    model = vision_transformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        torch.cuda.empty_cache()
        model, _loss, _metric = trainer(model, full_dataloader, batch_size, f'fit fold {e}', criterion, optimizer, device)

    return model


def train_session(epoch, batch_size, lr, debug, cv_num)->None:
    TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'), nrows = 100 if debug else None)
    train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta, mode = 'train')
    
    criterion = FocalLoss()
    full_dataset = MaskDataset(train_dir_list, mode = 'train', shuffle = False)

    total_loss, total_metric = [], []
    for idx, (train, valid, test) in enumerate(stratified_CV(full_dataset.dir_list, full_dataset.label, cv_num)):

        model = vision_transformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataloader = MaskDataLoader(MaskDataset(train, mode = 'train'), batch_size, collate_fn=collate_fn)
        valid_dataloader = MaskDataLoader(MaskDataset(valid, mode = 'train'), batch_size, collate_fn=collate_fn)
        test_dataloader = MaskDataLoader(MaskDataset(test, mode = 'train'), batch_size, collate_fn=collate_fn)
        print(f'fold {idx + 1}')
        for e in range(epoch):
            print(f'    --epoch {e + 1}/{epoch}--')
            model.train()
            torch.cuda.empty_cache()
            model, _loss, _metric = trainer(model, train_dataloader, batch_size, 'train', criterion, optimizer, device)
            print(f'        train epoch {e + 1} total loss: {_loss}, metric score: {_metric}')

            model.eval()
            torch.cuda.empty_cache()
            model, _loss, _metric = validater(model, valid_dataloader, batch_size, 'vaild', criterion, optimizer, device)
            print(f'        valid epoch {e + 1} total loss: {_loss}, metric score: {_metric}')

        model, _loss, _metric = validater(model, test_dataloader, batch_size, 'test', criterion, optimizer, device)
        total_loss.append(_loss), total_metric.append(_metric)
        print(f'fold{idx + 1} total loss: {_loss}, metric score: {_metric}')

    print(f'\nloss mean: {mean(total_loss)}, metric mean: {mean(total_metric)}')

    