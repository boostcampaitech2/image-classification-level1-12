import pandas as pd
import os, sys, random, tqdm
from torch import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append('/opt/ml/repos/VIT-base/')
from util import dirlister, CV
from dataloader import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn
from metric import accuracy
from loss import Maskloss
from model import Maskmodel

def sub_session(batch_size, debug):
    raise NotImplementedError
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
    sub_meta = pd.read_csv(os.path.join(SUB_DATA_ROOT, 'info.csv'), nrows = 100 if debug else None)
    sub_dir_list = dirlister(SUB_DATA_ROOT, sub_meta, mode = 'sub')
    sub_dataloader = MaskDataLoader(MaskDataset(sub_dir_list, mode = 'sub', shuffle = False), batch_size, collate_fn=sub_collate_fn)

def trainer(model, dataloader, batch_size, mode, criterion, optimizer, metric, device):
    total_metric, total_loss= 0, 0
    torch.cuda.empty_cache()
    for idx, (img, label) in tqdm(enumerate(dataloader), total = len(dataloader)//batch_size + 1, leave=False, desc = f'    {mode}: '):
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)
        label_pred  = model(img)

        if mode == 'train':
            loss_sum = criterion(label_pred, label)
            loss_sum.backward()
            optimizer.step()

        total_loss += loss_sum.cpu().item()
        metric_score = metric(label_pred, label)
        total_metric += metric_score

    return model, round(total_loss/idx, 3), round(total_metric/idx, 3)


def train_session(epoch, batch_size, lr, debug):
    TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'), nrows = 100 if debug else None)
    train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta, mode = 'train')
    
    met = accuracy
    criterion = Maskloss()
    
    train, valid = StratifiedKFold(n_splits = 1, shuffle = True)(train_dir_list)
    train_dataloader = MaskDataLoader(MaskDataset(train, mode = 'train'), batch_size, collate_fn=collate_fn)
    valid_dataloader = MaskDataLoader(MaskDataset(valid, mode = 'train'), batch_size, collate_fn=collate_fn)

    model = Maskmodel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #스케쥴러 추가
    
    for e in epoch:
        print(f'--epoch {e + 1}/{epoch}--')
        model.train()
        torch.cuda.empty_cache()
        model, _loss, _metric = trainer(model, train_dataloader, batch_size, 'train', criterion, optimizer, met, device)
        print(f'    train epoch {e + 1} total loss: {_loss/ (e + 1)}, metric score: {_metric / (e + 1)}')

        model.eval()
        torch.cuda.empty_cache()
        model, _loss, _metric = trainer(model, valid_dataloader, batch_size, 'vaild', criterion, optimizer, met, device)
        print(f'    valid epoch {e + 1} total loss: {_loss/ (e + 1)}, metric score: {_metric / (e + 1)}')

    return model
    