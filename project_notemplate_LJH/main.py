import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import datetime

import sys
sys.path.append('/opt/ml/repos/project_notemplete_LJH/')

from cust_util.util import dirlister, CV, to_label
from dataloader import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn, test_collate_fn
import loss, metric, Model
from loss import MaskLoss
from finetune import tuner

# fix random seeds for reproducibility
SEED = 32
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
epoch = 2
batch_size = 32
lr = 1e-5
cv_num = 3
freeze = False
debug = False
addition = 'to_layer_one'

print(f'\nDEBUG: {debug}')
print(f'estimated end time: {datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))) + datetime.timedelta(minutes=10*(epoch*cv_num + 1) + 3)}')

train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'), nrows = 100 if debug else None)
sub_meta = pd.read_csv(os.path.join(SUB_DATA_ROOT, 'info.csv'), nrows = 100 if debug else None)

train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta, mode = 'train')
sub_dir_list = dirlister(SUB_DATA_ROOT, sub_meta, mode = 'sub')
train_cv = CV(train_dir_list, cv_num)

total_dataloader = MaskDataLoader(MaskDataset(train_dir_list, mode = 'train'), batch_size, collate_fn=collate_fn)
sub_dataloader = MaskDataLoader(MaskDataset(sub_dir_list, mode = 'sub', shuffle = False), batch_size, collate_fn=sub_collate_fn)

met = metric.accuracy
criterion = loss.MaskLoss()
# criterion = nn.BCEWithLogitsLoss()  #시그모이드가 로스에 추가됨

cv_train_loss, cv_valid_loss= [], []
cv_test_acc, cv_test_loss = [], []
for cv_idx, (train, valid, test) in enumerate(train_cv):

    print(f'\nstart fold {cv_idx + 1}/{train_cv.maxfold}')
    test_dataloader = MaskDataLoader(MaskDataset(test, mode = 'train'), batch_size, collate_fn=test_collate_fn)
    train_dataloader = MaskDataLoader(MaskDataset(train, mode = 'train'), batch_size, collate_fn=collate_fn)
    valid_dataloader = MaskDataLoader(MaskDataset(valid, mode = 'train'), batch_size, collate_fn=test_collate_fn)

    model = Model.MaskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #train with freeze
    for e in range(epoch):

        print(f'    ---epoch{e + 1}---')
        torch.cuda.empty_cache()
        model.train()
        total_metric = 0
        total_loss = 0
        for idx, (img, label) in tqdm(enumerate(train_dataloader), total = len(train_dataloader)//batch_size + 1, leave=False, desc = '    train: '):
            
            optimizer.zero_grad()

            img, label = img.to(device), label.to(device)
            label_pred  = model(img)

            loss_sum = criterion(label_pred, label)
            loss_sum.backward()
            optimizer.step()

            total_loss += loss_sum.cpu().item()
            metric_score = met(label_pred, label)
            total_metric += metric_score

        print(f'    train epoch {e + 1} total loss: {total_loss/ (idx + 1)}, metric score: {total_metric / (idx + 1)}')
        cv_train_loss.append(total_loss/ (idx + 1))
        torch.cuda.empty_cache()

        model.eval()
        total_acc= 0
        total_loss = 0
        for idx, (img, label) in tqdm(enumerate(valid_dataloader), total = len(valid_dataloader)//batch_size + 1, leave = False, desc = '    valid: '):
            with torch.no_grad():
                img, label = img.to(device), label.to(device)

                label_pred  = model(img)
                loss_sum = criterion(label_pred, label)
                total_loss += loss_sum.cpu().item()
                total_acc += met(label_pred, label)

        print(f'    valid epoch {e + 1} total loss: {total_loss/ (idx + 1)}, metric score: {total_acc / (idx + 1)}')
        cv_valid_loss.append(total_loss/ (idx + 1))

    del img, label
    torch.cuda.empty_cache()

    ##evaluate
    metric_score = 0
    total_loss = 0
    model.eval()
    for idx, (img, label) in tqdm(enumerate(test_dataloader), total = len(test_dataloader)//batch_size + 1, leave=False, desc = '    test: '):
        with torch.no_grad():
            img, label= img.to(device), label.to(device)
            label_pred  = model(img)

            loss_sum = criterion(label_pred, label)
            total_loss += loss_sum.cpu().item()
            metric_score+= met(label_pred, label)

            
    idx += 1
    cv_test_acc.append(metric_score)
    cv_test_loss.append([total_loss/ (idx)])

    print(f'    fold {cv_idx + 1} test total loss: {total_loss/ (idx)}, metric score: {metric_score / idx}')  


#finetune
torch.cuda.empty_cache()
model = Model.MaskModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = loss.MaskLoss()

for e in range(epoch):
    torch.cuda.empty_cache()
    model.train()
    for idx, (img, label) in tqdm(enumerate(train_dataloader), total = len(total_dataloader)//batch_size + 1, leave=False, desc = f'    finetune {e}: '):
        optimizer.zero_grad()

        img, label = img.to(device), label.to(device)
        label_pred  = model(img)

        loss_sum = criterion(label_pred, label)
        loss_sum.backward()
        optimizer.step()


#make submission
print('start submission')
out = []
model.eval()
for idx, (img) in tqdm(enumerate(sub_dataloader), total = len(sub_dir_list)//batch_size + 1, leave = False):
    with torch.no_grad():
        img = img.to(device)
        labels = model(img)

        for label in labels:
            out.append(torch.argmax(label, dim = -1))

out_csv = sub_meta.copy()
out_csv['ans'] = out
savedir = '/opt/ml/repos/project_notemplate_LJH/results/'
filename =f'{model.name}_epoch{epoch}_lr{lr}_batchsize{batch_size}_freeze{freeze}_{addition}.csv'
torch.save(model.state_dict(), savedir + f'_{addition}_weight.pt')

print(out_csv.head())
out_csv.to_csv(os.path.join(savedir, filename), index = False)

print('complete')