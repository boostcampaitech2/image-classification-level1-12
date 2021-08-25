import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

import sys
sys.path.append('/opt/ml/repos/project_notemplete_LJH/')

from cust_util.util import dirlister, CV, to_label
from dataloader import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn
import loss, metric, Model

device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
sub_meta = pd.read_csv(os.path.join(SUB_DATA_ROOT, 'info.csv'))

train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta)
train_cv = CV(train_dir_list, 5)

sub_dir_list = dirlister(SUB_DATA_ROOT, sub_meta, mode = 'sub')
sub_dataloader = MaskDataLoader(MaskDataset(sub_dir_list, meta = sub_meta, mode = 'sub'), 1, collate_fn=sub_collate_fn)

#total_dataloader = MaskDataLoader(MaskDataset(train_dir_list, meta = train_meta), 1000, collate_fn=collate_fn)



acc = metric.accuracy
criterion = loss.F1_loss
# criterion = nn.BCEWithLogitsLoss()  #시그모이드가 로스에 추가됨
epoch = 1
batch_size = 64
cv_train_loss, cv_valid_loss, cv_train_acc, cv_valid_acc= [], [], [], []
cv_test_acc, cv_test_loss = [], []

for idx, (train, valid, test) in enumerate(train_cv):
    print(f'start fold {idx + 1}/{train_cv.maxfold}')
    
    test_dataloader = MaskDataLoader(MaskDataset(test, meta = train_meta), batch_size, collate_fn=collate_fn)
    train_dataloader = MaskDataLoader(MaskDataset(train, meta = train_meta), batch_size, collate_fn=collate_fn)
    valid_dataloader = MaskDataLoader(MaskDataset(valid, meta = train_meta), batch_size, collate_fn=collate_fn)

    model = Model.MaskModel().to(device)
    for param in model.backbone.parameters():
            param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    #train with freeze
    lr = 0.05
    train_loss, valid_loss = [[],[],[]], [[],[],[]]
    train_acc, valid_acc = [[],[],[]], [[],[],[]]

    

    for e in range(epoch):
        print(f'---epoch{e}---')
        model.train()
        for idx, (img, age, gender, mask) in tqdm(enumerate(train_dataloader), total = len(train)//batch_size + 1):
            optimizer.zero_grad()
            img, age, gender, mask = img.to(device), age.to(device),gender.to(device),mask.to(device)

            age_pred, gender_pred, mask_pred  = model(img)
            loss_age, loss_mask, loss_gender = criterion(age_pred, age), criterion(mask_pred, mask), criterion(gender_pred, gender)
            acc_age, acc_mask, acc_gender = acc(age_pred, age), acc(mask_pred, mask), acc(gender_pred, gender)

            loss_sum = loss_age+loss_mask+loss_gender
            loss_sum.backward()
            optimizer.step()

            train_loss[0].append(loss_age.item())
            train_loss[1].append(loss_mask.item())
            train_loss[2].append(loss_gender.item())

            train_acc[0].append(acc_age)
            train_acc[1].append(acc_mask)
            train_acc[2].append(acc_gender)
        print(f'    train epoch {e} total loss: {loss_sum.item()}, age_acc: {acc_age}, mask_acc: {acc_mask}, gender_acc = {acc_gender}')


        model.eval()
        for idx, batch in valid_dataloader:
            
            with torch.no_grad():
                age_pred, gender_pred, mask_pred  = model(img)
                loss_age, loss_mask, loss_gender = criterion(age_pred, age), criterion(mask_pred, mask), criterion(gender_pred, gender)
                acc_age, acc_mask, acc_gender = acc(age_pred, age), acc(mask_pred, mask), acc(gender_pred, gender)
                loss_sum = loss_age+loss_mask+loss_gender

                valid_loss[0].append(loss_age.item())
                valid_loss[1].append(loss_mask.item())
                valid_loss[2].append(loss_gender.item())

                valid_acc[0].append(acc_age.item())
                valid_acc[1].append(acc_mask.item())
                valid_acc[2].append(acc_gender.item())
            
        print(f'    valid epoch {e} total loss: {loss_sum.item()}, age_acc: {acc_age.item()}, mask_acc: {acc_mask.item()}, gender_acc = {acc_gender.item()}')
    
    cv_train_loss.append(train_loss)
    cv_valid_loss.append(valid_loss)
    cv_train_acc.append(train_acc)
    cv_valid_acc.append(valid_acc)

    ##evaluate
    acc_age, acc_mask, acc_gender = 0,0,0
    model.eval()
    for idx, batch in test_dataloader:
        age_pred, gender_pred, mask_pred  = model(img)
        loss_age, loss_mask, loss_gender = criterion(age_pred, age), criterion(mask_pred, mask), criterion(gender_pred, gender)
        acc_age+= acc(age_pred, age)
        acc_mask +=acc(mask_pred, mask)
        acc_gender +=acc(gender_pred, gender)
        loss_sum = loss_age+loss_mask+loss_gender

    idx += 1
    cv_test_acc.append([acc_age.item()/idx, acc_mask.item()/idx, acc_gender.item()/idx])
    cv_test_loss.append([loss_sum.item(), loss_age.item(), loss_mask.item(), loss_gender.item()])

    print(f'    test epoch {e} total loss: {loss_sum.item()}, average_age_acc: {acc_age.item()}, average_mask_acc: {acc_mask.item()}, average_gender_acc = {acc_gender.item()}')  
        

out = []
for batch in sub_dataloader:
    age, gender, mask = model(batch)
    out.append(to_label(mask, age, gender))

out_csv = sub_meta.copy()
out_csv['ans'] = out
out_csv.to_csv('/opt/ml/repos/project_LJH/submission.py', index = False)
print('complete')