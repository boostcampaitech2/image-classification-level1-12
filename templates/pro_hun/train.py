import argparse
import datetime as dt
import math
import os
import random
import time
from functools import partial
from pathlib import Path
import wandb

import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from PIL import Image
from pytz import timezone
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torchvision.models.resnet import resnet50, resnet152
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision.transforms.transforms import GaussianBlur, RandomRotation, RandomHorizontalFlip, GaussianBlur

from data_preprocessing.data_split import Run_Split
from model.loss import batch_loss
from model.metric import batch_acc, batch_f1, epoch_mean
from model.model import resnet_finetune, efficient_model
from utils.util import ensure_dir, notification, prepare_device, fix_randomseed


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class Mask_Dataset(object):
    def __init__(self, transforms, name, df, path, folder):
        self.transforms = transforms
        self.name = name
        self.path = path
        self.folder = folder
        self.imgs = sorted(
            os.listdir(os.path.join(self.path, f"{self.folder}/{self.name}_image"))
        )
        self.df = df


    def __getitem__(self, idx):
        img_path = self.df["path"][idx]
        target = self.df["label"][idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image = img)
            img = augmented['image']


        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(project='Mask_classification', entity='herjh0405')
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0001,
        type=float,
        help="learning rate for training",
    )
    args.add_argument(
        "-bs", "--batch_size", default=128, type=int, help="batch size for training"
    )
    args.add_argument("--epoch", default=10, type=int, help="training epoch size")
    args.add_argument("--fold_size", default=5, type=int, help="StratifiedKFold size")
    args.add_argument(
        "--train_path",
        default="/opt/ml/image-classification-level1-12/templates/data/train",
        type=str,
        help="train_path",
    )
    args.add_argument(
        "--model_save",
        default="/opt/ml/image-classification-level1-12/templates/pro_hun/output/model_save",
        type=str,
        help="model_save_path",
    )
    args.add_argument(
        "--normalize_mean",
        default=(0.5601, 0.5241, 0.5014),
        type=float,
        help="Normalize mean value",
    )
    args.add_argument(
        "--normalize_std",
        default=(0.2331, 0.2430, 0.2456),
        type=float,
        help="Normalize std value",
    )
    # Original:train_with_label.csv, Crop:train_with_crop.csv
    args.add_argument(
        "--image_data",
        default="train_with_label.csv",
        type=str,
        help="Use Original or Original+Crop",
    )
    # Original:image_folder, Crop:image_crop_all
    args.add_argument("--image_folder", default="image_all", type=str, help="Split_image folder",)

    args = args.parse_args()
    config = wandb.config
    config.learning_rate = args.learning_rate
    fix_randomseed(12)

    train_path = args.train_path
    train_label = pd.read_csv(os.path.join(train_path, args.image_data))
    run_split = Run_Split(os.path.join(train_path, args.image_folder))
    fold_num = args.fold_size
    train_list, val_list = run_split.train_val_split(train_label, fold_num)

    # GPU가 사용가능하면 사용하고, 아니면 CPU 사용
    device = prepare_device()
    print(f"{device} is using!")

    data_transform = albumentations.Compose([
        albumentations.Resize(512, 384, cv2.INTER_LINEAR),
        albumentations.GaussianBlur(3, sigma_limit=(0.1, 2)),
        albumentations.Normalize(mean=args.normalize_mean, std=args.normalize_std),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.OneOf([
    	# 	albumentations.HorizontalFlip(p=1),
    	# 	albumentations.Rotate([-10, 10], p=1),
        # ], p=0.5),
        albumentations.pytorch.transforms.ToTensorV2(),
        # albumentations.RandomCrop(224, 224),
        # albumentations.RamdomCrop, CenterCrop, RandomRotation
        # albumentations.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    ])


    now = (
        dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    )
    model_save_path = args.model_save
    dirname = os.path.join(model_save_path, f"model_{now}")
    ensure_dir(dirname)

    st_time = time.time()
    for i in range(fold_num):
        # Resnent 18 네트워크의 Tensor들을 GPU에 올릴지 Memory에 올릴지 결정함
        mnist_resnet = resnet_finetune(resnet18, 18).to(device)
        wandb.watch(mnist_resnet)
        # mnist_resnet = efficient_model('efficientnet-b0', 18).to(device)

        # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy
        loss_fn = FocalLoss()
        # weight 업데이트를 위한 optimizer를 Adam으로 사용함
        optimizer = torch.optim.Adam(mnist_resnet.parameters(), lr=args.learning_rate)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
        lrs = []

        train_dataset = Mask_Dataset(
            data_transform, f"train{i}", train_list[i], train_path, args.image_folder
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            # 배치마다 어떤 작업을 해주고 싶을 때, 이미지 크기가 서로 맞지 않는 경우 맞춰줄 때 사용
            collate_fn=collate_fn,
            # 마지막 남은 데이터가 배치 사이즈보다 작을 경우 무시
            #  num_workers=2
        )
        val_dataset = Mask_Dataset(data_transform, f"val{i}", val_list[i], train_path, args.image_folder)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            #  num_workers=2
        )

        dataloaders = {"train": train_loader, "test": val_loader}
        TRAIN_FLAG = 'train'
        TEST_FLAG = 'test'
        ### 학습 코드 시작
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        # flag = True
        # early_ind = 0
        pred_f1 = 0.0
        for epoch in range(args.epoch):
            # if not (flag):
            #     break
            for phase in [TRAIN_FLAG, TEST_FLAG]:
                n_iter = 0
                running_loss = 0.0
                running_acc = 0.0
                running_f1 = 0.0

                if phase == TRAIN_FLAG:
                    mnist_resnet.train()  # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
                elif phase == TEST_FLAG:
                    mnist_resnet.eval()  # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

                for ind, (images, labels) in enumerate(
                    tqdm.tqdm(dataloaders[phase], leave=False)
                ):
                    images = torch.stack(list(images), dim=0).to(device)
                    labels = torch.tensor(list(labels)).to(device)

                    optimizer.zero_grad()  # parameter gradient를 업데이트 전 초기화함

                    with torch.set_grad_enabled(
                        phase == TRAIN_FLAG
                    ):  # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                        logits = mnist_resnet(images)
                        _, preds = torch.max(
                            logits, 1
                        )  # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
                        loss = loss_fn(logits, labels)

                        if phase == TRAIN_FLAG:
                            loss.backward()  # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                            optimizer.step()  # 계산된 gradient를 가지고 모델 업데이트
                            lr_sched.step()
                            lrs.append(optimizer.param_groups[0]["lr"])

                    running_loss += batch_loss(loss, images)  # 한 Batch에서의 loss 값 저장
                    running_acc+= batch_acc(
                        preds, labels.data
                    )  # 한 Batch에서의 Accuracy 값 저장
                    running_f1 += batch_f1(
                        preds.cpu().numpy(), labels.cpu().numpy(), "macro"
                    )
                    n_iter += 1
                    if ind%100==0:
                        wandb.log({'loss':loss})
                        wandb.log({'lr':lrs[-1]})


                # 한 epoch이 모두 종료되었을 때,
                data_len = len(dataloaders[phase].dataset)
                epoch_loss = epoch_mean(running_loss, data_len)
                epoch_acc = epoch_mean(running_acc, data_len)
                epoch_f1 = epoch_mean(running_f1, n_iter)

                print(
                    f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, F1 Score : {epoch_f1:.3f}"
                )
                if (
                    phase == TEST_FLAG and best_test_accuracy < epoch_acc
                ):  # phase가 test일 때, best accuracy 계산
                    best_test_accuracy = epoch_acc
                if (
                    phase == TEST_FLAG and best_test_loss > epoch_loss
                ):  # phase가 test일 때, best loss 계산
                    best_test_loss = epoch_loss
                # Early Stopping Code
                # if phase == TEST_FLAG:
                #     if pred_f1 <= epoch_f1:
                #         pred_f1 = epoch_f1
                #         torch.save(mnist_resnet, os.path.join(dirname, f"model_mnist{i}.pickle"))
                #         print(f"{epoch}번째 모델 저장!")
                #         early_ind = 0
                #     else:
                #         print(f"{epoch}번째 모델 pass")
                # early_ind += 1
                # if early_ind == 2:
                #     flag = False
                #     break
        torch.save(mnist_resnet, os.path.join(dirname, f"model_mnist{i}.pickle"))
        print("학습 종료!")
        print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}")
    ed_time = time.time()
    total_minute = (round(ed_time - st_time, 2)) // 60
    print(f"총 학습 시간 : {total_minute}분 소요되었습니다.")
    notification(best_test_accuracy)
