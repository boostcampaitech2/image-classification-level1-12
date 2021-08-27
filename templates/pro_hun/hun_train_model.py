import datetime as dt
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import cv2
import albumentations
import albumentations.pytorch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet152, resnet50
from torchvision.transforms.transforms import RandomRotation
import tqdm
from PIL import Image
from pytz import timezone
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import resnet18
from torchvision.transforms import Normalize, Resize, ToTensor

import notification
from hun_kfold import Run_Split

random_seed = 12
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def resnet_finetune(model, classes):
    model = model(pretrained=True)
    # for params in model.parameters():
    #     params.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=classes, bias=True)

    print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
    print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc.weight.shape[0])

    torch.nn.init.xavier_uniform_(model.fc.weight)
    stdv = 1.0 / math.sqrt(model.fc.weight.size(1))
    model.fc.bias.data.uniform_(-stdv, stdv)

    # model.fc = nn.Linear(in_features=2048, out_features=512, bias=True)
    # model.bc = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model.relu = nn.ReLU(inplace=True)
    # model.dropout = nn.Dropout(p=0.2)
    # model.fc2= nn.Linear(in_features=512, out_features=classes, bias=True)

    # print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
    # print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc2.weight.shape[0])

    # torch.nn.init.xavier_uniform_(model.fc2.weight)
    # stdv = 1.0 / math.sqrt(model.fc2.weight.size(1))
    # model.fc2.bias.data.uniform_(-stdv, stdv)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


class Mask_Dataset(object):
    def __init__(self, transforms, name, df):
        self.transforms = transforms
        self.name = name
        self.imgs = list(
            sorted(
                os.listdir(
                    f"/opt/ml/image-classification-level1-12/templates/data/train/image_all/{self.name}_image"
                )
            )
        )
        self.df = df


    def __getitem__(self, idx):
        # img_path = Path(self.df["path"][idx])
        img_path = self.df["path"][idx]
        target = self.df["label"][idx]
        
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # if self.transforms is not None:
        #     augmented = self.transforms(image = img)
        #     img = augmented['image']

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
                
        return img, target
        

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    train_path = "/opt/ml/image-classification-level1-12/templates/data/train"
    train_label = pd.read_csv(os.path.join(train_path, "train_with_label.csv"))
    run_split = Run_Split(os.path.join(train_path, "image_all"))
    train_list, val_list = run_split.train_val_split(train_label)


    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음
    print(f"{device} is using!")


    # data_transform = albumentations.Compose([
    #     albumentations.Resize(512, 384, cv2.INTER_LINEAR),
    #     albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    #     albumentations.OneOf([
	# 		albumentations.HorizontalFlip(p=1),
	# 		albumentations.Rotate([-10, 10], p=1),
    #     ], p=0.5),
    #     albumentations.pytorch.transforms.ToTensorV2(),
    #     # albumentations.RandomCrop(224, 224),
    #     # albumentations.RamdomCrop, CenterCrop, RandomRotation
    #     # albumentations.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    # ])


    data_transform = transforms.Compose(
        [
            Resize((512, 384), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            # RandomRotation([-8, +8])
        ]
    )

    LEARNING_RATE = 0.0001  # 학습 때 사용하는 optimizer의 학습률 옵션 설정
    NUM_EPOCH = 10  # 학습 때 mnist train 데이터 셋을 얼마나 많이 학습할지 결정하는 옵션

    now = (
        dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    )
    dirname = Path(
        os.path.join(
            "/opt/ml/image-classification-level1-12/templates/pro_hun/output/model",
            f"model_{now}",
        )
    )
    dirname.mkdir(parents=True, exist_ok=False)
    st_time = time.time()
    for i in range(5):
        # Resnent 18 네트워크의 Tensor들을 GPU에 올릴지 Memory에 올릴지 결정함
        mnist_resnet = resnet_finetune(resnet18, 18).to(device)  

        loss_fn = (
            torch.nn.CrossEntropyLoss()
        )  # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy
        optimizer = torch.optim.Adam(
            mnist_resnet.parameters(), lr=LEARNING_RATE
        )  # weight 업데이트를 위한 optimizer를 Adam으로 사용함

        train_dataset = Mask_Dataset(data_transform, f"train{i}", train_list[i])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            # 배치마다 어떤 작업을 해주고 싶을 때, 이미지 크기가 서로 맞지 않는 경우 맞춰줄 때 사용
            collate_fn=collate_fn,
            # 마지막 남은 데이터가 배치 사이즈보다 작을 경우 무시
            # drop_last=True,
            #  num_workers=2
        )
        val_dataset = Mask_Dataset(data_transform, f"val{i}", val_list[i])
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=128,
            collate_fn=collate_fn,
            # drop_last=True,
            #  num_workers=2
        )

        dataloaders = {"train": train_loader, "test": val_loader}

        ### 학습 코드 시작
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        # flag = True
        # early_ind = 0
        pred_f1 = 0.0
        for epoch in range(NUM_EPOCH):
            epoch_f1 = 0
            n_iter = 0

            # if not (flag):
            #     break
            for phase in ["train", "test"]:
                running_loss = 0.0
                running_acc = 0.0
                if phase == "train":
                    mnist_resnet.train()  # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
                elif phase == "test":
                    mnist_resnet.eval()  # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

                for ind, (images, labels) in enumerate(
                    tqdm.tqdm(dataloaders[phase], leave=False)
                ):
                    images = torch.stack(list(images), dim=0).to(device)
                    labels = torch.tensor(list(labels)).to(device)

                    optimizer.zero_grad()  # parameter gradient를 업데이트 전 초기화함

                    with torch.set_grad_enabled(
                        phase == "train"
                    ):  # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                        logits = mnist_resnet(images)
                        _, preds = torch.max(
                            logits, 1
                        )  # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
                        loss = loss_fn(logits, labels)

                        if phase == "train":
                            loss.backward()  # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                            optimizer.step()  # 계산된 gradient를 가지고 모델 업데이트
                    running_loss += loss.item() * images.size(0)  # 한 Batch에서의 loss 값 저장
                    running_acc += torch.sum(
                        preds == labels.data
                    )  # 한 Batch에서의 Accuracy 값 저장
                    epoch_f1 += f1_score(
                        preds.cpu().numpy(), labels.cpu().numpy(), average="macro"
                    )
                    n_iter += 1

                # 한 epoch이 모두 종료되었을 때,
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_acc / len(dataloaders[phase].dataset)
                epoch_f1 = epoch_f1 / n_iter

                print(
                    f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, F1 Score : {epoch_f1:.3f}"
                )
                if (
                    phase == "test" and best_test_accuracy < epoch_acc
                ):  # phase가 test일 때, best accuracy 계산
                    best_test_accuracy = epoch_acc
                if (
                    phase == "test" and best_test_loss > epoch_loss
                ):  # phase가 test일 때, best loss 계산
                    best_test_loss = epoch_loss
                # if phase == "test":
                #     if pred_f1 <= epoch_f1:
                #         pred_f1 = epoch_f1
                        
                #         print(f"{epoch}번째 모델 저장!")
                        # early_ind = 0
                #     else:
                #         print(f"{epoch}번째 모델 pass")
                        # early_ind += 1
                        # if early_ind == 2:
                        #     flag = False
                        #     break
        torch.save(mnist_resnet, os.path.join(dirname, f"model_mnist{i}.pickle"))
        print("학습 종료!")
        print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}")
    # torch.save(mnist_resnet, os.path.join(dirname, f"model_mnist{i}.pickle"))
    ed_time = time.time()
    total_minute = (round(ed_time - st_time, 2)) // 60
    print(f"총 학습 시간 : {total_minute}분 소요되었습니다.")
    notification