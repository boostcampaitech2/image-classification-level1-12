import argparse
import datetime as dt
import os
import time

import albumentations
import albumentations.pytorch
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from data_preprocessing.data_split import Run_Split
from model.loss import batch_loss, FocalLoss
from model.metric import batch_acc, batch_f1, epoch_mean
from model.model import resnet_finetune
from pytz import timezone
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from utils.util import ensure_dir, fix_randomseed, notification, prepare_device

import wandb


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
            augmented = self.transforms(image=img)
            img = augmented["image"]

        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(project="Mask_classification", entity="herjh0405")
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
        default="/opt/ml/image-classification-level1-12/data/train",
        type=str,
        help="train_directory_path",
    )
    args.add_argument(
        "--model_save",
        default="/opt/ml/image-classification-level1-12/output/model_save",
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
        default="train_with_all.csv",
        type=str,
        help="CSV according to image type(Original, Crop, All)",
    )
    # Original:image_all, Crop:image_crop_all
    args.add_argument(
        "--image_dir",
        default="ori_crop_split",
        type=str,
        help="Directory according to image type",
    )

    args = args.parse_args()
    config = wandb.config
    config.learning_rate = args.learning_rate
    fix_randomseed(12)

    train_path = args.train_path
    train_label = pd.read_csv(os.path.join(train_path, args.image_data))
    run_split = Run_Split(os.path.join(train_path, args.image_dir))
    fold_num = args.fold_size
    train_list, val_list = run_split.train_val_split(train_label, fold_num)

    # GPU??? ?????????????????? ????????????, ????????? CPU ??????
    device = prepare_device()
    print(f"{device} is using!")

    data_transform = albumentations.Compose(
        [
            albumentations.Resize(512, 384, cv2.INTER_LINEAR),
            albumentations.GaussianBlur(3, sigma_limit=(0.1, 2)),
            albumentations.Normalize(mean=args.normalize_mean, std=args.normalize_std),
            albumentations.HorizontalFlip(
                p=0.5
            ),  # Same with transforms.RandomHorizontalFlip()
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )

    now = (
        dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    )
    model_save_path = args.model_save
    dirname = os.path.join(model_save_path, f"model_{now}")
    ensure_dir(dirname)

    st_time = time.time()
    for i in range(fold_num):
        # Resnent 18 ??????????????? Tensor?????? GPU??? ????????? Memory??? ????????? ?????????
        mnist_resnet = resnet_finetune(resnet18, 18).to(device)
        wandb.watch(mnist_resnet)

        # ?????? ?????? ??? ?????? ???????????? Cross entropy loss??? objective function?????? ?????? - https://en.wikipedia.org/wiki/Cross_entropy
        loss_fn = FocalLoss()
        # weight ??????????????? ?????? optimizer??? Adam?????? ?????????
        optimizer = torch.optim.Adam(mnist_resnet.parameters(), lr=args.learning_rate)

        train_dataset = Mask_Dataset(
            data_transform, f"train{i}", train_list[i], train_path, args.image_dir
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            # ???????????? ?????? ????????? ????????? ?????? ???, ????????? ????????? ?????? ?????? ?????? ?????? ????????? ??? ??????
            collate_fn=collate_fn,
            # ????????? ?????? ???????????? ?????? ??????????????? ?????? ?????? ??????
        )
        val_dataset = Mask_Dataset(
            data_transform, f"val{i}", val_list[i], train_path, args.image_dir
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )

        dataloaders = {"train": train_loader, "test": val_loader}
        TRAIN_FLAG = "train"
        TEST_FLAG = "test"
        ### ?????? ?????? ??????
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        for epoch in range(args.epoch):
            for phase in [TRAIN_FLAG, TEST_FLAG]:
                n_iter = 0
                running_loss = 0.0
                running_acc = 0.0
                running_f1 = 0.0

                if phase == TRAIN_FLAG:
                    mnist_resnet.train()  # ???????????? ????????? train ????????? ?????? gradient??? ????????????, ?????? sub module (?????? ?????????, ???????????? ???)??? train mode??? ????????? ??? ????????? ???
                elif phase == TEST_FLAG:
                    mnist_resnet.eval()  # ???????????? ????????? eval ?????? ?????? ?????? sub module?????? eval mode??? ????????? ??? ?????? ???

                for ind, (images, labels) in enumerate(
                    tqdm.tqdm(dataloaders[phase], leave=False)
                ):
                    images = torch.stack(list(images), dim=0).to(device)
                    labels = torch.tensor(list(labels)).to(device)

                    optimizer.zero_grad()  # parameter gradient??? ???????????? ??? ????????????

                    with torch.set_grad_enabled(
                        phase == TRAIN_FLAG
                    ):  # train ????????? ????????? gradient??? ????????????, ?????? ?????? gradient??? ???????????? ?????? ????????? ?????????
                        logits = mnist_resnet(images)
                        _, preds = torch.max(
                            logits, 1
                        )  # ???????????? linear ????????? ????????? ?????? ??? ([0.9,1.2, 3.2,0.1,-0.1,...])??? ?????? output index??? ?????? ?????? ?????????([2])??? ?????????
                        loss = loss_fn(logits, labels)

                        if phase == TRAIN_FLAG:
                            loss.backward()  # ????????? ?????? ?????? ?????? ?????? CrossEntropy ????????? ?????? gradient ??????
                            optimizer.step()  # ????????? gradient??? ????????? ?????? ????????????

                    running_loss += batch_loss(loss, images)  # ??? Batch????????? loss ??? ??????
                    running_acc += batch_acc(
                        preds, labels.data
                    )  # ??? Batch????????? Accuracy ??? ??????
                    running_f1 += batch_f1(
                        preds.cpu().numpy(), labels.cpu().numpy(), "macro"
                    )
                    n_iter += 1
                    if ind % 100 == 0:
                        wandb.log({"loss": loss})
                        wandb.log({"lr": args.learning_rate})

                # ??? epoch??? ?????? ??????????????? ???,
                data_len = len(dataloaders[phase].dataset)
                epoch_loss = epoch_mean(running_loss, data_len)
                epoch_acc = epoch_mean(running_acc, data_len)
                epoch_f1 = epoch_mean(running_f1, n_iter)

                print(
                    f"?????? epoch-{epoch}??? {phase}-????????? ????????? ?????? Loss : {epoch_loss:.3f}, ?????? Accuracy : {epoch_acc:.3f}, F1 Score : {epoch_f1:.3f}"
                )
                if (
                    phase == TEST_FLAG and best_test_accuracy < epoch_acc
                ):  # phase??? test??? ???, best accuracy ??????
                    best_test_accuracy = epoch_acc
                if (
                    phase == TEST_FLAG and best_test_loss > epoch_loss
                ):  # phase??? test??? ???, best loss ??????
                    best_test_loss = epoch_loss

        torch.save(mnist_resnet, os.path.join(dirname, f"model_mnist{i}.pickle"))
        print("?????? ??????!")
        print(f"?????? accuracy : {best_test_accuracy}, ?????? ?????? loss : {best_test_loss}")
    ed_time = time.time()
    total_minute = (round(ed_time - st_time, 2)) // 60
    print(f"??? ?????? ?????? : {total_minute}??? ?????????????????????.")
    notification(best_test_accuracy)
