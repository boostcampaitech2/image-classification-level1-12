import argparse
import datetime as dt
import os
import random
import time

import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pytz import timezone
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Normalize, Resize, ToTensor
from tqdm.notebook import tqdm

from utils.util import prepare_device

random_seed = 12
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):

        # img = Image.open(img_path).convert("RGB")
        # image = cv2.imread(self.img_paths[index])
        # if self.transform is not None:
        #     augmented = self.transform(image = image)
        #     img = augmented['image']
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "--test_path",
        default="/opt/ml/image-classification-level1-12/templates/data/eval",
        type=str,
        help="test_path",
    )
    args.add_argument(
        "--result_save",
        default="/opt/ml/image-classification-level1-12/templates/pro_hun/output/sub",
        type=str,
        help="result_save_path",
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

    args = args.parse_args()

    test_dir = args.test_path
    submission = pd.read_csv(os.path.join(test_dir, "info.csv"))
    image_dir = os.path.join(test_dir, "images")

    device = prepare_device()

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    transform = transforms.Compose(
        [
            Resize((512, 384), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=args.normalize_mean, std=args.normalize_std),
        ]
    )

    # transform = albumentations.Compose([
    # 	albumentations.Resize(512, 384, cv2.INTER_LINEAR),
    # 	albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    # 	albumentations.pytorch.transforms.ToTensorV2(),
    # 	# albumentations.RandomCrop(224, 224),
    # 	# albumentations.RamdomCrop, CenterCrop, RandomRotation
    # 	# albumentations.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    # ])

    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset, shuffle=False, num_workers=2)

    # model_path = '/opt/ml/image-classification-level1-12/templates/pro_hun/output/model/model_2021-08-25_004053'
    model_path = input("학습한 모델의 경로를 입력해주세요 : ")

    model_listdir = os.listdir(model_path)
    all_predictions = [[] for _ in range(len(loader))]
    for moli in model_listdir:
        model = torch.load(os.path.join(model_path, moli))
        model.eval()

        idx = 0
        st_time = time.time()
        for images in loader:
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = F.softmax(pred, dim=1).cpu().numpy()
                if len(all_predictions[idx]) == 0:
                    all_predictions[idx] = pred / len(model_listdir)
                else:
                    all_predictions[idx] += pred / len(model_listdir)
            idx += 1
            if idx % 500 == 0:
                ed_time = time.time()
                use_time = round(ed_time - st_time, 2)
                remian_time = round((use_time / idx) * (len(loader) - idx), 2)
                print(f"\r{moli} 걸린 시간 : {use_time:10}, \t남은 시간: {remian_time}", end="")

    all_predictions = [all_pre.argmax() for all_pre in all_predictions]
    submission["ans"] = all_predictions

    # 제출할 파일을 저장합니다.
    now = (
        dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    )
    submission.to_csv(os.path.join(args.result_save, f"sub_{now}.csv"))
    print("test inference is done!")
