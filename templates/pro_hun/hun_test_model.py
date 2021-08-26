import datetime as dt
import os
import random
import time

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
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    test_dir = "/opt/ml/image-classification-level1-12/templates/data/eval"
    submission = pd.read_csv(os.path.join(test_dir, "info.csv"))
    image_dir = os.path.join(test_dir, "images")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose(
        [
            Resize((512, 384), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ]
    )
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
                    all_predictions[idx] = pred / 5
                else:
                    all_predictions[idx] += pred / 5
            idx += 1
            if idx % 500 == 0:
                ed_time = time.time()
                use_time = round(ed_time - st_time, 2)
                remian_time = round((use_time / idx) * (len(loader) - idx), 2)
                print(
                    f"\r{moli} 걸린 시간 : {use_time:10}, \t남은 시간: {remian_time}", end=""
                )

    all_predictions = [all_pre.argmax() for all_pre in all_predictions]
    submission["ans"] = all_predictions

    import datetime as dt

    from pytz import timezone

    # 제출할 파일을 저장합니다.
    now = (
        dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    )
    submission.to_csv(
        f"/opt/ml/image-classification-level1-12/templates/pro_hun/output/sub/sub_{now}.csv",
        index=False,
    )
    print("test inference is done!")
