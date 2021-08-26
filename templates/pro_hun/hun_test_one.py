import datetime as dt
import os
import random
import time

import cv2
import albumentations
import albumentations.pytorch

import numpy as np
import pandas as pd
import torch
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
		image = cv2.imread(self.img_paths[index])
		
		if self.transform is not None:
			augmented = self.transform(image = image)
			img = augmented['image']

		# img = Image.open(img_path).convert("RGB")

		# if self.transforms is not None:
		#     img = self.transforms(img)
				
		return img
     #    image = Image.open(self.img_paths[index])

     #    if self.transform:
     #        image = self.transform(image)
     #    return image

	def __len__(self):
		return len(self.img_paths)


if __name__ == "__main__":
	model = torch.load(
		"/opt/ml/image-classification-level1-12/templates/pro_hun/output/model/model_2021-08-26_223157/model_mnist0.pickle"
	)
	model.eval()

	test_dir = "/opt/ml/image-classification-level1-12/templates/data/eval"
	submission = pd.read_csv(os.path.join(test_dir, "info.csv"))
	image_dir = os.path.join(test_dir, "images")

	# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
	image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

#     transform = transforms.Compose(
#         [
#             Resize((512, 384), Image.BILINEAR),
#             ToTensor(),
#             Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
#         ]
	#     )
	transform = albumentations.Compose([
		albumentations.Resize(512, 384, cv2.INTER_LINEAR),
		albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
		
		albumentations.pytorch.transforms.ToTensorV2(),
		# albumentations.RandomCrop(224, 224),
		# albumentations.RamdomCrop, CenterCrop, RandomRotation
		# albumentations.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
	])

	dataset = TestDataset(image_paths, transform)

	loader = DataLoader(dataset, shuffle=False, num_workers=8)

	device = torch.device(
		"cuda:0" if torch.cuda.is_available() else "cpu"
	)  # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

	all_predictions = []
	time_ck = 0
	st_time = time.time()
	for images in loader:
		time_ck += 1
		with torch.no_grad():
			images = images.to(device)
			pred = model(images)
			pred = pred.argmax(dim=-1)
			all_predictions.extend(pred.cpu().numpy())
		if time_ck % 1000 == 0:
			ed_time = time.time()
			use_time = ed_time - st_time
			remian_time = round((use_time / time_ck) * (len(loader) - time_ck), 2)
			print(f"\r걸린 시간 : {use_time}, \t남은 시간: {remian_time}", end="")
	submission["ans"] = all_predictions

	# 제출할 파일을 저장합니다.
	now = (
		dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
	)
	submission.to_csv(
		f"/opt/ml/image-classification-level1-12/templates/pro_hun/output/sub/sub_{now}.csv", index=False
	)
	print("test inference is done!")