from models.mobilenet import MobileNet
from models.densenet import DenseNet
from models.resnet import ResNet
from data_generation.data_sets import TestDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    submission = pd.read_csv("./data/eval/info.csv")
    transform_base = transforms.Compose([
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.5601, 0.5241, 0.5014], [0.2331, 0.2430, 0.2456])]) 

    
    path = "./data/eval/images"
    testset = TestDataset(submission, path, transform_base)
    testloader = DataLoader(testset, batch_size = 128, num_workers = 4)
    
    model = MobileNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load('./results/models/mobile_centercrop280/20_0.0069_0.9990.pt'))

    lst = []

    for idx, x in enumerate(testloader):
        print(idx)
        x = x.to(DEVICE)
        y = model(x)
        y = torch.argmax(y, dim = 1)
        for i in y.cpu().numpy():
            lst.append(i)
    
    submission['ans'] = lst
    submission.to_csv("./results/submissions/submission12_mobile_centercrop280_epoch_20.csv", index = False)
            