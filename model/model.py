import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True)

        self.linear_mask_1 = nn.Linear(1000, 256, bias=True)
        self.linear_mask_2 = nn.Linear(256, 3, bias=True)

        self.linear_gender_1 = nn.Linear(1000, 256, bias=True)
        self.linear_gender_2 = nn.Linear(256, 1, bias=True)

        self.linear_age_1 = nn.Linear(1000, 256, bias=True)
        self.linear_age_2 = nn.Linear(256, 3, bias=True)

    def forward(self, x):
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x_mask = self.linear_mask_1(x)
        x_mask = F.relu(x_mask)
        x_mask = F.dropout(x_mask, training=self.training)
        x_mask = self.linear_mask_2(x_mask)

        x_gender = self.linear_gender_1(x)
        x_gender = F.relu(x_gender)
        x_gender = F.dropout(x_gender, training=self.training)
        x_gender = self.linear_mask_2(x_gender)

        x_age = self.linear_age_1(x)
        x_age = F.relu(x_age)
        x_age = F.dropout(x_age, training=self.training)
        x_age = self.linear_age_2(x_age)

        return x_mask, x_gender, x_age


