import torch
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
        org_vgg19 = models.vgg19_bn(pretrained=True)

        self.vgg_features = org_vgg19.features
        self.vgg_avgpool = org_vgg19.avgpool

        self.classifier_common_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
        )

        self.classifier_mask_fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 3),
        )

        self.classifier_gender_fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 2)
        )

        self.regression_male_age_fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

        self.regression_female_age_fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # batch_size, 3, image_rows, image_columns
        x = self.vgg_features(x)
        # batch_size, 512, 12(depending on image_rows(384)), 9(depending on image_columns(288))
        x = self.vgg_avgpool(x)
        # batch_size, 512, 7, 7
        x = torch.flatten(x, 1)
        # batch_size, 25088
        x = self.classifier_common_fc(x)
        # batch_size, 4096

        # batch_size, 4096
        x_mask = self.classifier_mask_fc(x)
        # batch_size, 3

        # batch_size, 4096
        x_gender = self.classifier_gender_fc(x)
        # batch_size, 2
        x_gender_prob = F.softmax(x_gender, dim=1)
        # batch_size, 2

        # (batch_size, 4096) 2개
        x_male_female_age = torch.cat([self.regression_male_age_fc(x), self.regression_female_age_fc(x)], dim=1)
        # (batch_size, 2) X. (batch_size, 2)
        x_weighted_male_female_age = x_male_female_age * x_gender_prob
        # batch_size, 2
        x_age = torch.sum(x_weighted_male_female_age, dim=1)
        # batch_size,

        return x_mask, x_gender, x_age


