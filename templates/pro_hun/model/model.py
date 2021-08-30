import math

import torch.nn as nn


def resnet_finetune(model, classes):
    """
    resnet model명과 output class 개수를 입력해주면
    그것에 맞는 모델을 반환, pretrained, bias initialize
    """

    model = model(pretrained=True)
    # for params in model.parameters():
    #     params.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=classes, bias=True)

    print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
    print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc.weight.shape[0])

    nn.init.xavier_uniform_(model.fc.weight)
    stdv = 1.0 / math.sqrt(model.fc.weight.size(1))
    model.fc.bias.data.uniform_(-stdv, stdv)

    # model.fc = nn.Linear(in_features=512, out_features=128, bias=True)
    # model.bc = nn.BatchNorm2d(
    #     128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    # )
    # model.relu = nn.ReLU(inplace=True)
    # model.dropout = nn.Dropout(p=0.2)
    # model.fc2 = nn.Linear(in_features=128, out_features=classes, bias=True)

    # print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
    # print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc2.weight.shape[0])

    # nn.init.xavier_uniform_(model.fc2.weight)
    # stdv = 1.0 / math.sqrt(model.fc2.weight.size(1))
    # model.fc2.bias.data.uniform_(-stdv, stdv)

    return model
