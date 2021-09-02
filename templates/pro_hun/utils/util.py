import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix


def ensure_dir(dirname):
    # dirname을 path로 가지는 빈 directory 생성
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def prepare_device():
    # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def fix_randomseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def notification(best_acc):
    url = "https://hooks.slack.com/services/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"  # 웹후크 URL 입력
    message = "학습이 완료되었습니다!!! {best_acc}"  # 메세지 입력
    title = f"New Incoming Message :zap:"  # 타이틀 입력
    slack_data = {
        "username": "NotificationBot",  # 보내는 사람 이름
        "icon_emoji": ":satellite:",
        # "channel" : "#somerandomcahnnel",
        "attachments": [
            {
                "color": "#9733EE",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ],
            }
        ],
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {"Content-Type": "application/json", "Content-Length": byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


def draw_confusion_matrix(self, target, pred):
    cm = confusion_matrix(target, pred)
    df = pd.DataFrame(
        cm / np.sum(cm, axis=1)[:, None], index=list(range(18)), columns=list(range(18))
    )
    df = df.fillna(0)  # NaN 값을 0으로 변경

    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle("Confusion Matrix")
    sn.heatmap(df, annot=True, cmap=sn.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig("confusion_matrix.png")
