from pathlib import Path
import torch
import json
import sys
import requests


def ensure_dir(dirname):
	# dirname을 path로 가지는 빈 directory 생성
	dirname = Path(dirname)
	if not dirname.is_dir():
		dirname.mkdir(parents=True, exist_ok=False)


def prepare_device():
	# 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return device


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
