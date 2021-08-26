import json
import random
import sys

import requests

if __name__ == "__main__":
    url = "https://hooks.slack.com/services/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"  # 웹후크 URL 입력
    message = "학습이 완료되었습니다!!!"  # 메세지 입력
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
