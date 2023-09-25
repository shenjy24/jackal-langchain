import time
import jwt
import requests

from utils import generate_random_str

api_key = "db0fe343671af8239b93b5baf11ea729.yHLTj7SdtqaPAHGd"


def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def invoke():
    url = "https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_pro/invoke"
    data = {
        "prompt": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "我是人工智能助手"},
            {"role": "user", "content": "你叫什么名字"},
            {"role": "assistant", "content": "我叫chatGLM"},
            {"role": "user", "content": "你都可以做些什么事"},
        ],
        "temperature": 0.8,    # 采样温度，控制输出的随机性，取值范围是：(0.0,1.0]
        # "top_p": 0.7,        # 用温度取样的另一种方法，称为核取样. 取值范围是：(0.0, 1.0) 开区间，默认值为 0.7. 不要同时调整两个参数
        "request_id": generate_random_str(),
    }
    headers = {"Authorization": generate_token(api_key, 300)}
    response = requests.post(url, json=data, headers=headers)

    # 检查响应状态码
    if response.status_code == 200:
        print("POST请求成功:")
        print(response.json())  # 输出响应JSON数据
    else:
        print("POST请求失败. 状态码:", response.status_code)


if __name__ == '__main__':
    invoke()
