import requests

from llms.zhipu import zhipu_api_key
from llms.zhipu.zhipu_util import generate_token
from utils import generate_random_str


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
    headers = {"Authorization": generate_token(zhipu_api_key)}
    response = requests.post(url, json=data, headers=headers)

    # 检查响应状态码
    if response.status_code == 200:
        print("POST请求成功:")
        print(response.json())  # 输出响应JSON数据
    else:
        print("POST请求失败. 状态码:", response.status_code)


if __name__ == '__main__':
    invoke()
