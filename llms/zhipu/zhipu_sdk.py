import os

import zhipuai


def sdk_sse_invoke(prompts: [], temperature: float = 0.95, request_id: str = None):
    """
    使用智谱SDK的SSE调用接口，返回流式数据

    Args
        prompts: 提示词数组，格式如下：
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "我是人工智能助手"},
                {"role": "user", "content": "你叫什么名字"},
                {"role": "assistant", "content": "我叫chatGLM"},
                {"role": "user", "content": "你都可以做些什么事"},
            ]
        temperature: 采样温度，控制输出的随机性，必须为正数，取值范围是：(0.0,1.0]，不能等于 0,默认值为 0.95
                     值越大，会使输出更随机，更具创造性；值越小，输出会更加稳定或确定
        request_id: 由用户端传参，需保证唯一性；用于区分每次请求的唯一标识，用户端不传时平台会默认生成。
    """
    zhipuai.api_key = os.environ.get("ZHIPU_API_KEY")
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_pro",
        prompt=prompts,
        temperature=temperature,
        request_id=request_id,
        incremental=True
    )
    return response
