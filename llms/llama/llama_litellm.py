from litellm import completion
import openai  # openai v1.0.0+


def basic_usage():
    response = completion(
        model="ollama/llama3",
        messages=[{"content": "你好，你是谁?", "role": "user"}],
        api_base="http://localhost:11434"
    )
    print(response)


def stream_usage():
    response = completion(
        model="ollama/llama3",
        messages=[{"content": "你好，你是谁?", "role": "user"}],
        api_base="http://localhost:11434",
        stream=True,
    )
    print(response)


def openai_usage():
    # 使用 litellm 的 openai 代理
    client = openai.OpenAI(api_key="anything", base_url="http://localhost:4000")  # set proxy to base_url
    # request sent to model set on litellm proxy, `litellm --model`
    response = client.chat.completions.create(model="ollama/llama3", messages=[
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ])
    print(response)


if __name__ == "__main__":
    openai_usage()
