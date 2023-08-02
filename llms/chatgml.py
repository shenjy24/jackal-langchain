import time
from langchain.llms import ChatGLM
from langchain import PromptTemplate, LLMChain


def test_chatglm(question):
    # default endpoint_url for a local deployed ChatGLM api server
    endpoint_url = "http://127.0.0.1:8000"

    template = """{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)


if __name__ == '__main__':
    start_time = time.time()
    question = "北京和上海两座城市有什么不同？"
    answer = test_chatglm(question)
    print(f"answer: {answer}")
    print(f"函数 {test_chatglm.__name__} 的运行时间为: {time.time() - start_time} 秒")
