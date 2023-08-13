import time
import timeit

import openai
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import ChatGLM

# default endpoint_url for a local deployed ChatGLM api server
endpoint_url = "http://region-9.seetacloud.com:41927"
# endpoint_url = "http://localhost:8000"


def chatglm_llm():
    """
    ChatGLM 大语言模型
    :return:
    """

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        # history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )
    return llm


def test_chatglm(question):
    template = """{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        # history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)


def chatglm_openai(question):
    openai.api_base = f"{endpoint_url}/v1"
    print(openai.api_base)
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
            model="chatglm2-6b",
            messages=[
                {"role": "user", "content": question}
            ],
            stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)


def retriever(query, llm):
    # 加载本地文件
    loader = TextLoader('../doc/state_of_the_union.txt', encoding='utf8')
    # embedding模型
    embedding = HuggingFaceEmbeddings()
    index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
    return index.query(question=query, llm=llm)


def retriever_openai(query):
    # 加载本地文件
    loader = TextLoader('../doc/state_of_the_union_cn.txt', encoding='utf8')
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index.query(question=query)


if __name__ == '__main__':
    start_time = time.time()
    q = "Python入门"
    # answer = retriever_openai(question)
    # answer = retriever(question, chatglm_llm())
    answer = test_chatglm(q)
    # chatglm_openai(q)
    print(f"answer: {answer}")
    print(f"函数 {test_chatglm.__name__} 的运行时间为: {time.time() - start_time}")
