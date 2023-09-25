import time

import openai
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import ChatGLM

from llms.zhipu.zhipu import ZhiPu
from utils import generate_random_str


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


def zhipu_llm(question):
    template = """{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = ZhiPu(top_p=0.9, request_id=generate_random_str())

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)


def zhipu_openai(question):
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
    loader = TextLoader('../../doc/state_of_the_union.txt', encoding='utf8')
    # embedding模型
    embedding = HuggingFaceEmbeddings()
    index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
    return index.query(question=query, llm=llm)


def retriever_openai(query):
    # 加载本地文件
    loader = TextLoader('../../doc/state_of_the_union_cn.txt', encoding='utf8')
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index.query(question=query)


if __name__ == '__main__':
    start_time = time.time()
    q = "Python入门"
    # answer = retriever_openai(question)
    # answer = retriever(question, chatglm_llm())
    answer = zhipu_llm(q)
    # chatglm_openai(q)
    print(f"answer: {answer}")
    print(f"函数 {zhipu_llm.__name__} 的运行时间为: {time.time() - start_time}")
