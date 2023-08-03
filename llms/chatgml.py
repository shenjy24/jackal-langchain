import os
import time

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import ChatGLM
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

# openai_api_key = "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"
# os.environ['OPENAI_API_KEY'] = openai_api_key

def chatglm_llm():
    """
    ChatGLM 大语言模型
    :return:
    """
    # default endpoint_url for a local deployed ChatGLM api server
    endpoint_url = "http://127.0.0.1:8000"

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        # history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )
    return llm


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


def retriever(query, llm):
    # 加载本地文件
    loader = TextLoader('../doc/state_of_the_union.txt', encoding='utf8')
    # embedding模型
    embedding = HuggingFaceEmbeddings()
    index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
    # index = VectorstoreIndexCreator().from_loaders([loader])

    return index.query(question=query, llm=llm)
    # return index.query(question=query)


if __name__ == '__main__':
    start_time = time.time()
    question = "总结下核心内容，用中文回答"
    answer = retriever(question, chatglm_llm())
    print(f"answer: {answer}")
    print(f"函数 {test_chatglm.__name__} 的运行时间为: {time.time() - start_time} 秒")
