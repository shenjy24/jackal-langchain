import os
import time

from langchain import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import ChatGLM
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma


def build_knowledge_base():
    loaders = [
        TextLoader('../doc/state_of_the_union_cn.txt', encoding='utf8')
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # document split
    chunk_size = 26
    chunk_overlap = 4  # chunk 重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    # vector store
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="chroma")

    # similarity search
    question = "俄罗斯总统做了什么"
    found_docs = vectordb.similarity_search(question, k=3)
    print("found docs size: " + str(len(found_docs)))
    print(found_docs)
    vectordb.persist()


def get_answer(question):
    embeddings = get_embeddings()
    docsearch = Chroma(persist_directory="chroma", embedding_function=embeddings)
    docs = docsearch.similarity_search(question, include_metadata=True)
    print(docs)
    # llm = get_llm()
    llm = OpenAI(openai_api_key=os.environ.get("openai_api_key"), callbacks=[StreamingStdOutCallbackHandler()], streaming=True)
    chain = load_qa_chain(llm)
    return chain.run(input_documents=docs, question=question)


def load_document(directory):
    """
    加载目录下的文件，存入向量数据库
    """
    # 加载目录下所有文件
    loader = DirectoryLoader(directory)
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    print(documents)
    # 切割加载的 document
    split_docs = get_split_docs(documents)
    # 初始化 embeddings 对象
    embeddings = get_embeddings()
    # 持久化数据
    docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma")
    docsearch.persist()


def load_single_document(doc):
    """
    加载目录下的文件，存入向量数据库
    """
    # 加载目录下所有文件
    loader = TextLoader(doc, encoding="utf-8")
    # 将数据转成 document 对象，每个文件会作为一个 document
    document = loader.load()
    print(document)
    # 切割加载的 document
    split_docs = get_split_docs(document)
    # 初始化 embeddings 对象
    embeddings = get_embeddings()
    # 持久化数据
    docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma")
    docsearch.persist()


def get_split_docs(documents):
    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的 document
    return text_splitter.split_documents(documents)


def get_embeddings():
    """
    获取 Embeddings 模型
    """
    return HuggingFaceEmbeddings()


def get_llm():
    # default endpoint_url for a local deployed ChatGLM api server
    # endpoint_url = "http://localhost:8000/"
    endpoint_url = "http://region-9.seetacloud.com:34302/"

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        # history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )
    return llm


if __name__ == "__main__":
    # load_single_document("../doc/state_of_the_union.txt")
    start_time = time.time()
    print(f"开始: {start_time}")
    answer = get_answer("对文档进行总结")
    # print(f"答案: {answer}")
    print(f"函数 {get_answer.__name__} 的运行时间为: {time.time() - start_time} 秒")
