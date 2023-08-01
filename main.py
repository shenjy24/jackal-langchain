import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

openai_api_key = "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"
llm = OpenAI(openai_api_key=openai_api_key)
chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
os.environ['OPENAI_API_KEY'] = openai_api_key


def retriever():
    # 加载本地文件
    loader = TextLoader('./doc/state_of_the_union.txt', encoding='utf8')
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "What did the president say about Ketanji Brown Jackson"
    return index.query(query)


def test_llm(prompt):
    print(llm.predict(prompt))
    print(chat_model.predict(prompt))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_llm('hi')
    print(retriever())
