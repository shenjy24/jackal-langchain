import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

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
