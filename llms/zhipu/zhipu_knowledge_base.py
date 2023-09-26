from typing import List

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from llms.zhipu.zhipu_sdk import sdk_sse_invoke


def get_zhipu_answer(question):
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma(persist_directory="chroma", embedding_function=embeddings)
    docs = docsearch.similarity_search(question, include_metadata=True)
    context = get_context(docs)
    prompts = get_prompts(context, question)
    response = sdk_sse_invoke(prompts=prompts)  # Invoke SSE and get the response object
    for event in response.events():
        yield event.data


def get_prompts(context, question):
    prompt_template = PromptTemplate.from_template(
        '''
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer:
        '''
    )
    prompt = prompt_template.format(context=context, question=question)
    return [{"role": "user", "content": f"{prompt}"}]


def get_context(docs: List[Document]):
    """
    从文档列表中组合成上下文
    """
    if not docs:
        return ""
    document_separator: str = "\n\n"
    doc_strings = [doc.page_content for doc in docs]
    return document_separator.join(doc_strings)
