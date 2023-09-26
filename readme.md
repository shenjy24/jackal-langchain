## 一. LangChain 入门
### 1. 入门文档
[Data connection](https://python.langchain.com/docs/modules/data_connection/)

### 2. 接入LLM
[LLMs](https://python.langchain.com/docs/integrations/llms/chatglm)

### 3. 环境变量
`SERPAPI_API_KEY` 和 `OPENAI_API_KEY` 等环境变量都配置在 PyCharm 中的 Environment Variables 中。

## 二. 构建个人知识库
### 1. 接入智谱API
[智谱API官方文档](https://maas.aminer.cn/dev/api#chatglm_pro)

#### (1) 智谱SSE接口调用
```python
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
```

#### (2) 知识库流程
```python
def get_zhipu_answer(question):
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma(persist_directory="chroma", embedding_function=embeddings)
    docs = docsearch.similarity_search(question, include_metadata=True)
    print(f"docs={docs}")
    context = get_context(docs)
    prompts = get_prompts(context, question)
    print(f"prompts={prompts}")
    response = sdk_sse_invoke(prompts=prompts)  # Invoke SSE and get the response object
    for event in response.events():
        yield event.data

        
def get_context(docs: List[Document]):
    """
    从文档列表中组合成上下文
    """
    if not docs:
        return ""
    document_separator: str = "\n\n"
    doc_strings = [doc.page_content for doc in docs]
    return document_separator.join(doc_strings)


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
```

#### (3) FastAPI流式输出
```python
@app.get("/zhipu/{prompt}")
async def events(prompt: str):
    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    return StreamingResponse(get_zhipu_answer(prompt), headers=headers)
```