from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_model(model_name):
    if model_name == "baichuan":
        return get_baichuan_model()
    elif model_name == "ollama":
        return get_ollama_model()

    return get_ollama_model()


# 百川模型
def get_baichuan_model():
    return BaichuanTextEmbeddings(baichuan_api_key="sk-*")


# Ollama模型
def get_ollama_model():
    return OllamaEmbeddings(base_url="http://localhost:11434", model="llama3")


if __name__ == "__main__":
    embeddings = get_embedding_model("ollama")
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    print(query_result)
    doc_result = embeddings.embed_documents([text])
    print(doc_result[0][:5])
