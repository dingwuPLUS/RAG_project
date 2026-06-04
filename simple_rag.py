from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json


def main():
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print("\nRAG system ready! Ask your questions. (Type 'exit' to quit)\n")

    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        # 检索相关文档
        docs = vectordb.similarity_search(query, k=3)

        # 构建 prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下参考信息回答用户的问题。如果参考信息中没有相关内容，请说"根据已有信息无法回答"。

参考信息：
{context}

用户问题：{query}

回答："""

        # 调用 Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2:7b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=60
        )

        if response.status_code == 200:
            answer = response.json()["response"]
            print(f"Answer: {answer}")
        else:
            print(f"Error: {response.status_code}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()