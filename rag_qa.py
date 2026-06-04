from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA


def main():
    print("Loading vector store...")
    # 加载之前构建的向量数据库
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print("Initializing LLM...")
    # 修改这里：使用 OllamaLLM 替代 Ollama
    llm = OllamaLLM(model="qwen2:7b", temperature=0.7)

    print("Setting up RAG chain...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\nRAG system ready! Ask your questions. (Type 'exit' to quit)\n")
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        result = qa_chain.invoke({"query": query})
        # result = qa_chain({"query": query})
        print(f"Answer: {result['result']}")
        print("\n--- Sources ---")
        for i, doc in enumerate(result['source_documents']):
            print(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}")
        print("-" * 30 + "\n")


if __name__ == "__main__":
    main()