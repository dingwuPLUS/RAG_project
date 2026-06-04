from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


def main():
    print("Loading vector store...")
    # 加载之前构建的向量数据库，必须使用相同的embedding模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print("Initializing LLM...")
    # 初始化Ollama，连接到之前下载的本地模型
    llm = Ollama(model="qwen2:7b", temperature=0.7)  # 这里模型名称需要和之前 ollama run 的模型一致

    print("Setting up RAG chain...")
    # 创建检索器：从向量库中检索最相关的Top-3个文档片段作为上下文
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 创建问答链：stuff 模式意味着将所有检索到的片段一次性放入prompt中
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # 返回来源，便于追溯
    )

    print("\nRAG system ready! Ask your questions. (Type 'exit' to quit)\n")
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        # 调用RAG链生成答案
        result = qa_chain({"query": query})
        print(f"Answer: {result['result']}")
        print("\n--- Sources ---")
        # 打印答案的来源，增加可解释性
        for i, doc in enumerate(result['source_documents']):
            print(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')} (Page: {doc.metadata.get('page', 'N/A')})")
        print("-" * 30 + "\n")


if __name__ == "__main__":
    main()