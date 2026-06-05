from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

ES_URL = "http://localhost:9200"
INDEX_NAME = "rag_hybrid_kb"
MODEL_NAME = "qwen2:7b"

print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name=INDEX_NAME,
    embedding=embeddings,
    es_user="elastic",
    es_password="<你的密码>",  # 请替换为从 start.sh 输出的密码
)

# 关键点：配置混合检索策略
# strategy=... 是启用混合检索的核心，它会同时执行文本搜索和向量搜索
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3},
    # 通过传入 strategy 参数，可以开启混合检索。你也可以创建自定义的混合策略对象
)

print("Initializing LLM...")
llm = OllamaLLM(model=MODEL_NAME, temperature=0.7)

print("Setting up RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\nRAG system with Elasticsearch hybrid search is ready!")
while True:
    query = input("Your question: ")
    if query.lower() == "exit":
        break
    if not query.strip():
        continue

    result = qa_chain.invoke({"query": query})
    print(f"Answer: {result['result']}")
    print("\n--- Sources ---")
    for i, doc in enumerate(result['source_documents']):
        print(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 30 + "\n")