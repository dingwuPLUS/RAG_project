from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 加载已存在的向量库
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 1. 查看集合信息
print("=" * 50)
print("📊 向量库统计信息")
print("=" * 50)
print(f"总文档块数: {vectordb._collection.count()}")
print(f"集合名称: {vectordb._collection.name}")
print(f"集合 UUID: {vectordb._collection.id}")
print()

# 2. 查看所有文档块内容
print("=" * 50)
print("📄 所有文档块内容")
print("=" * 50)

# 获取所有数据（最多显示前10条）
all_data = vectordb.get(limit=10)

for i, (content, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
    print(f"\n--- 文档块 {i + 1} ---")
    print(f"来源: {metadata.get('source', 'Unknown')}")
    print(f"内容预览: {content[:200]}..." if len(content) > 200 else f"内容: {content}")
    print(f"元数据: {metadata}")

# 3. 测试检索功能
print("\n" + "=" * 50)
print("🔍 测试检索功能")
print("=" * 50)

test_query = "什么是奥特曼？"
results = vectordb.similarity_search_with_score(test_query, k=3)

print(f"查询: {test_query}")
for i, (doc, score) in enumerate(results):
    print(f"\n结果 {i + 1} (相似度分数: {score:.4f})")
    print(f"来源: {doc.metadata.get('source', 'Unknown')}")
    print(f"内容: {doc.page_content[:150]}...")

# 4. 导出到文本文件（便于查看）
print("\n" + "=" * 50)
print("💾 导出所有数据到文件")
print("=" * 50)

with open("vectorstore_export.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("向量数据库导出文件\n")
    f.write(f"总文档块数: {vectordb._collection.count()}\n")
    f.write("=" * 60 + "\n\n")

    for i, (content, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
        f.write(f"\n{'=' * 60}\n")
        f.write(f"文档块 {i + 1}\n")
        f.write(f"来源: {metadata.get('source', 'Unknown')}\n")
        f.write(f"元数据: {metadata}\n")
        f.write("-" * 60 + "\n")
        f.write(content + "\n")

print("✅ 数据已导出到 vectorstore_export.txt")