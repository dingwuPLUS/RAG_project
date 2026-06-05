import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
import glob

# 1. 配置 Elasticsearch 连接
ES_URL = "http://localhost:9200"
INDEX_NAME = "rag_hybrid_kb"

# 2. 初始化 Elasticsearch 客户端（用于管理索引）
es_client = Elasticsearch(ES_URL)

# 3. 如果索引已存在，先删除以便重建
if es_client.indices.exists(index=INDEX_NAME):
    es_client.indices.delete(index=INDEX_NAME)
    print(f"已删除旧索引: {INDEX_NAME}")

# 4. 创建索引映射 (Mappings)，同时定义 text 和 dense_vector 字段
# 这步是混合索引的关键，它告诉 Elasticsearch 如何存储和处理文本与向量
index_mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},                     # 用于 BM25 全文检索
            "metadata": {"type": "object"},               # 存储元数据
            "vector": {"type": "dense_vector", "dims": 384} # 向量维度需与嵌入模型输出一致
        }
    }
}
es_client.indices.create(index=INDEX_NAME, body=index_mapping)
print(f"已创建索引: {INDEX_NAME}")

# 5. 初始化嵌入模型（使用你已经熟悉的模型）
# 该模型的向量输出维度是 384，与上面 mapping 中的 dims 设置一致
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 6. 加载和切分文档（复用你之前的代码）
def load_documents(data_dir="docs"):
    """加载指定目录下的TXT文件"""
    documents = []
    txt_files = glob.glob(f"{data_dir}/**/*.txt", recursive=True)
    for txt_file in txt_files:
        try:
            # 尝试 UTF-8 编码
            loader = TextLoader(txt_file, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ 加载成功: {os.path.basename(txt_file)}")
        except UnicodeDecodeError:
            try:
                # 尝试 GBK 编码
                loader = TextLoader(txt_file, encoding='gbk')
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ 加载成功: {os.path.basename(txt_file)} (GBK)")
            except Exception as e:
                print(f"❌ 加载失败 {txt_file}: {e}")
    return documents

print("Step 1: Loading documents...")
raw_documents = load_documents("docs")
if not raw_documents:
    print("请将 .txt 文件放入 docs 文件夹后重试。")
    exit()

print(f"Step 2: Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_splits = text_splitter.split_documents(raw_documents)
print(f"共切分为 {len(doc_splits)} 个文本块。")

# 7. 创建 ElasticsearchStore 实例
# 并批量写入文档，ElasticsearchStore 会自动处理向量化并存入 vector 字段
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    index_name=INDEX_NAME,
    embedding=embeddings,
    es_user="elastic",
    es_password="<你的密码>", # 请替换为从 start.sh 输出的密码
)

print("Step 3: Adding documents to Elasticsearch...")
# 使用 from_documents 批量添加，这对于 ElasticsearchStore 是最高效的方式
vector_store.add_documents(doc_splits)
print("✅ 文档已成功写入 Elasticsearch。")
print(f"📊 索引 '{INDEX_NAME}' 已创建，同时支持 BM25 和向量检索。")