import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_documents(data_dir="docs"):
    """加载指定目录下的PDF和TXT文件"""
    documents = []

    # 定义每种文件对应的加载器
    loaders = {
        "pdf": PyPDFLoader,
        "txt": TextLoader,
    }

    for ext, LoaderClass in loaders.items():
        # 使用DirectoryLoader批量加载指定扩展名的文件
        loader = DirectoryLoader(
            data_dir,
            glob=f"**/*.{ext}",
            loader_cls=LoaderClass,
            show_progress=True
        )
        try:
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} {ext} files.")
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    return documents


def main():
    print("Step 1: Loading documents...")
    raw_documents = load_documents("docs")
    if not raw_documents:
        print("No documents found in 'docs/' directory.")
        return

    print("Step 2: Splitting documents...")
    # 初始化文本分割器，设置每块大小和重叠长度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(splits)} chunks.")

    print("Step 3: Creating embeddings and vector store...")
    # 加载一个轻量级的中文语义嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 创建一个Chroma向量数据库，并将文档分块存入
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"  # 指定持久化存储的目录
    )
    vectordb.persist()  # 保存到硬盘
    print("Vector store created and persisted at './chroma_db'")


if __name__ == "__main__":
    main()