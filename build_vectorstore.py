import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# 自定义 TextLoader 以支持不同编码
class CustomTextLoader(TextLoader):
    def __init__(self, file_path, encoding='utf-8'):
        # 尝试 UTF-8，如果失败则尝试 GBK
        try:
            super().__init__(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            super().__init__(file_path, encoding='gbk')


def load_documents(data_dir="docs"):
    """加载指定目录下的PDF和TXT文件"""
    documents = []

    # 加载 PDF 文件
    if os.path.exists(data_dir):
        pdf_loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        try:
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} PDF files.")
        except Exception as e:
            print(f"Error loading PDF files: {e}")

        # 加载 TXT 文件（使用自定义编码）
        for file_path in glob.glob(f"{data_dir}/**/*.txt", recursive=True):
            try:
                # 先尝试 UTF-8
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded: {os.path.basename(file_path)} (UTF-8)")
            except UnicodeDecodeError:
                try:
                    # 再尝试 GBK
                    loader = TextLoader(file_path, encoding='gbk')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded: {os.path.basename(file_path)} (GBK)")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return documents


def main():
    print("Step 1: Loading documents...")
    raw_documents = load_documents("docs")

    if not raw_documents:
        print("\n❌ No documents found in 'docs/' directory.")
        print("Please add .pdf or .txt files to the 'docs' folder.")
        return

    print(f"\n✅ Total documents loaded: {len(raw_documents)}")

    print("Step 2: Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(splits)} chunks.")

    print("Step 3: Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector store created and persisted at './chroma_db'")


if __name__ == "__main__":
    import glob

    main()