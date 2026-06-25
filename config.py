import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RAGConfig:
    # 文档路径
    doc_dir: str = "./docs"
    # 分块策略："fixed", "recursive", "semantic", "markdown"
    # split_method: str = "recursive"
    # split_method: str = "semantic"
    split_method: str = "markdown"
    chunk_split_path: str = "./chunks"
    chunk_size: int = 384
    chunk_overlap: int = 64
    # 嵌入模型（HuggingFace可用）
    dense_model_name: str = "BAAI/bge-small-zh-v1.5"  # 中文首选
    rebuild: bool = True
    # FAISS索引路径
    index_path: str = "./storage/faiss_index"
    # 混合检索权重 (dense分数 * alpha + sparse分数 * (1-alpha))
    hybrid_alpha: float = 0.7
    # 重排序模型
    rerank_model_name: str = "BAAI/bge-reranker-base"
    # 召回数量
    top_k: int = 5
    # 生成模型
    # llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct"  # 显存不够可换更小
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # 显存不够可换更小

    use_api: bool = False
    api_key: str = ""  # 从环境变量读取更安全
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = "gpt-3.5-turbo"

    # Ollama 接入
    use_ollama: bool = True  # 使用 Ollama 替换本地加载
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "deepseek-r1:7b"  # 你在 Ollama 中下载的模型名
    # 设备
    device: str = "cuda" if os.system("which nvidia-smi") else "cpu"