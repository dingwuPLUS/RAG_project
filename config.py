import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RAGConfig:
    # 文档路径
    doc_dir: str = "./data/docs"
    # 分块策略："fixed", "recursive", "semantic"
    split_method: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 嵌入模型（HuggingFace可用）
    dense_model_name: str = "BAAI/bge-small-zh-v1.5"  # 中文首选
    # FAISS索引路径
    index_path: str = "./storage/faiss_index"
    # 混合检索权重 (dense分数 * alpha + sparse分数 * (1-alpha))
    hybrid_alpha: float = 0.7
    # 重排序模型
    rerank_model_name: str = "BAAI/bge-reranker-base"
    # 召回数量
    top_k: int = 5
    # 生成模型
    llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct"  # 显存不够可换更小
    # 设备
    device: str = "cuda" if os.system("which nvidia-smi") else "cpu"