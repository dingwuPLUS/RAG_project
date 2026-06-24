"""
FAISS 向量存储：管理稠密向量索引、文档映射及持久化。
"""

import os
import logging
import pickle
from typing import List, Tuple

import numpy as np
import faiss

# 根据你的项目结构调整 Document 导入路径
from document_loader.loaders import Document

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    基于 FAISS 的向量存储。
    默认使用 IndexFlatIP（内积），配合 L2 归一化向量即等效余弦相似度。
    """

    def __init__(self, dim: int, index_path: str = None):
        """
        Args:
            dim: 向量维度
            index_path: 索引文件前缀（不含扩展名），用于保存/加载
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 内积索引
        self.id_to_doc = {}                  # 内部 FAISS ID -> Document 对象
        self.index_path = index_path

    def add(self, vectors: np.ndarray, docs: List[Document]):
        """
        向索引中添加向量和对应文档。
        Args:
            vectors: 形状 (n, dim) 的 ndarray，通常已归一化
            docs: 对应文档列表，长度必须与 vectors 行数一致
        """
        if vectors.shape[0] != len(docs):
            raise ValueError("向量数量与文档数量不一致")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与索引维度 {self.dim} 不匹配")

        start_id = self.index.ntotal
        self.index.add(vectors.astype(np.float32))
        for i, doc in enumerate(docs):
            self.id_to_doc[start_id + i] = doc
        logger.info(f"已添加 {len(docs)} 个向量，当前总向量数: {self.index.ntotal}")

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[List[Document], List[float]]:
        """
        查询最相似的 k 个文档。
        Args:
            query_vec: 查询向量，形状 (dim,) 或 (1, dim)，应已归一化
            k: 返回数量
        Returns:
            (docs, scores) 文档列表和对应的相似度分数（内积值）
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self.index.search(query_vec.astype(np.float32), k)
        docs = []
        valid_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx in self.id_to_doc:
                docs.append(self.id_to_doc[idx])
                valid_scores.append(float(score))
        return docs, valid_scores

    def save(self):
        """持久化索引和文档映射"""
        if not self.index_path:
            raise ValueError("未设置 index_path，无法保存")
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.id_to_doc, f)
        logger.info(f"索引已保存至 {self.index_path}.faiss / .pkl")

    def load(self):
        """从文件加载索引和文档映射"""
        if not self.index_path:
            raise ValueError("未设置 index_path，无法加载")
        faiss_file = f"{self.index_path}.faiss"
        pkl_file = f"{self.index_path}.pkl"
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"索引文件不存在: {faiss_file} / {pkl_file}")

        self.index = faiss.read_index(faiss_file)
        with open(pkl_file, "rb") as f:
            self.id_to_doc = pickle.load(f)
        self.dim = self.index.d
        logger.info(f"索引已加载，共 {self.index.ntotal} 个向量")

    def __len__(self):
        return self.index.ntotal