"""
混合检索器：结合稠密检索（FAISS）与稀疏检索（BM25），分数加权融合。
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

# 请根据你的项目结构调整以下导入路径
# 假设 FAISSStore 和 BM25Searcher 已实现
from vector_store.faiss_store import FAISSStore
from embedding.sparse_embedder import BM25Searcher
from document_loader.loaders import Document

logger = logging.getLogger(__name__)


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """对分数进行 Min-Max 归一化到 [0,1]"""
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val < 1e-8:
        return np.ones_like(scores) * 0.5
    return (scores - min_val) / (max_val - min_val)


class HybridRetriever:
    """
    混合检索器：结合稠密向量检索和 BM25 稀疏检索。
    分数融合方式为加权求和：alpha * dense_norm + (1-alpha) * sparse_norm。
    """

    def __init__(
        self,
        dense_store: FAISSStore,
        bm25_searcher: BM25Searcher,
        alpha: float = 0.7
    ):
        """
        Args:
            dense_store: FAISS 向量存储（已完成归一化索引，相似度用内积即余弦）
            bm25_searcher: BM25 检索器（需实现 search(query, k) 方法，返回 (doc_ids, scores)）
            alpha: 稠密分数权重，建议 0.6~0.8
        """
        self.dense = dense_store
        self.sparse = bm25_searcher
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        dense_query_vec: np.ndarray,
        top_k: int = 5
    ) -> List[Document]:
        """
        混合检索并返回 top_k 文档列表。

        Args:
            query: 原始查询文本（用于 BM25）
            dense_query_vec: 查询的稠密向量，形状 (dim,)，已归一化
            top_k: 最终返回的文档数

        Returns:
            List[Document] 按混合分数降序排列
        """
        # 1. 稠密检索（取 top_k * 2 扩大候选池）
        dense_docs, dense_scores = self.dense.search(dense_query_vec, k=top_k * 2)
        # 2. 稀疏检索（同样取 top_k * 2）
        sparse_ids, sparse_scores = self.sparse.search(query, k=top_k * 2)
        sparse_docs = [self.sparse.id_to_doc[i] for i in sparse_ids]

        # 3. 分数归一化
        d_norm = min_max_normalize(np.array(dense_scores))
        s_norm = min_max_normalize(np.array(sparse_scores))

        # 4. 融合分数（用 chunk_id 作为唯一标识）
        doc_map = {}  # chunk_id -> (doc, fused_score)
        for doc, score in zip(dense_docs, d_norm):
            cid = doc.metadata.get("chunk_id")
            if cid is not None:
                doc_map[cid] = [doc, self.alpha * score]

        for doc, score in zip(sparse_docs, s_norm):
            cid = doc.metadata.get("chunk_id")
            if cid is not None:
                if cid in doc_map:
                    doc_map[cid][1] += (1 - self.alpha) * score
                else:
                    doc_map[cid] = [doc, (1 - self.alpha) * score]

        # 5. 按融合分数降序排序，取 top_k
        sorted_items = sorted(doc_map.values(), key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items]