"""
重排序器：使用 Cross-Encoder 对候选文档重新打分。
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import CrossEncoder

from document_loader.loaders import Document

logger = logging.getLogger(__name__)


class Reranker:
    """
    基于 Cross-Encoder 的重排序器。
    对给定查询与候选文档列表计算相关性分数，返回 top_k 个文档。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace 上的 Cross-Encoder 模型名称
            device: 'cuda' 或 'cpu'
        """
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"重排序模型 {model_name} 已加载")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        对候选文档列表重排序。

        Args:
            query: 原始查询
            documents: 候选文档列表
            top_k: 返回的文档数

        Returns:
            重排序后的文档列表（按分数降序）
        """
        if not documents:
            return []

        # 构造 (query, document_text) 对
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)  # 返回 float 列表或 ndarray

        # 排序
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        reranked = [documents[i] for i in sorted_indices]
        return reranked