"""
稠密向量嵌入器：基于 Sentence-Transformers 模型
"""

import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DenseEmbedder:
    """稠密向量嵌入，用于语义检索"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace 上的 Sentence-Transformer 模型名
            device: 'cuda' 或 'cpu'
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"稠密嵌入模型 {model_name} 加载完成，设备: {device}")

    def embed(self, texts: List[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        将文本列表编码为向量矩阵。

        Args:
            texts: 文本列表
            normalize: 是否归一化（L2 归一化，使余弦相似度可用内积计算）
            show_progress: 是否显示进度条

        Returns:
            numpy 数组，形状 (len(texts), dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    @property
    def dim(self) -> int:
        """向量维度"""
        return self.model.get_sentence_embedding_dimension()