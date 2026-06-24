"""
稀疏检索（BM25）：基于 rank_bm25 库，支持中文分词
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from document_loader.loaders import Document

logger = logging.getLogger(__name__)


# ------------------ 中文分词工具（可选）------------------
def tokenize_chinese(text: str) -> List[str]:
    """使用 jieba 进行中文分词（若安装）"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 回退到简单按字分割
        logger.warning("jieba 未安装，中文将按字分割")
        return list(text)


def tokenize_default(text: str) -> List[str]:
    """通用分词：英文按空格，中文调用 tokenize_chinese"""
    # 简单判断是否含中文
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return tokenize_chinese(text)
    else:
        return text.lower().split()


class BM25Searcher:
    """
    基于 BM25 的稀疏检索器。
    对文档列表建立索引，支持查询时返回 doc_id 和分数。
    """

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: 分词函数，输入文本字符串，返回词列表。
                       默认使用 tokenize_default（jieba中文/空格英文）
        """
        self.tokenizer = tokenizer or tokenize_default
        self.bm25 = None
        self.documents: List[Document] = []          # 原始 Document 列表
        self.id_to_doc: Dict[int, Document] = {}     # 内部 id -> Document 映射
        self.doc_tokens: List[List[str]] = []        # 每个文档的分词结果

    def index(self, documents: List[Document]):
        """
        构建 BM25 索引。

        Args:
            documents: 文档列表（每个 Document 有 page_content）
        """
        if not documents:
            logger.warning("文档列表为空，跳过 BM25 索引构建")
            return

        self.documents = documents
        self.id_to_doc = {}
        self.doc_tokens = []
        for i, doc in enumerate(documents):
            self.id_to_doc[i] = doc
            tokens = self.tokenizer(doc.page_content)
            self.doc_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.doc_tokens)
        logger.info(f"BM25 索引构建完成，文档数: {len(self.doc_tokens)}")

    def search(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        查询并返回 top-k 文档的内部 ID 和分数。

        Args:
            query: 查询字符串
            k: 返回数量

        Returns:
            (doc_ids, scores) 两个列表，长度可能小于 k（若索引文档不足）
        """
        if not self.bm25 or not self.doc_tokens:
            return [], []

        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        # 按分数降序取 top-k
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # 只取 top-k 索引
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        result_ids = [int(i) for i in top_indices]
        result_scores = [float(scores[i]) for i in top_indices]
        return result_ids, result_scores

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """直接返回 Document 列表"""
        ids, _ = self.search(query, k)
        return [self.id_to_doc[i] for i in ids if i in self.id_to_doc]