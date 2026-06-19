"""
文本分块器：固定大小滑动窗口、递归字符分割、语义分块
所有函数遵循统一接口：输入 List[Document] -> 输出 List[Document]
"""

import re
import logging
from typing import List, Callable, Optional
import numpy as np

# 请根据你的项目结构调整 Document 导入路径
from document_loader.loaders import Document

logger = logging.getLogger(__name__)


# ==================== 固定大小分块 ====================
def fixed_size_split(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    固定大小滑动窗口分块。
    按字符数切割，相邻块之间保持 overlap 字符重叠。
    """
    chunks = []
    for doc in documents:
        text = doc.page_content
        # 若文本短于 chunk_size，直接保留
        if len(text) <= chunk_size:
            new_meta = {**doc.metadata, "chunk_id": len(chunks)}
            chunks.append(Document(text, new_meta))
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                new_meta = {
                    **doc.metadata,
                    "chunk_id": len(chunks),
                    "chunk_start": start,
                    "chunk_end": end
                }
                chunks.append(Document(chunk_text, new_meta))
                # 移动窗口，重叠部分不超出文本长度
                start += chunk_size - chunk_overlap
                if start >= len(text):
                    break
    logger.info(f"固定大小分块完成，共生成 {len(chunks)} 个块")
    return chunks


# ==================== 递归字符分割 ====================
def recursive_split(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    递归字符分割：按分隔符优先级逐级切分，保持语义完整性。
    默认分隔符：["\n\n", "\n", "。", ".", "！", "？", " ", ""]
    chunk_overlap 参数会尽量在块之间保留重叠字符（通过滑动窗口后处理实现）。
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", "！", "？", " ", ""]

    all_chunks = []
    for doc in documents:
        text = doc.page_content
        # 第一步：按分隔符递归拆分为基本段落（尽可能不大于 chunk_size）
        raw_chunks = _recursive_split_text(text, separators, chunk_size)

        # 第二步：合并较小的段落，并叠加滑动窗口重叠
        merged_chunks = _merge_with_overlap(raw_chunks, chunk_size, chunk_overlap)

        for chunk_text in merged_chunks:
            new_meta = {**doc.metadata, "chunk_id": len(all_chunks)}
            all_chunks.append(Document(chunk_text, new_meta))
    logger.info(f"递归分块完成，共生成 {len(all_chunks)} 个块")
    return all_chunks


def _recursive_split_text(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """
    递归核心：对于超长文本，依次使用 separators 中的分隔符进行切割，
    直到所有片段长度 <= chunk_size 或分隔符用尽。
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    # 取当前第一个分隔符
    sep = separators[0]
    remaining_seps = separators[1:]

    # 如果分隔符为空字符串，则按字符硬切
    if sep == "":
        return _split_by_character(text, chunk_size)

    # 按分隔符分割
    parts = text.split(sep)
    final_parts = []
    for part in parts:
        if len(part) <= chunk_size:
            final_parts.append(part)
        else:
            # 当前部分仍然过长，使用下一级分隔符递归
            if remaining_seps:
                final_parts.extend(_recursive_split_text(part, remaining_seps, chunk_size))
            else:
                # 没有更多分隔符，按字符硬切
                final_parts.extend(_split_by_character(part, chunk_size))
    # 去掉空字符串（可能由连续分隔符产生），但保留有意义的空行？这里简单过滤
    return [p for p in final_parts if p]


def _split_by_character(text: str, chunk_size: int) -> List[str]:
    """按字符长度硬切分（不重叠），作为最底层分块手段"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def _merge_with_overlap(raw_chunks: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    将较小的块合并，使每个块尽量接近 chunk_size，并在相邻块之间保留 overlap 个字符的重叠。
    这是一个带滑动窗口的合并算法。
    """
    if not raw_chunks:
        return []

    # 先将原始块拼接成完整文本，然后用滑动窗口重新切分（最简单且保证重叠）
    # 但这会丢失一些边界信息。更好的做法：合并小块后，用滑动窗口在合并后的大块上切。
    # 此处采用一种折中：先把小块合并到接近 chunk_size，然后在相邻块间重叠取文本。
    merged = []
    buffer = ""
    for chunk in raw_chunks:
        if len(buffer) + len(chunk) <= chunk_size:
            buffer += chunk
        else:
            if buffer:
                merged.append(buffer)
            # 重叠处理：让新块从上一个块的末尾开始（保留 overlap 字符重叠）
            if overlap > 0 and buffer:
                # buffer 的尾部作为新块的开头
                buffer = buffer[-overlap:] + chunk
                # 如果 buffer 太长，可能需要再切分；这里简单处理，后续可优化
                while len(buffer) > chunk_size:
                    merged.append(buffer[:chunk_size])
                    buffer = buffer[chunk_size - overlap:]
            else:
                buffer = chunk
    if buffer:
        merged.append(buffer)
    # 若最后一个块太短，可以和前一个合并（但会破坏重叠，这里选择保留）
    return merged


# ==================== 语义分块 ====================
class SemanticSplitter:
    """
    基于句子嵌入相似度的语义分块器。
    在相邻句子相似度低于阈值处切分，确保块内语义连贯。
    需要提供 embed_fn，输入为文本列表，返回 numpy 向量数组 (n, dim)。
    """
    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        similarity_threshold: float = 0.8,
        min_sentences: int = 1
    ):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.min_sentences = min_sentences

        # 初始化 nltk 分句工具
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            self.sent_tokenize = nltk.sent_tokenize
        except ImportError:
            raise ImportError("请安装 nltk: pip install nltk")

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text = doc.page_content
            sentences = self.sent_tokenize(text)
            if not sentences:
                continue
            if len(sentences) <= self.min_sentences:
                new_meta = {**doc.metadata, "chunk_id": len(chunks), "method": "semantic"}
                chunks.append(Document(text, new_meta))
                continue

            # 计算句子嵌入并归一化
            embeddings = self.embed_fn(sentences)  # (n, dim)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-8, None)

            # 计算相邻句子余弦相似度
            similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

            # 确定分割点：相似度低于阈值的位置
            split_points = [0]
            for i, sim in enumerate(similarities):
                if sim < self.threshold:
                    split_points.append(i + 1)  # 在句子 i+1 之前切分
            split_points.append(len(sentences))

            # 按分割点合并句子
            for start, end in zip(split_points[:-1], split_points[1:]):
                chunk_sentences = sentences[start:end]
                chunk_text = " ".join(chunk_sentences)
                if chunk_text.strip():
                    new_meta = {
                        **doc.metadata,
                        "chunk_id": len(chunks),
                        "method": "semantic",
                        "sentences_range": f"{start}-{end-1}"
                    }
                    chunks.append(Document(chunk_text.strip(), new_meta))
        logger.info(f"语义分块完成，共生成 {len(chunks)} 个块")
        return chunks