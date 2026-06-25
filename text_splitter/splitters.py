"""
文本分块器：固定大小滑动窗口、递归字符分割、语义分块
所有函数遵循统一接口：输入 List[Document] -> 输出 List[Document]
"""

import re
import os
import shutil
import logging
from typing import List, Callable, Optional
import numpy as np
from pathlib import Path

from nltk.draw import cfg

# 请根据你的项目结构调整 Document 导入路径
from document_loader.loaders import Document

logger = logging.getLogger(__name__)

# ==================== Markdown 分块 ====================
def markdown_split(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    save_dir: str = "./chunks"
) -> List[Document]:
    """
    基于Markdown标题层级的分块器。
    1. 按标题（#、##、###等）划分大段
    2. 若某段长度超过chunk_size，再用递归字符分割细化
    3. 保留标题作为块的起始，确保语义连贯
    """
    import re
    chunks = []
    for doc in documents:
        text = doc.page_content
        # 匹配所有标题行（#开头，后面有空格）
        title_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        # 构建标题索引：[(level, title_text, start_line, end_line)]
        sections = []
        current_start = 0
        for i, line in enumerate(lines):
            match = re.match(title_pattern, line)
            if match:
                # 保存上一段
                if i > current_start:
                    sections.append((None, None, current_start, i))
                # 开始新段落
                sections.append((len(match.group(1)), match.group(2).strip(), i, i+1))
                current_start = i
            # 非标题行，继续累积
        if current_start < len(lines):
            sections.append((None, None, current_start, len(lines)))

        # 合并相邻的同一标题级别（同级别标题之间内容合并到一个块）
        merged_sections = []
        for level, title, start, end in sections:
            if level is not None:  # 标题行
                merged_sections.append({
                    'level': level,
                    'title': title,
                    'start': start,
                    'end': end,
                    'content': ''
                })
            else:  # 内容块
                if merged_sections:
                    merged_sections[-1]['end'] = end
                    merged_sections[-1]['content'] = '\n'.join(lines[start:end]).strip()
                else:
                    # 文档开头无标题
                    merged_sections.append({
                        'level': 0,
                        'title': '',
                        'start': start,
                        'end': end,
                        'content': '\n'.join(lines[start:end]).strip()
                    })

        # 按标题级别合并：低级别标题（如#）的内容包含其下的所有子标题
        # 这里采用简单策略：直接按标题切分，每个标题及其下内容作为一个独立块
        for sec in merged_sections:
            if sec['title']:
                full_text = f"# {sec['title']}\n{sec['content']}" if sec['level'] > 0 else sec['content']
            else:
                full_text = sec['content']

            if not full_text.strip():
                continue

            # 如果内容超过chunk_size，用递归字符分割进一步切分
            if len(full_text) > chunk_size:
                sub_chunks = recursive_split(
                    [Document(full_text, doc.metadata)],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "。"],
                    save_dir=save_dir
                )
                for sc in sub_chunks:
                    new_meta = {**doc.metadata, "chunk_id": len(chunks), "method": "markdown_title"}
                    chunks.append(Document(sc.page_content, new_meta))
            else:
                new_meta = {**doc.metadata, "chunk_id": len(chunks), "method": "markdown_title"}
                chunks.append(Document(full_text, new_meta))

    logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个块")
    save_chunks_to_txt(chunks, os.path.join(save_dir, "markdown"), prefix="md")
    return chunks

# ==================== 固定大小分块 ====================
def fixed_size_split(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    save_dir:str = "./chunks"
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
    save_chunks_to_txt(chunks, os.path.join(save_dir, "fixed_size"), prefix="fixed")
    return chunks


# ==================== 递归字符分割 ====================
def recursive_split(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
    save_dir:str = "./chunks"
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
    save_chunks_to_txt(all_chunks, os.path.join(save_dir, "recursive"), prefix="recursive")
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

def clear_and_create_dir(output_dir: str):
    """清空并重新创建输出目录"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def save_chunks_to_txt(chunks: List[Document], output_dir: str, prefix: str = "chunk"):
    """
    将分块结果保存为多个文本文件，文件名为 prefix_索引.txt
    每个文件包含元信息和内容，便于查看。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        filepath = os.path.join(output_dir, f"{prefix}_{idx:04d}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write("=" * 50 + "\n")
            f.write(chunk.page_content)
    print(f"✅ 已保存 {len(chunks)} 个块到目录：{output_dir}")


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
        min_sentences: int = 1,
        save_dir: str = "./chunks"
    ):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.save_dir = save_dir

        # 初始化 nltk 分句工具
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
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
        save_chunks_to_txt(chunks, os.path.join(self.save_dir, "semantic"), prefix="semantic")
        return chunks