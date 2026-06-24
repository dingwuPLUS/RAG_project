"""
查询增强模块：HyDE、多查询生成、查询改写
所有增强器需要外部注入生成器（generate 方法）和嵌入器（embed 方法）
"""

import logging
from typing import List, Callable, Optional
import numpy as np

# 假设你的项目中已有 Document 定义
from document_loader.loaders import Document

logger = logging.getLogger(__name__)


class HyDEEnhancer:
    """
    HyDE (Hypothetical Document Embeddings)：
    使用 LLM 为查询生成一个假设答案，然后用该答案的向量去检索，
    以弥合问题-文档之间的语义差距。
    """

    def __init__(self, generator: Callable, embedder: Callable):
        """
        Args:
            generator: 具有 generate(prompt, max_new_tokens=..., temperature=...) 方法的对象
            embedder: 具有 embed(texts: List[str]) -> np.ndarray 方法的对象
        """
        self.generator = generator
        self.embedder = embedder

    def generate_hypothetical_doc(self, query: str) -> str:
        """生成假设文档（简短答案）"""
        prompt = (
            f"请为下面的问题写一段简短的答案，不要解释，只给出答案：\n{query}"
        )
        # 低温生成确保稳定性
        return self.generator.generate(prompt, max_new_tokens=256, temperature=0.3)

    def enhance(self, query: str) -> str:
        """
        返回用于嵌入的文本（即假设文档），后续用此文本的向量代替原问题向量检索。
        """
        hypo_doc = self.generate_hypothetical_doc(query)
        logger.info(f"HyDE 生成假设文档: {hypo_doc[:100]}...")
        return hypo_doc


class MultiQueryEnhancer:
    """
    多查询生成：利用 LLM 生成 3~5 个同义改写查询，
    分别检索后合并结果并去重。
    """

    def __init__(self, generator: Callable, num_queries: int = 4):
        """
        Args:
            generator: 生成器对象
            num_queries: 生成的改写查询数量
        """
        self.generator = generator
        self.num_queries = num_queries

    def generate_alternative_queries(self, query: str) -> List[str]:
        """生成多个改写查询"""
        prompt = (
            f"请为以下问题生成 {self.num_queries} 个不同表达方式的等价问题，"
            f"每行一个，不要编号，不要解释。\n原问题：{query}"
        )
        response = self.generator.generate(prompt, max_new_tokens=256, temperature=0.8)
        # 按行拆分，过滤空行
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # 限制数量
        alternatives = lines[:self.num_queries]
        if not alternatives:
            # 如果生成失败，回退为原查询
            alternatives = [query]
        logger.info(f"多查询生成: {alternatives}")
        return alternatives

    def enhance(self, query: str) -> List[str]:
        """返回原查询 + 生成的多个改写查询"""
        alts = self.generate_alternative_queries(query)
        # 包含原查询，避免丢失
        return [query] + alts


class QueryRewriter:
    """
    查询改写：基于对话历史，将模糊的用户问题改写为独立的、明确的查询。
    适用于多轮对话场景。
    """

    def __init__(self, generator: Callable):
        self.generator = generator

    def rewrite(self, query: str, history_text: Optional[str] = None) -> str:
        """
        根据历史对话改写当前问题。
        Args:
            query: 当前用户问题
            history_text: 对话历史的文本表示（如 HistoryManager.get_context() 输出）
        Returns:
            改写后的问题
        """
        if not history_text or history_text.strip() == "":
            # 无历史则原样返回
            return query

        prompt = (
            "根据对话历史，将用户的问题改写为一个独立、完整的问题。"
            "如果问题已足够明确，直接返回原问题。\n\n"
            f"对话历史：\n{history_text}\n\n"
            f"用户问题：{query}\n"
            "改写问题："
        )
        rewritten = self.generator.generate(prompt, max_new_tokens=128, temperature=0.3)
        return rewritten.strip() or query