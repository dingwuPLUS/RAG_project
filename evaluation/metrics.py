"""
RAG 系统评估指标：
- 检索质量：Hit Rate@k, MRR (Mean Reciprocal Rank)
- 生成质量：基于 Ragas 的 faithfulness, answer relevancy, context precision 等
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


# ------------------ 检索评估 ------------------
def hit_rate(
        retrieved_ids: List[List[Any]],
        relevant_ids: List[List[Any]],
        k: Optional[int] = None
) -> float:
    """
    计算 Hit Rate@k。

    Args:
        retrieved_ids: 每个查询的召回文档 ID 列表，按分数降序排列，如 [[1,2,3], [4,5]]
        relevant_ids: 每个查询的真实相关文档 ID 列表，如 [[1], [4,6]]
        k: 只考虑前 k 个结果，None 表示使用全部。

    Returns:
        平均命中率 (0~1)
    """
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("retrieved_ids 和 relevant_ids 长度必须一致")
    if not retrieved_ids:
        return 0.0

    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k = retrieved[:k] if k is not None else retrieved
        if any(doc_id in set(relevant) for doc_id in top_k):
            hits += 1
    return hits / len(retrieved_ids)


def mrr(
        retrieved_ids: List[List[Any]],
        relevant_ids: List[List[Any]],
        k: Optional[int] = None
) -> float:
    """
    计算 Mean Reciprocal Rank (MRR)。

    Args:
        retrieved_ids: 每个查询的召回文档 ID 列表，按分数降序排列。
        relevant_ids: 每个查询的真实相关文档 ID 列表。
        k: 只考虑前 k 个结果，None 表示使用全部。

    Returns:
        MRR 值 (0~1)
    """
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("retrieved_ids 和 relevant_ids 长度必须一致")
    if not retrieved_ids:
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k = retrieved[:k] if k is not None else retrieved
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


def retrieval_report(
        retrieved_ids: List[List[Any]],
        relevant_ids: List[List[Any]],
        k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    生成检索评估报告：不同 k 下的 Hit Rate 和 MRR。
    """
    report = {}
    for k in k_values:
        report[f"hit_rate@{k}"] = hit_rate(retrieved_ids, relevant_ids, k)
        report[f"mrr@{k}"] = mrr(retrieved_ids, relevant_ids, k)
    # 全量
    report["hit_rate"] = hit_rate(retrieved_ids, relevant_ids)
    report["mrr"] = mrr(retrieved_ids, relevant_ids)
    return report


# ------------------ 生成评估（基于 Ragas）------------------
def _check_ragas():
    try:
        import ragas
        return True
    except ImportError:
        raise ImportError(
            "请安装 ragas 库: pip install ragas"
        )


def evaluate_generation(
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    使用 Ragas 评估生成质量。

    Args:
        questions: 问题列表。
        answers: RAG 系统生成的回答列表。
        contexts_list: 每个回答对应的检索上下文列表（每个元素是字符串列表）。
        ground_truths: （可选）标准答案列表。
        metrics: 要计算的指标列表，可选 'faithfulness','answer_relevancy',
                 'context_precision','context_recall','answer_correctness'等。
                 如果为 None，则默认计算所有支持的指标。

    Returns:
        字典，指标名 -> 得分
    """
    _check_ragas()
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset

    # 准备数据
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if ground_truths is not None:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    # 选择指标
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }
    if metrics is None:
        # 默认：不使用需要 ground_truth 的指标，如果提供了 ground_truth 则加入
        selected = [faithfulness, answer_relevancy, context_precision, context_recall]
        if ground_truths is not None:
            selected.append(answer_correctness)
    else:
        selected = [metric_map[m] for m in metrics if m in metric_map]
        if not selected:
            raise ValueError("未选择任何有效指标")

    # 运行评估
    result = evaluate(dataset, metrics=selected)
    # 转换为字典
    scores = {name: float(score) for name, score in result.items()}
    logger.info(f"生成评估结果: {scores}")
    return scores