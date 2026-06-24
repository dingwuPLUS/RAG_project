"""
命令行交互界面：支持多轮对话、查询增强、来源展示。
"""

import logging
from typing import Optional
import numpy as np

from generation.generator import LocalGenerator
from embedding.dense_embedder import DenseEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from dialogue.history_manager import HistoryManager
from augmentation.query_enhance import HyDEEnhancer, MultiQueryEnhancer, QueryRewriter

logger = logging.getLogger(__name__)


def run_cli(
    generator: LocalGenerator,
    embedder: DenseEmbedder,
    retriever: HybridRetriever,
    reranker: Reranker,
    history_manager: HistoryManager,
    use_hyde: bool = False,
    use_multi_query: bool = False,
):
    """
    启动命令行交互 RAG 系统。

    Args:
        generator: 本地 LLM 生成器
        embedder: 稠密向量嵌入器
        retriever: 混合检索器
        reranker: 重排序器
        history_manager: 对话历史管理器
        use_hyde: 是否启用 HyDE 增强
        use_multi_query: 是否启用多查询融合（注意不可同时启用 HyDE）
    """
    # 初始化增强器
    hyde_enhancer = HyDEEnhancer(generator, embedder) if use_hyde else None
    multi_enhancer = MultiQueryEnhancer(generator) if use_multi_query else None
    rewriter = QueryRewriter(generator)

    print("\n" + "="*50)
    print("  RAG 系统命令行交互界面")
    print("  输入 'exit' 退出，输入 'clear' 清空历史")
    print("="*50 + "\n")

    while True:
        try:
            query = input("\n用户：").strip()
            if not query:
                continue
            if query.lower() == "exit":
                print("再见！")
                break
            if query.lower() == "clear":
                history_manager.clear()
                print("对话历史已清空。")
                continue

            # 1. 查询改写（基于对话历史）
            history_context = history_manager.get_context()
            rewritten_query = rewriter.rewrite(query, history_context)
            if rewritten_query != query:
                print(f"（改写后查询：{rewritten_query}）")

            # 2. 查询增强（若开启）
            query_vectors = []
            retrieval_queries = [rewritten_query]

            if hyde_enhancer:
                hypo_doc = hyde_enhancer.generate_hypothetical_doc(rewritten_query)
                retrieval_queries = [hypo_doc]  # 用假设文档检索
                print(f"（HyDE 生成假设文档：{hypo_doc[:100]}...）")

            if multi_enhancer:
                retrieval_queries = multi_enhancer.enhance(rewritten_query)
                print(f"（生成 {len(retrieval_queries)-1} 个辅助查询）")

            # 3. 检索（支持多查询融合）
            all_docs = []
            for rq in retrieval_queries:
                q_vec = embedder.embed([rq])[0]
                docs = retriever.retrieve(rq, q_vec, top_k=10)
                all_docs.extend(docs)

            # 去重（按 chunk_id）
            seen = {}
            unique_docs = []
            for doc in all_docs:
                cid = doc.metadata.get("chunk_id")
                if cid not in seen:
                    seen[cid] = doc
                    unique_docs.append(doc)
            candidates = unique_docs[:20]

            # 4. 重排序
            top_docs = reranker.rerank(rewritten_query, candidates, top_k=5)

            # 5. 生成回答
            context_texts = [d.page_content for d in top_docs]
            prompt = generator.format_rag_prompt(
                query=rewritten_query,
                contexts=context_texts
            )

            answer = generator.generate(
                prompt,
                max_new_tokens=300,  # 缩短最大长度，正常回答无需512
                temperature=0.7,
                stop_strings=["用户：", "\n用户", "Human:", "Assistant:"]  # 遇到这些就停止
            )
            # 再次强制截断，以防 stop_strings 没有捕获
            for stop in ["用户：", "\n用户", "Human:", "Assistant:"]:
                idx = answer.find(stop)
                if idx != -1:
                    answer = answer[:idx]
                    break

            # 6. 展示结果
            print(f"\n助手：{answer}")
            print("\n参考来源：")
            for i, doc in enumerate(top_docs):
                src = doc.metadata.get("source", "未知")
                page = doc.metadata.get("page", "")
                chunk_id = doc.metadata.get("chunk_id", "")
                print(f"  [{i+1}] {src} (page: {page}, chunk: {chunk_id})")

            # 7. 更新历史
            history_manager.add(query, answer)

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理出错: {e}", exc_info=True)
            print(f"系统错误：{e}")