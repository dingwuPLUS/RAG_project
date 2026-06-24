"""
RAG 全栈系统主入口
将文档加载、分块、嵌入、检索、生成、对话、UI 全部串联
"""

import os
import logging
import argparse

from config import RAGConfig
from document_loader.loaders import load_documents
from text_splitter import get_splitter
from embedding.dense_embedder import DenseEmbedder
from embedding.sparse_embedder import BM25Searcher
from vector_store.faiss_store import FAISSStore
from retrieval import HybridRetriever, Reranker
from generation import LocalGenerator
from dialogue import HistoryManager, create_summary_function
from augmentation import HyDEEnhancer, MultiQueryEnhancer, QueryRewriter
from ui import run_cli, launch_web_ui

logging.getLogger("httpx").setLevel(logging.WARNING)     # 隐藏 HTTP 请求日志
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)  # 隐藏模型加载细节
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_or_load_index(cfg: RAGConfig, embedder: DenseEmbedder):
    """
    构建或加载 FAISS 索引和 BM25 索引。
    如果索引文件存在则直接加载，否则重新构建。
    """
    faiss_path = cfg.index_path + ".faiss"
    pkl_path = cfg.index_path + ".pkl"

    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        logger.info("发现已有索引，正在加载...")
        # 需要先知道向量维度，可以通过加载 FAISS 文件获取
        import faiss
        index = faiss.read_index(faiss_path)
        dim = index.d
        store = FAISSStore(dim=dim, index_path=cfg.index_path)
        store.load()
        # BM25 也需要重建（因为需要文档对象）
        # 这里我们假设索引与文档分块结果绑定，需要重新从文档构建
        # 如果 BM25 索引也要持久化，可自行扩展（pickle 整个对象）
        logger.warning("BM25 索引需重新构建，请确保文档仍在原路径")
        # 为简化，这里选择重新构建所有索引（保证一致）
        logger.info("重新构建全部索引以确保一致性...")
        return None, None  # 返回 None 触发重建
    else:
        logger.info("未找到索引，将全新构建")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="RAG 全栈系统")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli",
                        help="交互模式：命令行(cli) 或 Web界面(web)")
    parser.add_argument("--doc_dir", type=str, default="./docs",
                        help="文档目录")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新构建索引")
    args = parser.parse_args()

    # ---------- 1. 加载配置 ----------
    cfg = RAGConfig()
    cfg.doc_dir = args.doc_dir

    # ---------- 2. 加载文档并分块 ----------
    logger.info("加载文档...")
    docs = load_documents(cfg.doc_dir)
    if not docs:
        logger.error(f"在 {cfg.doc_dir} 中没有找到支持的文档，请放入 PDF/DOCX/TXT/MD 等文件。")
        return

    logger.info(f"共加载 {len(docs)} 个文档片段")

    # 选择分块策略并分块
    splitter = get_splitter(
        cfg.split_method,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap
    )
    chunks = splitter(docs)
    logger.info(f"分块完成，共 {len(chunks)} 个块")

    # ---------- 3. 初始化嵌入模型 ----------
    logger.info("加载嵌入模型...")
    embedder = DenseEmbedder(cfg.dense_model_name, device=cfg.device)

    # ---------- 4. 构建向量存储 ----------
    logger.info("构建/加载向量索引...")
    if args.rebuild or not (os.path.exists(cfg.index_path + ".faiss") and os.path.exists(cfg.index_path + ".pkl")):
        # 计算所有块的嵌入
        chunk_texts = [c.page_content for c in chunks]
        dense_vectors = embedder.embed(chunk_texts)
        store = FAISSStore(dim=dense_vectors.shape[1], index_path=cfg.index_path)
        store.add(dense_vectors, chunks)
        store.save()
        logger.info("FAISS 索引已保存")

        # BM25 索引
        bm25 = BM25Searcher()
        bm25.index(chunks)
        # 这里可将 BM25 对象 pickle 保存，简化起见每次重建
    else:
        # 直接从文件加载
        import faiss
        index = faiss.read_index(cfg.index_path + ".faiss")
        store = FAISSStore(dim=index.d, index_path=cfg.index_path)
        store.load()
        # BM25 重建（需要文档对象）
        bm25 = BM25Searcher()
        bm25.index(chunks)

    # ---------- 5. 初始化检索器与重排序 ----------
    logger.info("加载重排序模型...")
    retriever = HybridRetriever(store, bm25, alpha=cfg.hybrid_alpha)
    reranker = Reranker(cfg.rerank_model_name, device=cfg.device)

    # ---------- 6. 初始化生成器 ----------
    logger.info("加载语言模型...")
    generator = LocalGenerator(
        cfg.llm_model_name,
        device=cfg.device,
        load_in_4bit=True if cfg.device == "cuda" else False,
        use_streamer=False  # CLI 下建议 False，Web 下可 True
    )

    # ---------- 7. 初始化对话管理 ----------
    summary_fn = create_summary_function(generator)
    history_mgr = HistoryManager(
        max_turns=5,
        max_tokens_approx=1500,
        summarizer=summary_fn
    )

    # ---------- 8. 启动交互 ----------
    logger.info("系统就绪！")
    if args.mode == "web":
        launch_web_ui(share=False)
    else:
        run_cli(
            generator=generator,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            history_manager=history_mgr,
            use_hyde=True,      # 可在 cli 内部改为 True 体验
            use_multi_query=True
        )


if __name__ == "__main__":
    main()