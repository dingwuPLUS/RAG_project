"""
Gradio Web 界面：提供文件上传、知识库构建、问答交互。
"""

import logging
import time
from typing import List, Optional
import gradio as gr
import numpy as np

from generation.generator import LocalGenerator
from embedding.dense_embedder import DenseEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from document_loader.loaders import load_documents, Document
from text_splitter.splitters import recursive_split  # 假设分块模块存在
from vector_store.faiss_store import FAISSStore
from embedding.sparse_embedder import BM25Searcher
from dialogue.history_manager import HistoryManager
from augmentation.query_enhance import QueryRewriter, HyDEEnhancer

logger = logging.getLogger(__name__)


class RAGWebUI:
    """封装 RAG 系统各组件，供 Gradio 调用"""

    def __init__(self):
        self.generator: Optional[LocalGenerator] = None
        self.embedder: Optional[DenseEmbedder] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[Reranker] = None
        self.dense_store: Optional[FAISSStore] = None
        self.bm25: Optional[BM25Searcher] = None
        self.history_manager = HistoryManager(max_turns=5)
        self.rewriter = None
        self.is_ready = False

    def load_models(self, llm_model: str, embed_model: str, rerank_model: str, device: str):
        """加载模型（耗时操作，在启动或按钮触发时调用）"""
        try:
            self.generator = LocalGenerator(model_name=llm_model, device=device)
            self.embedder = DenseEmbedder(model_name=embed_model, device=device)
            self.reranker = Reranker(model_name=rerank_model, device=device)
            self.rewriter = QueryRewriter(self.generator)
            self.is_ready = True
            return "✅ 模型加载成功！现在可以上传文档并提问。"
        except Exception as e:
            return f"❌ 模型加载失败：{str(e)}"

    def build_knowledge_base(self, uploaded_files, chunk_size: int, chunk_overlap: int):
        """从上传的文件构建知识库"""
        if not self.is_ready:
            return "请先加载模型！"

        if not uploaded_files:
            return "请上传至少一个文档。"

        try:
            # 1. 加载文档
            all_docs = []
            for file in uploaded_files:
                # file 是文件路径（Gradio 上传后临时存储）
                docs = load_documents(file.name)
                all_docs.extend(docs)

            if not all_docs:
                return "未能从文件中提取到文本。"

            # 2. 分块
            chunks = recursive_split(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # 3. 嵌入
            chunk_texts = [c.page_content for c in chunks]
            dense_vectors = self.embedder.embed(chunk_texts)

            # 4. 构建 FAISS 索引
            self.dense_store = FAISSStore(dim=dense_vectors.shape[1])
            self.dense_store.add(dense_vectors, chunks)

            # 5. 构建 BM25 索引
            self.bm25 = BM25Searcher()
            self.bm25.index(chunks)

            # 6. 初始化检索器
            self.retriever = HybridRetriever(self.dense_store, self.bm25, alpha=0.7)

            return f"✅ 知识库构建完成！共 {len(chunks)} 个文档块。"
        except Exception as e:
            return f"❌ 构建失败：{str(e)}"

    def chat(self, query: str, history: List[List[str]], enable_hyde: bool):
        """处理用户提问，返回回答和更新后的历史"""
        if not self.is_ready or not self.retriever:
            yield history + [[query, "系统未就绪，请先加载模型并构建知识库。"]], history + [[query, "系统未就绪"]]
            return

        try:
            # 1. 查询改写
            history_context = self.history_manager.get_context()
            rewritten_query = self.rewriter.rewrite(query, history_context)

            # 2. HyDE 增强
            retrieval_query = rewritten_query
            if enable_hyde:
                hyde_enhancer = HyDEEnhancer(self.generator, self.embedder)
                retrieval_query = hyde_enhancer.generate_hypothetical_doc(rewritten_query)

            # 3. 检索
            q_vec = self.embedder.embed([retrieval_query])[0]
            candidates = self.retriever.retrieve(retrieval_query, q_vec, top_k=10)

            # 4. 重排序
            top_docs = self.reranker.rerank(rewritten_query, candidates, top_k=5)
            context_texts = [d.page_content for d in top_docs]

            # 5. 生成回答
            prompt = self.generator.format_rag_prompt(rewritten_query, context_texts)
            answer = self.generator.generate(prompt, max_new_tokens=512)

            # 6. 构建显示文本（回答 + 来源）
            display_answer = answer
            if top_docs:
                sources = "\n\n**参考来源：**\n"
                for i, doc in enumerate(top_docs):
                    src = doc.metadata.get("source", "未知")
                    page = doc.metadata.get("page", "")
                    sources += f"- [{i+1}] {src}"
                    if page:
                        sources += f" (页码: {page})"
                    sources += "\n"
                display_answer += sources

            # 7. 更新历史
            self.history_manager.add(query, answer)

            # 返回给 Gradio 的对话历史
            new_history = history + [[query, display_answer]]
            yield new_history, new_history

        except Exception as e:
            logger.error(f"对话出错：{e}", exc_info=True)
            error_msg = f"处理错误：{str(e)}"
            yield history + [[query, error_msg]], history + [[query, error_msg]]


def create_gradio_app():
    """创建 Gradio 界面"""
    app = RAGWebUI()

    with gr.Blocks(title="RAG 全栈系统") as demo:
        gr.Markdown("# RAG 全栈系统")
        gr.Markdown("支持 PDF/DOCX/TXT/MD 等多种文档，实现检索增强生成")

        with gr.Tab("1. 模型加载"):
            with gr.Row():
                llm_model = gr.Textbox(label="LLM 模型名称", value="Qwen/Qwen2-1.5B-Instruct")
                embed_model = gr.Textbox(label="嵌入模型名称", value="BAAI/bge-small-zh-v1.5")
                rerank_model = gr.Textbox(label="重排序模型名称", value="BAAI/bge-reranker-base")
            device = gr.Radio(label="设备", choices=["cuda", "cpu"], value="cuda")
            load_btn = gr.Button("加载模型", variant="primary")
            load_output = gr.Textbox(label="状态")

            load_btn.click(
                fn=app.load_models,
                inputs=[llm_model, embed_model, rerank_model, device],
                outputs=load_output
            )

        with gr.Tab("2. 知识库构建"):
            with gr.Row():
                files = gr.File(label="上传文档", file_count="multiple")
            with gr.Row():
                chunk_size = gr.Slider(label="分块大小", minimum=100, maximum=1500, value=500, step=50)
                chunk_overlap = gr.Slider(label="重叠长度", minimum=0, maximum=200, value=50, step=10)
            build_btn = gr.Button("构建知识库", variant="primary")
            build_output = gr.Textbox(label="状态")

            build_btn.click(
                fn=app.build_knowledge_base,
                inputs=[files, chunk_size, chunk_overlap],
                outputs=build_output
            )

        with gr.Tab("3. 对话"):
            enable_hyde = gr.Checkbox(label="启用 HyDE 增强", value=False)
            chatbot = gr.Chatbot(label="对话历史", height=500)
            msg = gr.Textbox(label="输入问题", placeholder="在此输入你的问题...")
            clear_btn = gr.Button("清空对话")
            state = gr.State([])

            def respond(message, chat_history, hyde_enabled, chat_state):
                """流式或一次性生成回答，Gradio 中简单处理为一次性"""
                chat_history = chat_history or []
                for new_history, updated_state in app.chat(message, chat_history, hyde_enabled):
                    yield new_history, updated_state

            msg.submit(
                fn=respond,
                inputs=[msg, chatbot, enable_hyde, state],
                outputs=[chatbot, state]
            ).then(lambda: "", None, msg)

            clear_btn.click(
                fn=lambda: ([], []),
                outputs=[chatbot, state]
            )

    return demo


def launch_web_ui(share: bool = False, server_port: int = 7860):
    """启动 Gradio Web 服务"""
    demo = create_gradio_app()
    demo.launch(share=share, server_port=server_port)