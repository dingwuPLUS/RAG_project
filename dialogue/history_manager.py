"""
对话历史管理器：记录多轮对话，并在长度超限时自动压缩（摘要）。
修复了 get_context 与 _compress_if_needed 的循环递归问题。
"""

import logging
from typing import List, Dict, Optional, Callable
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

logger = logging.getLogger(__name__)


class DialogueHistory:
    """单轮对话记录"""

    __slots__ = ("user", "assistant")

    def __init__(self, user: str, assistant: str):
        self.user = user
        self.assistant = assistant

    def to_dict(self) -> Dict[str, str]:
        return {"user": self.user, "assistant": self.assistant}

    def __repr__(self):
        return f"DialogueHistory(user={self.user[:30]}..., assistant={self.assistant[:30]}...)"


class HistoryManager:
    """
    对话历史管理器：
    - 记录每一轮用户与助手的交流
    - 提供最近 N 轮对话文本
    - 当历史超过最大轮数或 token 数时，自动调用摘要函数进行压缩
    """

    def __init__(
        self,
        max_turns: int = 5,
        max_tokens_approx: int = 2000,
        summarizer: Optional[Callable[[str], str]] = None
    ):
        self.history: List[DialogueHistory] = []
        self.max_turns = max_turns
        self.max_tokens_approx = max_tokens_approx
        self.summarizer = summarizer
        self.summary_text: Optional[str] = None

    def add(self, user_query: str, assistant_response: str):
        self.history.append(DialogueHistory(user_query, assistant_response))
        logger.debug(f"历史增加一轮，当前总轮数: {len(self.history)}")

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字符/ token）"""
        return len(text) // 1.5

    def _compress_if_needed(self):
        """检查是否需要压缩，并直接计算历史文本，避免调用 get_context 导致递归"""
        if not self.history or not self.summarizer:
            return

        # 直接拼接全部历史文本来估算 token（不经过 get_context）
        all_parts = []
        for turn in self.history:
            all_parts.append(f"用户：{turn.user}\n助手：{turn.assistant}")
        full_text = "\n".join(all_parts)
        current_tokens = self._estimate_tokens(full_text)

        if current_tokens > self.max_tokens_approx and len(self.history) > self.max_turns:
            # 需要压缩的早期轮数
            compress_count = len(self.history) - self.max_turns
            if compress_count <= 0:
                return

            early_turns = self.history[:compress_count]
            early_parts = []
            for turn in early_turns:
                early_parts.append(f"用户：{turn.user}\n助手：{turn.assistant}")
            early_text = "\n".join(early_parts)

            try:
                new_summary = self.summarizer(early_text)
                logger.info(f"生成历史摘要，压缩 {compress_count} 轮对话")
            except Exception as e:
                logger.error(f"摘要生成失败: {e}")
                return

            self.summary_text = new_summary
            self.history = self.history[compress_count:]  # 保留最近的 max_turns 轮

    def get_context(self, max_turns: Optional[int] = None) -> str:
        """获取上下文文本（含摘要和最近 max_turns 轮对话）"""
        if max_turns is None:
            max_turns = self.max_turns

        # 自动压缩检查
        self._compress_if_needed()

        recent = self.history[-max_turns:] if max_turns > 0 else []
        parts = []

        if self.summary_text:
            parts.append(f"历史对话摘要：{self.summary_text}\n")

        for turn in recent:
            parts.append(f"用户：{turn.user}\n助手：{turn.assistant}")

        return "\n".join(parts)

    def get_history_list(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        if max_turns is None:
            max_turns = self.max_turns
        self._compress_if_needed()
        return [t.to_dict() for t in self.history[-max_turns:]]

    def clear(self):
        self.history.clear()
        self.summary_text = None


# ------------------ 默认摘要函数（需注入 generator）------------------
def create_summary_function(generator, prompt_template: str = None) -> Callable[[str], str]:
    if prompt_template is None:
        prompt_template = (
            "请用简洁的语言总结以下对话历史中的关键信息和结论，不要包含无关细节。\n\n"
            "对话历史：\n{history}\n\n"
            "摘要："
        )

    def summarize(history_text: str) -> str:
        prompt = prompt_template.format(history=history_text)
        return generator.generate(prompt, max_new_tokens=256, temperature=0.3)

    return summarize