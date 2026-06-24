"""
本地 LLM 生成器：基于 HuggingFace Transformers，支持 4bit 量化、流式输出
含详细 RAG 系统提示词
"""

import logging
from typing import Optional, List, Dict, Any, Union, Generator
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
    pipeline
)

logger = logging.getLogger(__name__)


class LocalGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_streamer: bool = False,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_streamer = use_streamer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit and self.device.startswith("cuda"):
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("启用 4bit 量化")
            except ImportError:
                logger.warning("bitsandbytes 未安装，将使用全精度加载")
                load_in_4bit = False

        torch_dtype = model_kwargs.pop("torch_dtype", torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"模型 {model_name} 加载完成，设备: {self.device}")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.2,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        if self.use_streamer:
            return self._stream_generate(
                prompt, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, stop_strings, **kwargs
            )
        else:
            return self._full_generate(
                prompt, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, stop_strings, **kwargs
            )

    def _full_generate(self, prompt, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, stop_strings, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop_strings:
            for stop in stop_strings:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
                    break
        return text.strip()

    def _stream_generate(self, prompt, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, stop_strings, **kwargs):
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            **kwargs
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            if stop_strings:
                should_stop = False
                for stop in stop_strings:
                    if stop in (generated_text + new_text):
                        stop_pos = (generated_text + new_text).find(stop)
                        if stop_pos < len(generated_text):
                            yield ""
                            thread.join()
                            return
                        else:
                            yield new_text[:stop_pos - len(generated_text)]
                            thread.join()
                            return
            yield new_text
            generated_text += new_text
        thread.join()

    # ------------------------------------------------------------
    # 重点：详细完备的系统提示词 + RAG 提示构建
    # ------------------------------------------------------------
    DEFAULT_SYSTEM_PROMPT = (
        "你是一个严谨、可靠、负责任的知识助手，名为 RAG-Assistant。"
        "请严格遵循以下规则：\n"
        "1. **回答依据**：只能使用下方提供的“参考信息”来回答问题。如果参考信息中没有相关答案，请如实告知“根据现有资料无法回答”，不要尝试猜测或编造。\n"
        "2. **信息来源**：如果使用了参考信息中的具体事实，请尽量提及引用的片段编号（例如“依据参考片段2”），但不要生硬地罗列。\n"
        "3. **回答风格**：回答应简洁、准确、有条理，使用通俗易懂的中文。对于复杂问题，可以先给出总结再展开细节。\n"
        "4. **拒绝延伸**：只回答当前问题，不要主动添加其他建议、不要进行额外的延伸提问，也不要模拟新的对话。\n"
        "5. **语气**：保持友好、专业，不使用表情符号或过度热情的表达。\n"
        "6. **安全**：如果问题涉及违法、危险或有害内容，请礼貌拒绝并说明原因。\n"
        "现在，请根据以上规则回答用户问题。"
    )

    def format_rag_prompt(
        self,
        query: str,
        contexts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_history_turns: int = 3
    ) -> str:
        """
        构建 RAG 提示词。
        :param query: 用户当前问题
        :param contexts: 检索到的参考文本列表
        :param history: 对话历史，格式为 [{"user": "...", "assistant": "..."}, ...]
        :param system: 系统提示词，若未提供则使用内置 DEFAULT_SYSTEM_PROMPT
        :param max_history_turns: 保留最近几轮对话
        """
        if system is None:
            system = self.DEFAULT_SYSTEM_PROMPT

        # 拼接参考信息（带编号）
        context_str = "\n\n".join(
            f"[参考片段 {i+1}]\n{c}" for i, c in enumerate(contexts)
        )

        # 构建对话历史部分（如果存在）
        history_str = ""
        if history:
            recent = history[-max_history_turns:]
            parts = []
            for turn in recent:
                parts.append(f"用户：{turn.get('user', '')}\n助手：{turn.get('assistant', '')}")
            if parts:
                history_str = "对话历史：\n" + "\n".join(parts) + "\n"

        # 最终拼接
        prompt = f"""{system}

{history_str}
参考信息：
{context_str}

当前问题：{query}
请直接给出最终答案，不要继续生成对话：
助手："""
        return prompt

    def generate_with_rag(
        self,
        query: str,
        contexts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        prompt = self.format_rag_prompt(query, contexts, history, system)
        return self.generate(prompt, **kwargs)