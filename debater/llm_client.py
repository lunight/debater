"""统一的 LLM 客户端

支持 Anthropic API 兼容方式接入多个模型，支持同步和流式调用。
"""

import time
import traceback
from typing import Dict, Optional, Any, List, Generator, Tuple
from dataclasses import dataclass

import httpx
from anthropic import Anthropic

from .config import get_config
from .logger import llm_logger, log_prompt, log_response


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model_id: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0


class LLMClient:
    """LLM 客户端
    
    管理多个模型的连接和调用，每个模型可以有不同的 base_url 和 API key。
    支持同步调用和流式调用。
    """
    
    def __init__(self):
        self._clients: Dict[str, Anthropic] = {}
        self._config = get_config()
        self._init_clients()
    
    def _init_clients(self):
        """初始化所有模型的客户端"""
        for model_id, model_cfg in self._config.models.items():
            api_key = model_cfg.api_key
            if not api_key:
                raise ValueError(
                    f"模型 '{model_id}' 未配置 API Key。\n"
                    f"请在 config.yaml 中设置 'api_key'，"
                    f"或设置环境变量 {model_cfg.api_key_env or '(未指定)'}"
                )
            
            client_kwargs = {"api_key": api_key}
            if model_cfg.base_url:
                client_kwargs["base_url"] = model_cfg.base_url
            
            # 流式安全 timeout：连接10s，读取300s（配合 call_stream 的 300s token 间隔超时）
            # DeepSeek reasoning 可能持续数分钟不输出 token，需要足够长的 read timeout
            timeout = httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=10.0,
                pool=5.0,
            )
            client_kwargs["http_client"] = httpx.Client(timeout=timeout)
            
            self._clients[model_id] = Anthropic(**client_kwargs)
    
    def call(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """同步调用指定模型"""
        if model_id not in self._clients:
            raise ValueError(f"未知的模型 ID: {model_id}。可用: {list(self._clients.keys())}")
        
        model_cfg = self._config.models[model_id]
        client = self._clients[model_id]
        
        temp = temperature if temperature is not None else model_cfg.temperature
        max_tok = max_tokens if max_tokens is not None else model_cfg.max_tokens
        
        start_time = time.time()
        
        try:
            log_prompt(
                llm_logger,
                f"SYNC CALL | model={model_id}",
                system_prompt,
                user_prompt,
            )
            
            kwargs = {
                "model": model_cfg.model_id,
                "max_tokens": max_tok,
                "temperature": temp,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if model_cfg.extra_body:
                kwargs["extra_body"] = model_cfg.extra_body
            
            response = client.messages.create(**kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            
            usage = None
            if hasattr(response, "usage"):
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            
            log_response(
                llm_logger,
                f"SYNC RESPONSE | model={model_id} | latency={latency_ms:.0f}ms",
                content,
            )
            
            return LLMResponse(
                content=content,
                model_id=model_id,
                usage=usage,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            llm_logger.error(
                f"[call] model={model_id} SYNC CALL FAILED.\n"
                f"Error: {e}\n{traceback.format_exc()}"
            )
            raise RuntimeError(f"调用模型 '{model_id}' 失败: {e}") from e
    
    def call_stream(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Tuple[str, str], None, LLMResponse]:
        """流式调用指定模型
        
        关键改进：
        1. 使用 daemon 线程 + Queue 实现 token 间隔超时检测
        2. 流式失败时自动重试 1 次，仍失败则 fallback 到同步调用
           （避免 DeepSeek 等模型 SSE 中途断连导致失败）
        3. 所有失败都记录详细日志（stack trace + 上下文）
        
        Yields:
            (token, accumulated_text): 每个 token 和累积的完整文本
        
        Returns:
            LLMResponse: 最终完整响应（通过 StopIteration 的 value）
        """
        if model_id not in self._clients:
            raise ValueError(f"未知的模型 ID: {model_id}")
        
        model_cfg = self._config.models[model_id]
        temp = temperature if temperature is not None else model_cfg.temperature
        max_tok = max_tokens if max_tokens is not None else model_cfg.max_tokens
        
        MAX_STREAM_RETRIES = 2  # 总共最多尝试 2 次流式
        last_error = None
        
        for attempt in range(1, MAX_STREAM_RETRIES + 1):
            try:
                llm_logger.info(
                    f"[call_stream] model={model_id} attempt={attempt}/{MAX_STREAM_RETRIES} "
                    f"temp={temp} max_tokens={max_tok} prompt_len={len(user_prompt)}"
                )
                result = yield from self._call_stream_impl(
                    model_id, system_prompt, user_prompt, temp, max_tok, attempt
                )
                llm_logger.info(f"[call_stream] model={model_id} attempt={attempt} SUCCESS")
                return result
            except RuntimeError as e:
                last_error = e
                error_msg = str(e).lower()
                is_conn_error = any(
                    kw in error_msg
                    for kw in [
                        "peer closed connection",
                        "connection reset",
                        "broken pipe",
                        "remote protocol error",
                        "connection aborted",
                        "errno 54",
                        "errno 32",
                    ]
                )
                if is_conn_error and attempt < MAX_STREAM_RETRIES:
                    llm_logger.warning(
                        f"[call_stream] model={model_id} attempt={attempt} FAILED (connection error). "
                        f"Retrying...\nError: {e}\n{traceback.format_exc()}"
                    )
                    continue  # 重试
                # 非连接错误，或已是最后一次流式尝试 → 跳出，fallback 到同步
                break
        
        # 流式全部失败，fallback 到同步调用
        llm_logger.error(
            f"[call_stream] model={model_id} 流式调用 {MAX_STREAM_RETRIES} 次均失败，"
            f"fallback 到同步调用。\nLast error: {last_error}\n{traceback.format_exc()}"
        )
        try:
            resp = self.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
                max_tokens=max_tok,
            )
            llm_logger.info(
                f"[call_stream] model={model_id} 同步 fallback SUCCESS "
                f"content_len={len(resp.content)} latency_ms={resp.latency_ms:.0f}"
            )
            # 模拟流式输出（一次性 yield 完整文本，UI 会更新为最终结果）
            yield resp.content, resp.content
            return LLMResponse(
                content=resp.content,
                model_id=model_id,
                usage=resp.usage,
                latency_ms=resp.latency_ms,
            )
        except Exception as sync_e:
            llm_logger.critical(
                f"[call_stream] model={model_id} 同步 fallback 也失败！\n"
                f"Error: {sync_e}\n{traceback.format_exc()}"
            )
            raise RuntimeError(
                f"模型 '{model_id}' 流式调用失败（已重试），同步 fallback 也失败: {sync_e}"
            ) from sync_e
    
    def _call_stream_impl(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temp: float,
        max_tok: int,
        attempt: int = 1,
    ) -> Generator[Tuple[str, str], None, LLMResponse]:
        """内部流式实现（可被 call_stream 包裹重试）"""
        model_cfg = self._config.models[model_id]
        client = self._clients[model_id]
        
        start_time = time.time()
        full_text = ""
        token_count = 0
        
        log_prompt(
            llm_logger,
            f"STREAM CALL | model={model_id} | attempt={attempt}",
            system_prompt,
            user_prompt,
        )
        
        kwargs = {
            "model": model_cfg.model_id,
            "max_tokens": max_tok,
            "temperature": temp,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if model_cfg.extra_body:
            kwargs["extra_body"] = model_cfg.extra_body
        
        import queue as q_module
        import threading
        
        result_queue = q_module.Queue(maxsize=0)  # 无界队列，防止生产者死锁
        
        def _producer():
            """在独立线程中消费 SSE 流，将 token 放入队列"""
            try:
                with client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        token = ""
                        if hasattr(event, "delta") and event.delta:
                            delta = event.delta
                            if hasattr(delta, "text") and delta.text:
                                token = delta.text
                            elif hasattr(delta, "thinking") and delta.thinking:
                                token = delta.thinking
                            elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
                                token = delta.reasoning_content
                            # 处理 Anthropic tool_use content block（DeepSeek 等模型可能返回此格式）
                            elif hasattr(delta, "partial_json") and delta.partial_json:
                                token = delta.partial_json
                        
                        # 同时处理 content_block_start 中的 tool_use 信息
                        if not token and hasattr(event, "content_block") and event.content_block:
                            block = event.content_block
                            if hasattr(block, "type") and block.type == "tool_use":
                                if hasattr(block, "input") and block.input:
                                    token = f'<tool_call name="{getattr(block, "name", "unknown")}">\n{block.input}\n</tool_call>'
                                elif hasattr(block, "name"):
                                    token = f'<tool_call name="{block.name}">\n</tool_call>'
                        
                        if token:
                            result_queue.put(("token", token))
                    
                    # 流正常结束
                    result_queue.put(("done", None))
            except Exception as e:
                result_queue.put(("error", str(e), traceback.format_exc()))
        
        # 启动生产者线程（daemon，避免 shutdown 阻塞主线程）
        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()
        
        # 消费者：带 300 秒 token 间隔超时
        # DeepSeek/Kimi 等模型在复杂 reasoning 时可能数分钟不输出 token
        TOKEN_TIMEOUT = 300
        while True:
            try:
                item = result_queue.get(timeout=TOKEN_TIMEOUT)
                # 支持 2 元组 ("token"/"done", payload) 和 3 元组 ("error", msg, traceback)
                if len(item) == 2:
                    event_type, payload = item
                elif len(item) >= 3:
                    event_type = item[0]
                    payload = item[1:]
                else:
                    event_type = item[0]
                    payload = None
            except q_module.Empty:
                llm_logger.error(
                    f"[_call_stream_impl] model={model_id} attempt={attempt} "
                    f"TOKEN TIMEOUT ({TOKEN_TIMEOUT}s) 无新token，"
                    f"当前已输出 {token_count} tokens，content_len={len(full_text)}"
                )
                raise RuntimeError(
                    f"流式调用模型 '{model_id}' 卡住：{TOKEN_TIMEOUT}秒内无新token，"
                    f"可能是模型陷入reasoning loop或网络连接异常"
                )
            
            if event_type == "token":
                full_text += payload
                token_count += 1
                yield payload, full_text
            elif event_type == "done":
                break
            elif event_type == "error":
                # payload 可能是 str 或 tuple(str, str, traceback_str)
                if isinstance(payload, tuple) and len(payload) >= 3:
                    error_msg, tb_str = payload[0], payload[2]
                else:
                    error_msg = str(payload)
                    tb_str = "(no traceback)"
                llm_logger.error(
                    f"[_call_stream_impl] model={model_id} attempt={attempt} "
                    f"PRODUCER ERROR after {token_count} tokens.\n"
                    f"Error: {error_msg}\n{tb_str}"
                )
                raise RuntimeError(f"流式调用模型 '{model_id}' 失败: {error_msg}")
        
        latency_ms = (time.time() - start_time) * 1000
        llm_logger.info(
            f"[_call_stream_impl] model={model_id} attempt={attempt} STREAM DONE "
            f"tokens={token_count} content_len={len(full_text)} latency_ms={latency_ms:.0f}"
        )
        
        log_response(
            llm_logger,
            f"STREAM RESPONSE | model={model_id} | attempt={attempt} | tokens={token_count} | latency={latency_ms:.0f}ms",
            full_text,
        )
        
        return LLMResponse(
            content=full_text,
            model_id=model_id,
            usage=None,
            latency_ms=latency_ms,
        )
    
    def call_parallel(
        self,
        calls: List[Dict[str, Any]],
    ) -> Dict[str, LLMResponse]:
        """并行调用多个模型（同步）
        
        支持同一个 model_id 被多次调用（不同 call_id），
        返回结果以 call_id 为 key（若未指定则回退到 model_id）。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        def _call_single(call_cfg: Dict[str, Any]) -> tuple:
            mid = call_cfg["model_id"]
            call_id = call_cfg.get("call_id", mid)
            resp = self.call(
                model_id=mid,
                system_prompt=call_cfg.get("system_prompt", ""),
                user_prompt=call_cfg["user_prompt"],
                temperature=call_cfg.get("temperature"),
                max_tokens=call_cfg.get("max_tokens"),
            )
            return call_id, resp
        
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            futures = {executor.submit(_call_single, c): c.get("call_id", c["model_id"]) for c in calls}
            for future in as_completed(futures):
                call_id, resp = future.result()
                results[call_id] = resp
        
        return results
    
    def reload(self):
        """重新加载配置并刷新所有客户端连接"""
        self._config = get_config()
        self._clients.clear()
        self._init_clients()
    
