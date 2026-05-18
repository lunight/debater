"""Tool 基类

Prompt-based Tool Calling 的基础设施。
模型输出 <tool_call> XML 标签，ToolExecutor 解析并执行。
"""

import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..logger import tool_logger


@dataclass
class ToolResult:
    """Tool 执行结果"""
    content: str
    sources: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class Tool:
    """Tool 定义"""
    name: str
    description: str
    parameters: Dict[str, str]  # {param_name: param_description}
    
    def execute(self, **kwargs) -> ToolResult:
        """执行 tool，子类必须重写"""
        raise NotImplementedError
    
    def to_xml(self) -> str:
        """转换为 prompt 中使用的 XML 描述"""
        params_xml = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.parameters.items()
        ])
        return f"""<tool name="{self.name}">
<description>{self.description}</description>
<parameters>
{params_xml}
</parameters>
</tool>"""


class ToolExecutor:
    """Tool 执行器
    
    解析模型输出中的 <tool_call> 标签，执行对应 tool，
    并将结果以 <tool_result> 标签注入下一条 user message。
    
    每个 tool 执行都有 30 秒整体超时保护。
    """
    
    EXECUTION_TIMEOUT = 30  # 单个 tool 执行超时（秒）
    
    # 同时兼容 <tool_call>（规范格式）和 <tool>（部分模型的替代格式）
    TOOL_CALL_PATTERN = re.compile(
        r'<tool(?:_call)?\s+name="([^"]+)">\s*(.*?)\s*</tool(?:_call)?>',
        re.DOTALL
    )
    PARAM_PATTERN = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
    
    def __init__(self, tools: List[Tool]):
        self._tools = {t.name: t for t in tools}
    
    def has_tool_calls(self, text: str) -> bool:
        """检查文本中是否包含 tool call"""
        has = bool(self.TOOL_CALL_PATTERN.search(text))
        text_preview = text[:200].replace("\n", " ") if text else "(empty)"
        tool_logger.debug(
            f"[has_tool_calls] text_preview={text_preview!r} | result={has}"
        )
        return has
    
    def _parse_params(self, inner: str, tool_name: str) -> Dict[str, str]:
        """解析 tool_call 内部参数，兼容多种模型输出格式
        
        支持格式：
        1. 标准 XML 格式：<query>关键词</query>
        2. DeepSeek 格式：<param_name>query</param_name>关键词
        3. JSON 格式：{"query": "关键词", "region": "cn-zh"}
        """
        params = {}
        inner_preview = inner[:200].replace("\n", " ") if inner else "(empty)"
        
        # 先尝试标准 XML 格式：标签名即参数名
        for pmatch in self.PARAM_PATTERN.finditer(inner):
            key = pmatch.group(1)
            val = pmatch.group(2).strip()
            # 跳过 param_name 元标签（见下方兼容处理）
            if key == "param_name":
                continue
            params[key] = val
        
        # 兼容 DeepSeek 的 <param_name>key</param_name>value 格式
        if not params:
            param_name_match = re.search(r'<param_name>(\w+)</param_name>\s*(.*?)(?:\n|<|$)', inner, re.DOTALL)
            if param_name_match and tool_name in self._tools:
                actual_key = param_name_match.group(1)
                actual_val = param_name_match.group(2).strip()
                if actual_key and actual_val:
                    params[actual_key] = actual_val
                    tool_logger.info(
                        f"[_parse_params] tool={tool_name} 使用 DeepSeek 兼容格式 | "
                        f"param={actual_key} | val_preview={actual_val[:100]!r}"
                    )
        
        # 兼容 JSON 格式（部分模型输出 <tool> 标签内嵌 JSON）
        if not params:
            try:
                # 尝试提取 JSON 对象
                json_match = re.search(r'\{.*\}', inner, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        params = {k: str(v) for k, v in parsed.items()}
                        tool_logger.info(
                            f"[_parse_params] tool={tool_name} 使用 JSON 格式 | "
                            f"params={params}"
                        )
            except Exception:
                pass  # JSON 解析失败，继续
        
        if not params:
            tool_logger.warning(
                f"[_parse_params] tool={tool_name} 未解析出参数 | inner_preview={inner_preview!r}"
            )
        else:
            tool_logger.info(
                f"[_parse_params] tool={tool_name} 解析完成 | params={params}"
            )
        return params
    
    def _execute_single(self, tool_name: str, params: Dict[str, str]) -> ToolResult:
        """执行单个 tool（同步调用，超时由调用方控制）"""
        if tool_name not in self._tools:
            err = f"未知工具: {tool_name}"
            tool_logger.error(f"[_execute_single] {err} | known_tools={list(self._tools.keys())}")
            return ToolResult(content="", error=err)
        try:
            return self._tools[tool_name].execute(**params)
        except Exception as e:
            tb = traceback.format_exc()
            tool_logger.error(
                f"[_execute_single] tool={tool_name} CRASH\n"
                f"Error: {e}\n{tb}"
            )
            return ToolResult(
                content="",
                error=f"工具 '{tool_name}' 执行失败: {type(e).__name__}: {e}"
            )
    
    def extract_and_execute(self, text: str) -> List[Dict]:
        """提取并执行所有 tool calls（并行执行，整体超时）
        
        多个 tool call 会并行执行，而不是顺序执行，避免 N×15s 的卡顿。
        整体超时 60 秒。
        
        Returns:
            每个 tool call 的执行结果，格式为 {"name", "params", "result", "duration"}
        """
        matches = list(self.TOOL_CALL_PATTERN.finditer(text))
        if not matches:
            tool_logger.info("[extract_and_execute] 未检测到 tool call，直接返回")
            return []
        
        MAX_TOOLS_PER_CALL = 10  # 防护：单轮最多执行 10 个 tool call，防止模型异常输出过多
        if len(matches) > MAX_TOOLS_PER_CALL:
            tool_logger.warning(
                f"[extract_and_execute] 检测到异常多的 tool call: {len(matches)} 个，"
                f"超过上限 {MAX_TOOLS_PER_CALL}，只执行前 {MAX_TOOLS_PER_CALL} 个"
            )
            matches = matches[:MAX_TOOLS_PER_CALL]
        
        tool_logger.info(f"[extract_and_execute] 开始执行 {len(matches)} 个 tool call")
        
        def _run_one(match):
            tool_name = match.group(1)
            inner = match.group(2)
            params = self._parse_params(inner, tool_name)
            params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            
            tool_logger.info(f"[tool_call] name={tool_name} | params=({params_str}) | 开始执行")
            start = time.time()
            try:
                result = self._execute_single(tool_name, params)
                duration = time.time() - start
                if result.error:
                    tool_logger.warning(
                        f"[tool_call] name={tool_name} | 状态=FAILED | 耗时={duration:.2f}s | "
                        f"error={result.error[:200]}"
                    )
                else:
                    content_preview = (result.content or "")[:100].replace("\n", " ")
                    tool_logger.info(
                        f"[tool_call] name={tool_name} | 状态=SUCCESS | 耗时={duration:.2f}s | "
                        f"content_len={len(result.content or '')} | content_preview={content_preview!r}"
                    )
                return {
                    "name": tool_name,
                    "params": params,
                    "result": result,
                    "duration": round(duration, 2),
                }
            except Exception as e:
                duration = time.time() - start
                tb = traceback.format_exc()
                tool_logger.error(
                    f"[tool_call] name={tool_name} | 状态=CRASH | 耗时={duration:.2f}s | "
                    f"exception={type(e).__name__}: {e}\n{tb}"
                )
                return {
                    "name": tool_name,
                    "params": params,
                    "result": ToolResult(
                        content="",
                        error=f"工具 '{tool_name}' 执行崩溃: {type(e).__name__}: {e}"
                    ),
                    "duration": round(duration, 2),
                }
        
        # 并行执行所有 tool call（避免顺序执行导致的 N×15s 卡顿）
        results = []
        pool = ThreadPoolExecutor()
        futures = {pool.submit(_run_one, m): m for m in matches}
        overall_start = time.time()
        try:
            for future in futures:
                try:
                    results.append(future.result(timeout=self.EXECUTION_TIMEOUT))
                except FutureTimeoutError:
                    match = futures[future]
                    tool_name = match.group(1)
                    tool_logger.error(
                        f"[tool_call] name={tool_name} | 状态=TIMEOUT | 耗时={self.EXECUTION_TIMEOUT}s | "
                        f"原因=ThreadPoolExecutor 整体超时"
                    )
                    results.append({
                        "name": tool_name,
                        "params": self._parse_params(match.group(2), tool_name),
                        "result": ToolResult(
                            content="",
                            error=f"工具 '{tool_name}' 执行超时（>{self.EXECUTION_TIMEOUT}秒）"
                        ),
                        "duration": self.EXECUTION_TIMEOUT,
                    })
                except Exception as e:
                    match = futures[future]
                    tool_name = match.group(1)
                    tb = traceback.format_exc()
                    tool_logger.error(
                        f"[tool_call] name={tool_name} | 状态=TIMEOUT_EXCEPTION | 耗时={self.EXECUTION_TIMEOUT}s | "
                        f"exception={type(e).__name__}: {e}\n{tb}"
                    )
                    results.append({
                        "name": tool_name,
                        "params": self._parse_params(match.group(2), tool_name),
                        "result": ToolResult(
                            content="",
                            error=f"工具 '{tool_name}' 执行失败: {type(e).__name__}: {e}"
                        ),
                        "duration": self.EXECUTION_TIMEOUT,
                    })
        finally:
            pool.shutdown(wait=False)
        
        overall_duration = time.time() - overall_start
        tool_logger.info(
            f"[extract_and_execute] 完成 | 总耗时={overall_duration:.2f}s | "
            f"成功={sum(1 for r in results if not r['result'].error)}/"
            f"失败={sum(1 for r in results if r['result'].error)}/"
            f"总数={len(results)}"
        )
        return results
    
    def build_tool_result_message(self, results: List[Dict]) -> str:
        """构建 tool result 消息，注入到 user prompt 中"""
        lines = []
        for r in results:
            result = r["result"]
            if result.error:
                content = f"【工具执行失败】{result.error}"
            else:
                content = result.content
            lines.append(f"""<tool_result name="{r['name']}">
{content}
</tool_result>""")
        
        msg = "\n\n".join(lines)
        tool_logger.info(
            f"[build_tool_result_message] 构建完成 | results={len(results)} | msg_len={len(msg)}"
        )
        return msg
    
    def to_prompt_context(self) -> str:
        """生成 prompt 中使用的 tools 说明"""
        if not self._tools:
            return ""
        
        tools_xml = "\n\n".join([t.to_xml() for t in self._tools.values()])
        return f"""
你有以下工具可用：

{tools_xml}

如需使用工具，请按以下格式输出（会被自动执行并返回结果）：
<tool_call name="tool_name">
<参数名>参数值</参数名>
</tool_call>

例如搜索工具：
<tool_call name="web_search">
<query>保利物业 2025 毛利率</query>
</tool_call>

⚠️ 重要：决定调用工具时，**直接输出上面的 XML 格式**，不要写"我准备用搜索工具...""让我查一下..."之类的自然语言描述。系统只能识别 XML 标签，无法解析自然语言意图。
""".strip()
