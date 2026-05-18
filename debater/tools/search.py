"""Tavily Web Search Tool (DuckDuckGo 回退)

所有网络调用都有 15 秒超时保护，避免卡住。
"""

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from .base import Tool, ToolResult
from .ddg_search import DDGSearchTool
from ..logger import tool_logger


class TavilySearchTool(Tool):
    """使用 Tavily API 进行网络搜索，失败时自动回退到 DuckDuckGo"""
    
    TIMEOUT_SECONDS = 15
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络信息，获取最新数据、新闻、研究报告等。"
                        "优先使用 Tavily（质量更高），Tavily 不可用时自动回退到 DuckDuckGo。"
                        "中文查询会自动优化为国内结果，英文查询用国际结果。",
            parameters={
                "query": "搜索关键词（具体的数据点或事实，不要复制整个问题）",
            },
        )
        self._client = None
        self._ddg = DDGSearchTool()
    
    def _get_client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError:
                raise ImportError("tavily 未安装，请运行: pip install tavily")
            api_key = os.environ.get("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY 环境变量未设置")
            # 设置 15 秒超时，避免 requests.post 默认 60 秒导致整体卡住
            self._client = TavilyClient(api_key=api_key, timeout=self.TIMEOUT_SECONDS)
        return self._client
    
    def execute(self, query: str) -> ToolResult:
        # 先尝试 Tavily（客户端已设置 15 秒 timeout，直接用 try/except）
        tavily_error = None
        try:
            tool_logger.info(f"[TavilySearchTool] 开始搜索: query={query!r}")
            client = self._get_client()
            response = client.search(query=query, max_results=5, search_depth="advanced")
            
            results = response.get("results", [])
            tool_logger.info(f"[TavilySearchTool] 返回 {len(results)} 条结果")
            if results:
                lines = [f"搜索: {query} (来源: Tavily)", ""]
                sources = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "无标题")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    sources.append(url)
                    lines.append(f"{i}. {title}")
                    lines.append(f"   {content[:300]}...")
                    lines.append(f"   来源: {url}")
                    lines.append("")
                
                return ToolResult(content="\n".join(lines), sources=sources)
            else:
                tavily_error = "Tavily 返回空结果"
                tool_logger.warning(f"[TavilySearchTool] 空结果: query={query!r}")
        except Exception as e:
            tavily_error = f"Tavily 调用失败: {type(e).__name__}: {e}"
            tool_logger.error(
                f"[TavilySearchTool] 搜索失败: query={query!r}\n"
                f"Error: {e}\n{traceback.format_exc()}"
            )
        
        # Tavily 失败，回退到 DuckDuckGo（带超时）
        tool_logger.info(f"[TavilySearchTool] 回退到 DDG: query={query!r}")
        ddg_result = self._ddg.execute(query)
        if not ddg_result.error:
            content = f"[Tavily 不可用: {tavily_error or '无结果'}，已回退到 DuckDuckGo]\n\n" + ddg_result.content
            return ToolResult(content=content, sources=ddg_result.sources)
        
        # 两者都失败——返回详细的合并错误信息
        tool_logger.error(
            f"[TavilySearchTool] Tavily + DDG 均失败: query={query!r}\n"
            f"Tavily: {tavily_error}\nDDG: {ddg_result.error}"
        )
        return ToolResult(
            content="",
            sources=[],
            error=(
                f"【搜索工具链全部失败】\n"
                f"1. Tavily: {tavily_error or '无结果'}\n"
                f"2. DuckDuckGo 回退: {ddg_result.error}\n"
                f"建议：检查网络连接，或稍后再试。"
            )
        )
