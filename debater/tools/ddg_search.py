"""DuckDuckGo Search Tool (Tavily 回退方案)

无需 API key，免费使用，作为 Tavily 失败时的自动回退。
所有网络调用都有 15 秒超时保护。
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional
from .base import Tool, ToolResult
from ..logger import tool_logger


class DDGSearchTool(Tool):
    """使用 DuckDuckGo 搜索网络信息
    
    作为 Tavily 的免费回退方案，无需 API key。
    支持中文和英文搜索，自动根据查询语言选择区域。
    """
    
    TIMEOUT_SECONDS = 15
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络信息，获取最新数据、新闻、研究报告等。"
                        "支持中文和英文查询，中文搜索优先用国内结果，英文搜索用国际结果。",
            parameters={
                "query": "搜索关键词（具体的数据点或事实，不要复制整个问题）",
            },
        )
    
    def _detect_language(self, text: str) -> str:
        """检测查询语言"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        return "en"
    
    def execute(self, query: str) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(
                content="",
                error="ddgs 未安装，请运行: pip install ddgs"
            )
        
        try:
            lang = self._detect_language(query)
            region = "cn-zh" if lang == "zh" else "us-en"
            
            tool_logger.info(f"[DDGSearchTool] 开始搜索: query={query!r} region={region}")
            
            def _do_search():
                # 修复 Python 3.9 + macOS 下的 SSL 兼容性问题
                # duckduckgo-search >= 7.0 默认启用 TLS 1.3，但 macOS Python 3.9 不支持
                import ssl
                try:
                    ssl.OPENSSL_VERSION_INFO
                except AttributeError:
                    pass
                # 通过 verify=False 禁用 SSL 证书验证，绕过 TLS 版本问题
                with DDGS(verify=False) as ddgs:
                    return list(ddgs.text(
                        query,
                        region=region,
                        max_results=5,
                    ))
            
            # 使用 ThreadPoolExecutor 控制超时，但 shutdown(wait=False) 避免死锁
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(_do_search)
            try:
                results = future.result(timeout=self.TIMEOUT_SECONDS)
            except FutureTimeoutError:
                tool_logger.error(f"[DDGSearchTool] 搜索超时: query={query!r} timeout={self.TIMEOUT_SECONDS}s")
                return ToolResult(
                    content="",
                    sources=[],
                    error=f"DuckDuckGo 搜索超时（>{self.TIMEOUT_SECONDS}秒），可能是网络问题或 DuckDuckGo 服务不可用"
                )
            finally:
                pool.shutdown(wait=False)
            
            tool_logger.info(f"[DDGSearchTool] 返回 {len(results)} 条结果")
            if not results:
                return ToolResult(content="DuckDuckGo 未找到相关结果。", sources=[])
            
            lines = [f"搜索: {query} (来源: DuckDuckGo, 区域: {region})", ""]
            sources = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                body = r.get("body", "")
                href = r.get("href", "")
                sources.append(href)
                lines.append(f"{i}. {title}")
                lines.append(f"   {body[:300]}...")
                lines.append(f"   来源: {href}")
                lines.append("")
            
            return ToolResult(content="\n".join(lines), sources=sources)
            
        except Exception as e:
            tool_logger.error(
                f"[DDGSearchTool] 搜索失败: query={query!r}\n"
                f"Error: {e}\n{traceback.format_exc()}"
            )
            return ToolResult(
                content="",
                sources=[],
                error=f"DuckDuckGo 调用失败: {type(e).__name__}: {e}"
            )
