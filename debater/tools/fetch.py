"""Web Fetch Tool"""

import httpx
import traceback
from .base import Tool, ToolResult
from ..logger import tool_logger


class WebFetchTool(Tool):
    """获取网页内容"""
    
    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="获取指定 URL 的网页内容，用于深度阅读网页",
            parameters={
                "url": "网页URL",
            },
        )
    
    def execute(self, url: str) -> ToolResult:
        try:
            tool_logger.info(f"[WebFetchTool] 开始获取: url={url}")
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; DebaterBot/1.0)"
                })
                resp.raise_for_status()
                
                # 尝试提取正文（简单实现）
                text = resp.text
                # 移除 script/style 标签
                import re
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                # 提取 body 内容
                body_match = re.search(r'<body[^>]*>(.*?)</body>', text, re.DOTALL)
                if body_match:
                    text = body_match.group(1)
                # 移除 HTML 标签
                text = re.sub(r'<[^>]+>', ' ', text)
                # 清理空白
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 截取前 5000 字符
                if len(text) > 5000:
                    text = text[:5000] + "\n\n...（已截断）"
                
                tool_logger.info(f"[WebFetchTool] 获取成功: url={url} content_len={len(text)}")
                return ToolResult(
                    content=f"URL: {url}\n\n{text}",
                    sources=[url],
                )
                
        except Exception as e:
            tool_logger.error(
                f"[WebFetchTool] 获取失败: url={url}\n"
                f"Error: {e}\n{traceback.format_exc()}"
            )
            return ToolResult(content="", sources=[], error=str(e))
