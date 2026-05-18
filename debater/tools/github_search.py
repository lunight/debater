"""GitHub Search Tool

搜索 GitHub 上的公开仓库、代码、issue、PR 等。
无需 API key（使用 GitHub 公开搜索 API）。
"""

import httpx
import traceback
from .base import Tool, ToolResult
from ..logger import tool_logger


class GitHubSearchTool(Tool):
    """搜索 GitHub 公开内容"""
    
    def __init__(self):
        super().__init__(
            name="github_search",
            description="搜索 GitHub 上的开源仓库、代码、技术方案、issue 讨论等。"
                        "适合查找技术实现、开源工具、算法参考、最佳实践等。",
            parameters={
                "query": "搜索关键词（技术术语、库名称、问题描述等）",
            },
        )
    
    def execute(self, query: str) -> ToolResult:
        try:
            tool_logger.info(f"[GitHubSearchTool] 开始搜索: query={query!r}")
            with httpx.Client(timeout=15.0) as client:
                # GitHub 公开搜索 API
                resp = client.get(
                    "https://api.github.com/search/repositories",
                    params={
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 5,
                    },
                    headers={
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "DebaterBot/1.0",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                
                items = data.get("items", [])
                tool_logger.info(f"[GitHubSearchTool] 返回 {len(items)} 个仓库")
                if not items:
                    return ToolResult(content="未找到相关仓库。", sources=[])
                
                lines = [f"GitHub 搜索: {query}", f"总计: {data.get('total_count', 0)} 个仓库", ""]
                sources = []
                for i, repo in enumerate(items, 1):
                    name = repo.get("full_name", "unknown")
                    desc = repo.get("description", "无描述") or "无描述"
                    stars = repo.get("stargazers_count", 0)
                    lang = repo.get("language", "未知语言") or "未知语言"
                    url = repo.get("html_url", "")
                    sources.append(url)
                    lines.append(f"{i}. {name} ⭐{stars} ({lang})")
                    lines.append(f"   {desc[:150]}...")
                    lines.append(f"   链接: {url}")
                    lines.append("")
                
                return ToolResult(content="\n".join(lines), sources=sources)
                
        except httpx.HTTPStatusError as e:
            tool_logger.error(
                f"[GitHubSearchTool] HTTP 错误: query={query!r} status={e.response.status_code}\n"
                f"Response: {e.response.text[:500]}"
            )
            if e.response.status_code == 403:
                return ToolResult(content="", sources=[], error="GitHub API 速率限制，请稍后再试")
            return ToolResult(content="", sources=[], error=f"GitHub API 错误: {e}")
        except Exception as e:
            tool_logger.error(
                f"[GitHubSearchTool] 搜索失败: query={query!r}\n"
                f"Error: {e}\n{traceback.format_exc()}"
            )
            return ToolResult(content="", sources=[], error=str(e))
