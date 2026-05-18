"""Tool 系统

支持 prompt-based tool calling，包括 WebSearch、WebFetch、FileRead、FileGlob、CodeSearch 等。
"""

from .base import Tool, ToolResult, ToolExecutor
from .search import TavilySearchTool
from .fetch import WebFetchTool
from .github_search import GitHubSearchTool
from .filesystem import FileReadTool, FileGlobTool, CodeSearchTool
from .registry import ToolRegistry

__all__ = [
    "Tool", "ToolResult", "ToolExecutor",
    "TavilySearchTool", "WebFetchTool", "GitHubSearchTool",
    "FileReadTool", "FileGlobTool", "CodeSearchTool",
    "ToolRegistry",
]
