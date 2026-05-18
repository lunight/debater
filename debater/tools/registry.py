"""Tool 注册表"""

from typing import Dict, List, Optional
from .base import Tool, ToolExecutor
from .search import TavilySearchTool
from .ddg_search import DDGSearchTool
from .fetch import WebFetchTool
from .github_search import GitHubSearchTool
from .filesystem import FileReadTool, FileGlobTool, CodeSearchTool


class ToolRegistry:
    """Tool 注册表"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self._tools: Dict[str, Tool] = {}
        self.base_dir = base_dir
    
    def register_default_tools(self):
        """注册默认 tools"""
        # Tavily 暂时超量不可用，默认改用 DuckDuckGo（免费、无需 API key）
        # 恢复 Tavily 时改回: self.register(TavilySearchTool())
        self.register(DDGSearchTool())
        self.register(WebFetchTool())
        self.register(GitHubSearchTool())
        # 文件系统工具 — 移植自 Claude Code Read/Glob/Grep
        self.register(FileReadTool(base_dir=self.base_dir))
        self.register(FileGlobTool(base_dir=self.base_dir))
        self.register(CodeSearchTool(base_dir=self.base_dir))
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        return self._tools[name]
    
    def list_all(self) -> List[Tool]:
        return list(self._tools.values())
    
    def create_executor(self) -> ToolExecutor:
        """创建 ToolExecutor"""
        return ToolExecutor(self.list_all())
    
    def get_prompt_context(self) -> str:
        """获取用于 prompt 注入的 tools 说明"""
        executor = self.create_executor()
        return executor.to_prompt_context()
