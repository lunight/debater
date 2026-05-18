"""测试 tools/registry.py — ToolRegistry"""

import pytest

from debater.tools.registry import ToolRegistry
from debater.tools.base import Tool, ToolResult


class DummyTool(Tool):
    def __init__(self, name):
        super().__init__(name=name, description="dummy", parameters={})
    def execute(self, **kwargs):
        return ToolResult(content="dummy")


class TestToolRegistry:
    def test_register_and_get(self):
        """应能注册和获取工具"""
        reg = ToolRegistry()
        tool = DummyTool("t1")
        reg.register(tool)
        assert reg.get("t1") is tool

    def test_get_nonexistent_raises(self):
        """获取不存在的工具应抛出 KeyError"""
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_all(self):
        """list_all 应返回所有注册的工具"""
        reg = ToolRegistry()
        reg.register(DummyTool("a"))
        reg.register(DummyTool("b"))
        names = [t.name for t in reg.list_all()]
        assert sorted(names) == ["a", "b"]

    def test_register_default_tools(self):
        """register_default_tools 应注册所有默认工具"""
        reg = ToolRegistry()
        reg.register_default_tools()
        names = {t.name for t in reg.list_all()}
        assert "web_search" in names
        assert "web_fetch" in names
        assert "file_read" in names
        assert "file_glob" in names
        assert "code_search" in names

    def test_create_executor(self):
        """create_executor 应创建包含所有工具的 ToolExecutor"""
        reg = ToolRegistry()
        reg.register(DummyTool("a"))
        executor = reg.create_executor()
        # 工具 a 能执行
        results = executor.extract_and_execute('<tool_call name="a"></tool_call>')
        assert len(results) == 1
        assert results[0]["result"].content == "dummy"
        # 工具 b 不存在，返回错误
        results = executor.extract_and_execute('<tool_call name="b"></tool_call>')
        assert len(results) == 1
        assert "未知工具" in results[0]["result"].error

    def test_get_prompt_context_not_empty(self):
        """get_prompt_context 不应为空字符串"""
        reg = ToolRegistry()
        reg.register_default_tools()
        ctx = reg.get_prompt_context()
        assert len(ctx) > 0
        assert "web_search" in ctx
        assert "<tool_call" in ctx

    def test_get_prompt_context_empty_when_no_tools(self):
        """无工具时 prompt context 应为空"""
        reg = ToolRegistry()
        assert reg.get_prompt_context() == ""
