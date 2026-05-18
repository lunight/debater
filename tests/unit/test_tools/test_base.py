"""测试 tools/base.py — Tool, ToolResult, ToolExecutor"""

import pytest

from debater.tools.base import Tool, ToolResult, ToolExecutor


class FakeTool(Tool):
    """测试用的假 Tool"""
    def __init__(self, name="fake", result_text="result"):
        super().__init__(
            name=name,
            description="A fake tool for testing",
            parameters={"query": "search query", "limit": "max results"},
        )
        self.result_text = result_text
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(content=self.result_text, sources=["source1"])


class TestToolResult:
    def test_default_sources_empty_list(self):
        """ToolResult 默认 sources 应为空列表"""
        result = ToolResult(content="hello")
        assert result.sources == []
        assert result.error is None

    def test_with_sources(self):
        """ToolResult 应正确保存 sources"""
        result = ToolResult(content="hello", sources=["a", "b"])
        assert result.sources == ["a", "b"]

    def test_with_error(self):
        """ToolResult 应正确保存 error"""
        result = ToolResult(content="", error="something wrong")
        assert result.error == "something wrong"


class TestTool:
    def test_to_xml_format(self):
        """Tool.to_xml() 应生成正确的 XML 格式"""
        tool = FakeTool()
        xml = tool.to_xml()
        assert '<tool name="fake">' in xml
        assert "<description>A fake tool for testing</description>" in xml
        assert "query: search query" in xml
        assert "limit: max results" in xml

    def test_execute_returns_tool_result(self):
        """Tool.execute() 应返回 ToolResult"""
        tool = FakeTool()
        result = tool.execute(query="test")
        assert isinstance(result, ToolResult)
        assert result.content == "result"
        assert result.sources == ["source1"]


class TestToolExecutor:
    def test_empty_tools_prompt_context(self):
        """无工具时 prompt context 应为空"""
        executor = ToolExecutor([])
        assert executor.to_prompt_context() == ""

    def test_has_tool_calls_detects_xml(self):
        """应能检测到 <tool_call> XML 标签"""
        tool = FakeTool()
        executor = ToolExecutor([tool])

        text_with_call = '<tool_call name="fake"><query>hello</query></tool_call>'
        assert executor.has_tool_calls(text_with_call) is True

        text_without_call = "just some normal text"
        assert executor.has_tool_calls(text_without_call) is False

    def test_extract_and_execute_single_call(self):
        """应能提取并执行单个 tool call"""
        tool = FakeTool()
        executor = ToolExecutor([tool])

        text = '<tool_call name="fake"><query>hello world</query><limit>5</limit></tool_call>'
        results = executor.extract_and_execute(text)

        assert len(results) == 1
        assert results[0]["name"] == "fake"
        assert results[0]["params"]["query"] == "hello world"
        assert results[0]["params"]["limit"] == "5"
        assert results[0]["result"].content == "result"
        assert tool.calls == [{"query": "hello world", "limit": "5"}]

    def test_extract_and_execute_multiple_calls(self):
        """应能提取并执行多个 tool call"""
        tool = FakeTool()
        executor = ToolExecutor([tool])

        text = (
            '<tool_call name="fake"><query>first</query></tool_call>'
            'some text between'
            '<tool_call name="fake"><query>second</query></tool_call>'
        )
        results = executor.extract_and_execute(text)

        assert len(results) == 2
        assert results[0]["params"]["query"] == "first"
        assert results[1]["params"]["query"] == "second"

    def test_extract_unknown_tool_returns_error(self):
        """未知工具应返回错误结果"""
        tool = FakeTool()
        executor = ToolExecutor([tool])

        text = '<tool_call name="unknown"><x>1</x></tool_call>'
        results = executor.extract_and_execute(text)

        assert len(results) == 1
        assert results[0]["result"].error == "未知工具: unknown"

    def test_extract_and_execute_exception_handling(self):
        """工具执行异常应被捕获并转为 error"""
        class BrokenTool(Tool):
            def __init__(self):
                super().__init__(name="broken", description="x", parameters={})
            def execute(self, **kwargs):
                raise ValueError("boom")

        executor = ToolExecutor([BrokenTool()])
        results = executor.extract_and_execute('<tool_call name="broken"></tool_call>')

        assert len(results) == 1
        assert "boom" in results[0]["result"].error

    def test_build_tool_result_message_success(self):
        """build_tool_result_message 应正确格式化成功结果"""
        tool = FakeTool()
        executor = ToolExecutor([tool])
        results = [{"name": "fake", "params": {}, "result": ToolResult(content="ok")}]
        msg = executor.build_tool_result_message(results)
        assert '<tool_result name="fake">' in msg
        assert "ok" in msg

    def test_build_tool_result_message_error(self):
        """build_tool_result_message 应正确格式化错误结果"""
        tool = FakeTool()
        executor = ToolExecutor([tool])
        results = [{"name": "fake", "params": {}, "result": ToolResult(content="", error="fail")}]
        msg = executor.build_tool_result_message(results)
        assert "【工具执行失败】fail" in msg

    def test_to_prompt_context_contains_all_tools(self):
        """prompt context 应包含所有工具的 XML 描述"""
        t1 = FakeTool(name="t1")
        t2 = FakeTool(name="t2")
        executor = ToolExecutor([t1, t2])
        ctx = executor.to_prompt_context()
        assert '<tool name="t1">' in ctx
        assert '<tool name="t2">' in ctx
        assert "<tool_call" in ctx
