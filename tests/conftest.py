"""共享 fixtures"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """创建临时目录，测试结束后自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_tool():
    """返回一个可配置的 mock tool"""
    from debater.tools.base import Tool, ToolResult

    class MockTool(Tool):
        def __init__(self, name="mock", description="mock tool", result_text="ok"):
            super().__init__(name=name, description=description, parameters={"input": "str"})
            self.result_text = result_text
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return ToolResult(content=self.result_text)

    return MockTool
