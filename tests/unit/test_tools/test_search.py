"""测试 tools/search.py — TavilySearchTool"""

import pytest
from unittest.mock import patch, MagicMock

from debater.tools.search import TavilySearchTool
from debater.tools.base import ToolResult


class TestTavilySearchTool:
    def test_search_success(self):
        """成功搜索应返回格式化结果"""
        tool = TavilySearchTool()

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Test 1", "url": "https://a.com", "content": "Result A"},
                {"title": "Test 2", "url": "https://b.com", "content": "Result B"},
            ]
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = tool.execute(query="python testing")

        assert result.error is None
        assert "Test 1" in result.content
        assert "Result A" in result.content
        assert "https://a.com" in result.sources
        assert "https://b.com" in result.sources
        mock_client.search.assert_called_once_with(query="python testing", max_results=5, search_depth="advanced")

    def test_search_no_api_key_fallback_to_ddg(self):
        """无 API key 时应回退到 DuckDuckGo"""
        tool = TavilySearchTool()

        with patch.dict("os.environ", {}, clear=True):
            with patch("tavily.TavilyClient", side_effect=Exception("No API key")):
                with patch("debater.tools.search.DDGSearchTool.execute") as mock_ddg:
                    mock_ddg.return_value = ToolResult(
                        content="DDG fallback result",
                        sources=["https://example.com"],
                    )
                    result = tool.execute(query="test")

        assert result.error is None
        assert "DDG fallback result" in result.content
        assert "Tavily 不可用" in result.content

    def test_search_api_error_fallback_to_ddg(self):
        """API 调用失败时应回退到 DuckDuckGo"""
        tool = TavilySearchTool()

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Rate limited")

        with patch("tavily.TavilyClient", return_value=mock_client):
            with patch("debater.tools.search.DDGSearchTool.execute") as mock_ddg:
                mock_ddg.return_value = ToolResult(
                    content="DDG fallback result",
                    sources=["https://example.com"],
                )
                result = tool.execute(query="test")

        assert result.error is None
        assert "DDG fallback result" in result.content

    def test_search_both_fail(self):
        """Tavily 和 DuckDuckGo 都失败时应返回错误"""
        tool = TavilySearchTool()

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Rate limited")

        with patch("tavily.TavilyClient", return_value=mock_client):
            with patch("debater.tools.search.DDGSearchTool.execute") as mock_ddg:
                mock_ddg.return_value = ToolResult(
                    content="",
                    error="DDG also failed",
                )
                result = tool.execute(query="test")

        assert result.error is not None
        assert "Rate limited" in result.error
        assert "DDG also failed" in result.error

    def test_search_empty_results(self):
        """Tavily 空结果时应回退到 DuckDuckGo"""
        tool = TavilySearchTool()

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("tavily.TavilyClient", return_value=mock_client):
            with patch("debater.tools.search.DDGSearchTool.execute") as mock_ddg:
                mock_ddg.return_value = ToolResult(
                    content="DDG fallback: no results",
                    sources=[],
                )
                result = tool.execute(query="xyznonexistent")

        assert result.error is None
        assert "DDG fallback" in result.content
