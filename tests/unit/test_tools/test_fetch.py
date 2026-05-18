"""测试 tools/fetch.py — WebFetchTool"""

import pytest
from unittest.mock import patch, MagicMock

from debater.tools.fetch import WebFetchTool
from debater.tools.base import ToolResult


def _mock_httpx_response(html, status_code=200):
    """创建 mock httpx.Client 上下文管理器"""
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        from httpx import HTTPStatusError
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return mock_client
        def __exit__(self, *args):
            pass

    return MockClient, mock_client


class TestWebFetchTool:
    def test_fetch_success(self):
        """成功获取网页内容"""
        tool = WebFetchTool()
        html = "<html><body><p>Hello World</p></body></html>"

        MockClient, mock_client = _mock_httpx_response(html)
        with patch("httpx.Client", MockClient):
            result = tool.execute(url="https://example.com")

        assert result.error is None
        assert "Hello World" in result.content
        assert "<html>" not in result.content  # tags stripped
        assert result.sources == ["https://example.com"]

    def test_fetch_removes_scripts_and_styles(self):
        """应移除 script 和 style 标签"""
        tool = WebFetchTool()
        html = "<html><script>alert(1)</script><style>.x{}</style><body>content</body></html>"

        MockClient, _ = _mock_httpx_response(html)
        with patch("httpx.Client", MockClient):
            result = tool.execute(url="https://example.com")

        assert "alert(1)" not in result.content
        assert ".x{}" not in result.content
        assert "content" in result.content

    def test_fetch_http_error(self):
        """HTTP 错误应返回 error"""
        tool = WebFetchTool()

        MockClient, _ = _mock_httpx_response("Not Found", status_code=404)
        with patch("httpx.Client", MockClient):
            result = tool.execute(url="https://example.com")

        assert result.error is not None

    def test_fetch_network_error(self):
        """网络异常应被捕获"""
        tool = WebFetchTool()

        class BrokenClient:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                raise Exception("connection timeout")
            def __exit__(self, *args):
                pass

        with patch("httpx.Client", BrokenClient):
            result = tool.execute(url="https://example.com")

        assert result.error is not None
        assert "connection timeout" in result.error

    def test_fetch_truncates_long_content(self):
        """超长内容应被截断"""
        tool = WebFetchTool()
        html = f"<html><body>{'x' * 200_000}</body></html>"

        MockClient, _ = _mock_httpx_response(html)
        with patch("httpx.Client", MockClient):
            result = tool.execute(url="https://example.com")

        assert result.error is None
        assert len(result.content) <= 110_000  # 100k + truncation message
