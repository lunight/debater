"""测试 tools/filesystem.py — FileReadTool, FileGlobTool, CodeSearchTool"""

import pytest

from debater.tools.filesystem import FileReadTool, FileGlobTool, CodeSearchTool
from debater.tools.base import ToolResult


class TestFileReadTool:
    def test_read_existing_file(self, temp_dir):
        """应能读取存在的文件"""
        f = temp_dir / "test.txt"
        f.write_text("line1\nline2\nline3\n")

        tool = FileReadTool(base_dir=str(temp_dir))
        result = tool.execute(path="test.txt")

        assert result.error is None
        assert "line1" in result.content
        assert "line2" in result.content
        assert "line3" in result.content
        assert "3 lines total" in result.content

    def test_read_with_offset_and_limit(self, temp_dir):
        """应支持 offset 和 limit 分页"""
        f = temp_dir / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")

        tool = FileReadTool(base_dir=str(temp_dir))
        result = tool.execute(path="test.txt", offset="2", limit="2")

        lines = result.content.splitlines()
        body_lines = [l for l in lines if l.startswith("   ") and "|" in l]
        assert len(body_lines) == 2
        assert "b" in body_lines[0]
        assert "c" in body_lines[1]
        assert "showing 2-3" in result.content

    def test_read_nonexistent_file(self, temp_dir):
        """读取不存在的文件应返回错误"""
        tool = FileReadTool(base_dir=str(temp_dir))
        result = tool.execute(path="nonexistent.txt")

        assert result.error is not None
        assert "文件不存在" in result.error

    def test_read_directory_traversal_blocked(self, temp_dir):
        """目录遍历攻击应被阻止"""
        tool = FileReadTool(base_dir=str(temp_dir))
        result = tool.execute(path="../outside.txt")

        assert result.error is not None
        assert "只能读取项目目录内的文件" in result.error

    def test_read_directory_instead_of_file(self, temp_dir):
        """路径为目录时应返回错误"""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        tool = FileReadTool(base_dir=str(temp_dir))
        result = tool.execute(path="subdir")

        assert result.error is not None
        assert "路径不是文件" in result.error


class TestFileGlobTool:
    def test_glob_simple_pattern(self, temp_dir):
        """应能匹配简单模式"""
        (temp_dir / "a.py").write_text("x")
        (temp_dir / "b.py").write_text("x")
        (temp_dir / "c.txt").write_text("x")

        tool = FileGlobTool(base_dir=str(temp_dir))
        result = tool.execute(pattern="*.py")

        assert result.error is None
        assert "a.py" in result.content
        assert "b.py" in result.content
        assert "c.txt" not in result.content

    def test_glob_recursive_pattern(self, temp_dir):
        """应支持 ** 递归模式"""
        sub = temp_dir / "src" / "deep"
        sub.mkdir(parents=True)
        (sub / "found.py").write_text("x")

        tool = FileGlobTool(base_dir=str(temp_dir))
        result = tool.execute(pattern="**/*.py")

        assert "found.py" in result.content

    def test_glob_limit(self, temp_dir):
        """应尊重 limit 参数"""
        for i in range(10):
            (temp_dir / f"f{i}.py").write_text("x")

        tool = FileGlobTool(base_dir=str(temp_dir))
        result = tool.execute(pattern="*.py", limit="3")

        assert "Found 3 files" in result.content

    def test_glob_no_match(self, temp_dir):
        """无匹配时应返回空结果"""
        tool = FileGlobTool(base_dir=str(temp_dir))
        result = tool.execute(pattern="*.nonexistent")

        assert "未找到匹配的文件" in result.content


class TestCodeSearchTool:
    def test_search_finds_match(self, temp_dir):
        """应能找到匹配的文本"""
        f = temp_dir / "code.py"
        f.write_text("def hello():\n    pass\n\ndef world():\n    pass\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        result = tool.execute(query="def hello")

        assert result.error is None
        assert "code.py" in result.content
        assert "def hello" in result.content

    def test_search_regex_pattern(self, temp_dir):
        """应支持正则表达式搜索"""
        f = temp_dir / "code.py"
        f.write_text("foo123\nbar456\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        result = tool.execute(query=r"foo\d+")

        assert "foo123" in result.content

    def test_search_glob_filter(self, temp_dir):
        """glob 参数应限制搜索文件类型"""
        (temp_dir / "a.py").write_text("TARGET\n")
        (temp_dir / "b.txt").write_text("TARGET\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        result = tool.execute(query="TARGET", glob="*.py")

        assert "a.py" in result.content
        assert "b.txt" not in result.content

    def test_search_skips_binary_files(self, temp_dir):
        """应跳过二进制文件"""
        (temp_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        result = tool.execute(query="PNG")

        # png 在 skip_exts 中，不应被搜索
        assert "在 0 个文件中未找到匹配" in result.content or "image.png" not in result.content

    def test_search_respects_max_file_size(self, temp_dir):
        """大文件应被跳过"""
        f = temp_dir / "huge.py"
        f.write_text("TARGET\n" + "x" * 1_000_000)

        tool = CodeSearchTool(base_dir=str(temp_dir), max_file_size=100)
        result = tool.execute(query="TARGET")

        assert "在 0 个文件中未找到匹配" in result.content

    def test_search_literal_fallback_on_bad_regex(self, temp_dir):
        """无效正则应回退为字面量搜索"""
        f = temp_dir / "code.py"
        f.write_text("a[b\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        # "[" 单独是无效正则，但回退为字面量后应能匹配
        result = tool.execute(query="a[b")

        assert "a[b" in result.content

    def test_search_no_match(self, temp_dir):
        """无匹配时应返回空结果"""
        f = temp_dir / "code.py"
        f.write_text("hello world\n")

        tool = CodeSearchTool(base_dir=str(temp_dir))
        result = tool.execute(query="nonexistent")

        assert "未找到匹配" in result.content
