"""文件系统工具 — 移植自 Claude Code 的 Read/Glob/Grep

在辩论场景中，模型可能需要读取项目文件、搜索代码库来支持论证。
这些工具提供只读访问，不支持写入/修改（Bash/Write/Edit 因安全和场景原因不移植）。
"""

import fnmatch
import os
import re
from pathlib import Path
from typing import List, Optional

from .base import Tool, ToolResult


class FileReadTool(Tool):
    """读取文件内容 — 对应 Claude Read"""
    
    name = "file_read"
    description = "读取指定文件的内容，支持行范围截取。用于查看代码、配置文件、文档等。"
    parameters = {
        "path": "文件路径（相对于项目根目录或绝对路径）",
        "offset": "可选，起始行号（1-based）",
        "limit": "可选，最多读取行数，默认100行",
    }
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    def execute(self, path: str, offset: str = "1", limit: str = "100") -> ToolResult:
        try:
            file_path = self.base_dir / path
            file_path = file_path.resolve()
            
            # 安全检查：限制在项目目录内
            if not str(file_path).startswith(str(self.base_dir.resolve())):
                return ToolResult(content="", error="只能读取项目目录内的文件")
            
            if not file_path.exists():
                return ToolResult(content="", error=f"文件不存在: {path}")
            
            if not file_path.is_file():
                return ToolResult(content="", error=f"路径不是文件: {path}")
            
            # 读取内容
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            start = max(0, int(offset) - 1) if offset else 0
            max_lines = int(limit) if limit else 100
            end = min(start + max_lines, len(lines))
            
            selected = lines[start:end]
            result = "\n".join(selected)
            
            # 添加行号
            numbered = "\n".join([
                f"{start + i + 1:4d} | {line}"
                for i, line in enumerate(selected)
            ])
            
            header = f"File: {path} ({len(lines)} lines total, showing {start+1}-{end})"
            return ToolResult(
                content=f"{header}\n{'='*60}\n{numbered}",
                sources=[str(file_path)],
            )
            
        except Exception as e:
            return ToolResult(content="", error=f"读取失败: {e}")


class FileGlobTool(Tool):
    """文件搜索 — 对应 Claude Glob"""
    
    name = "file_glob"
    description = "搜索匹配模式的文件路径。支持 * 和 ** 通配符。用于探索代码库结构。"
    parameters = {
        "pattern": "搜索模式，如 '*.py', 'src/**/*.ts', 'README*'",
        "limit": "可选，最多返回结果数，默认20",
    }
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    def execute(self, pattern: str, limit: str = "20") -> ToolResult:
        try:
            max_results = int(limit) if limit else 20
            
            # 使用 pathlib.rglob 或 glob
            if pattern.startswith("/"):
                search_path = Path(pattern)
                if "**" in pattern:
                    parts = pattern.split("**")
                    base = Path(parts[0]) if parts[0] else Path("/")
                    rest = parts[1].lstrip("/") if len(parts) > 1 else ""
                    matches = list(base.rglob(rest)) if rest else list(base.rglob("*"))
                else:
                    matches = list(Path("/").glob(pattern[1:]))
            else:
                rel_pattern = pattern.lstrip("./")
                if "**" in rel_pattern:
                    parts = rel_pattern.split("**")
                    base = self.base_dir / parts[0].rstrip("/")
                    rest = parts[1].lstrip("/") if len(parts) > 1 else ""
                    if base.exists():
                        matches = list(base.rglob(rest)) if rest else list(base.rglob("*"))
                    else:
                        matches = []
                else:
                    matches = list(self.base_dir.glob(rel_pattern))
            
            # 过滤并限制结果
            results = []
            for m in matches:
                if m.is_file():
                    try:
                        rel = m.relative_to(self.base_dir)
                        results.append(str(rel))
                    except ValueError:
                        results.append(str(m))
                    if len(results) >= max_results:
                        break
            
            if not results:
                return ToolResult(content="未找到匹配的文件", sources=[])
            
            header = f"Found {len(results)} files matching '{pattern}'"
            return ToolResult(
                content=f"{header}\n" + "\n".join(f"  {r}" for r in results),
                sources=results,
            )
            
        except Exception as e:
            return ToolResult(content="", error=f"搜索失败: {e}")


class CodeSearchTool(Tool):
    """代码搜索 — 对应 Claude Grep"""
    
    name = "code_search"
    description = "在代码库中搜索匹配的内容。支持正则表达式，可限制文件类型。用于查找函数定义、引用、特定模式等。"
    parameters = {
        "query": "搜索字符串或正则表达式",
        "path": "可选，限制搜索的文件路径模式，如 '*.py', 'src/'",
        "glob": "可选，文件匹配模式，如 '*.py', '*.ts'",
    }
    
    def __init__(self, base_dir: Optional[str] = None, max_file_size: int = 500_000):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.max_file_size = max_file_size
    
    def execute(self, query: str, path: str = "", glob: str = "") -> ToolResult:
        try:
            # 编译正则
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error:
                # 如果正则编译失败，作为字面量搜索
                pattern = re.compile(re.escape(query), re.IGNORECASE)
            
            # 确定搜索范围
            search_dir = self.base_dir
            if path:
                search_dir = self.base_dir / path
            
            # 文件过滤模式
            file_glob = glob or "*"
            if "**" not in file_glob and "/" not in file_glob:
                # 简单扩展名模式，递归搜索
                if file_glob.startswith("*"):
                    file_pattern = file_glob
                else:
                    file_pattern = f"**/{file_glob}"
            else:
                file_pattern = file_glob
            
            # 常见二进制扩展名跳过
            skip_exts = {
                '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
                '.pdf', '.zip', '.tar', '.gz', '.rar',
                '.woff', '.woff2', '.ttf', '.eot', '.otf',
                '.mp3', '.mp4', '.avi', '.mov',
                '.exe', '.dll', '.so', '.dylib',
                '.pyc', '.pyo', '.class', '.o',
            }
            
            matches = []
            files_searched = 0
            max_matches = 50
            
            if search_dir.is_file():
                files_to_search = [search_dir]
            else:
                files_to_search = list(search_dir.rglob(file_pattern.lstrip("/")))
            
            for file_path in files_to_search:
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() in skip_exts:
                    continue
                if file_path.stat().st_size > self.max_file_size:
                    continue
                
                files_searched += 1
                if files_searched > 1000:  # 限制搜索文件数
                    break
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    
                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            try:
                                rel_path = file_path.relative_to(self.base_dir)
                            except ValueError:
                                rel_path = file_path
                            
                            matches.append({
                                "path": str(rel_path),
                                "line": i + 1,
                                "text": line.strip()[:200],  # 截断过长行
                            })
                            
                            if len(matches) >= max_matches:
                                break
                    
                    if len(matches) >= max_matches:
                        break
                        
                except Exception as e:
                    from ..logger import tool_logger
                    tool_logger.warning(
                        f"[CodeSearchTool] 跳过无法读取的文件: {file_path}\n"
                        f"Error: {type(e).__name__}: {e}"
                    )
                    continue
            
            if not matches:
                return ToolResult(
                    content=f"在 {files_searched} 个文件中未找到匹配 '{query}'",
                    sources=[],
                )
            
            lines = [f"Found {len(matches)} matches for '{query}' (searched {files_searched} files):"]
            for m in matches:
                lines.append(f"\n{m['path']}:{m['line']}")
                lines.append(f"  {m['text']}")
            
            if len(matches) >= max_matches:
                lines.append("\n... (results truncated, refine your search)")
            
            return ToolResult(
                content="\n".join(lines),
                sources=[m["path"] for m in matches],
            )
            
        except Exception as e:
            return ToolResult(content="", error=f"搜索失败: {e}")
