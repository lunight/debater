"""Skill 加载器

解析 Claude Code 格式的 SKILL.md：
- YAML frontmatter + Markdown body
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Skill:
    """Skill 定义"""
    name: str
    description: str
    body: str
    meta: Dict  # frontmatter 中的其他字段
    path: str    # 文件路径


class SkillLoader:
    """Skill 加载器"""
    
    DEFAULT_PATHS = [
        Path.home() / ".config" / "agents" / "skills",
        Path("/Users/lunight/dev/debater/skills"),
    ]
    
    def __init__(self, extra_paths: Optional[List[str]] = None):
        self.paths = list(self.DEFAULT_PATHS)
        if extra_paths:
            self.paths.extend([Path(p) for p in extra_paths])
    
    def load_all(self) -> Dict[str, Skill]:
        """加载所有 skill（支持嵌套目录，最多两层）"""
        skills = {}
        for base_path in self.paths:
            if not base_path.exists():
                continue
            # 第一层：直接子目录
            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skill = self._parse_skill(skill_file)
                    if skill:
                        skills[skill.name] = skill
                # 第二层：skill bundle（如 gstack/ 下的各个 skill）
                for sub_dir in skill_dir.iterdir():
                    if not sub_dir.is_dir():
                        continue
                    sub_skill_file = sub_dir / "SKILL.md"
                    if sub_skill_file.exists():
                        skill = self._parse_skill(sub_skill_file)
                        if skill:
                            skills[skill.name] = skill
        return skills
    
    def _parse_skill(self, path: Path) -> Optional[Skill]:
        """解析单个 SKILL.md"""
        try:
            content = path.read_text(encoding="utf-8")
            
            # 解析 frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                else:
                    frontmatter = {}
                    body = content
            else:
                frontmatter = {}
                body = content
            
            name = frontmatter.get("name", path.parent.name)
            description = frontmatter.get("description", "")
            meta = {k: v for k, v in frontmatter.items() if k not in ("name", "description")}
            
            return Skill(
                name=name,
                description=description,
                body=body,
                meta=meta,
                path=str(path),
            )
        except Exception as e:
            print(f"[SKILL] 解析失败 {path}: {e}")
            return None
