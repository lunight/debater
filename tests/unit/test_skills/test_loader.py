"""测试 skills/loader.py — SkillLoader, Skill"""

import pytest

from debater.skills.loader import Skill, SkillLoader


class TestSkill:
    def test_skill_dataclass(self):
        """Skill dataclass 应正确存储属性"""
        skill = Skill(
            name="test",
            description="A test skill",
            body="# Test\nContent",
            meta={"version": "1.0"},
            path="/tmp/test/SKILL.md",
        )
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert "Content" in skill.body
        assert skill.meta["version"] == "1.0"


class TestSkillLoader:
    def test_parse_valid_skill_file(self, temp_dir):
        """应能正确解析有效的 SKILL.md"""
        skill_dir = temp_dir / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: test-skill
description: |
  This is a test skill.
allowed-tools:
  - Read
  - Bash
---

# Test Skill

Some content here.
""")

        loader = SkillLoader(extra_paths=[str(temp_dir)])
        skills = loader.load_all()

        assert "test-skill" in skills
        skill = skills["test-skill"]
        assert skill.name == "test-skill"
        assert "test skill" in skill.description
        assert skill.meta["allowed-tools"] == ["Read", "Bash"]
        assert "Test Skill" in skill.body

    def test_parse_without_frontmatter(self, temp_dir):
        """无 frontmatter 时应使用默认解析"""
        skill_dir = temp_dir / "plain-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("Just plain markdown.\n")

        loader = SkillLoader(extra_paths=[str(temp_dir)])
        skills = loader.load_all()

        assert "plain-skill" in skills
        skill = skills["plain-skill"]
        assert skill.name == "plain-skill"  # fallback to dirname
        assert skill.body == "Just plain markdown.\n"

    def test_parse_malformed_yaml_returns_none(self, temp_dir, capsys):
        """YAML 解析失败时应返回 None 且不崩溃"""
        skill_dir = temp_dir / "bad-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\ninvalid: yaml: [\n---\n")

        loader = SkillLoader(extra_paths=[str(temp_dir)])
        skills = loader.load_all()

        assert "bad-skill" not in skills

    def test_nested_bundle_loading(self, temp_dir):
        """应支持两层嵌套目录（skill bundle）"""
        bundle_dir = temp_dir / "gstack"
        bundle_dir.mkdir()

        sub1 = bundle_dir / "skill-a"
        sub1.mkdir()
        (sub1 / "SKILL.md").write_text("---\nname: skill-a\n---\nA")

        sub2 = bundle_dir / "skill-b"
        sub2.mkdir()
        (sub2 / "SKILL.md").write_text("---\nname: skill-b\n---\nB")

        loader = SkillLoader(extra_paths=[str(temp_dir)])
        skills = loader.load_all()

        assert "skill-a" in skills
        assert "skill-b" in skills

    def test_nonexistent_path_ignored(self):
        """不存在的路径应被静默忽略"""
        loader = SkillLoader(extra_paths=["/nonexistent/path"])
        skills = loader.load_all()
        assert isinstance(skills, dict)
