"""测试角色系统"""

import pytest
from debater.roles import ROLES, get_role, get_role_display_name


class TestRoles:
    def test_bull_role_exists(self):
        assert "bull" in ROLES
        role = ROLES["bull"]
        assert role.id == "bull"
        assert role.name == "支持者"
        assert "🐂" in role.emoji
        assert len(role.generate_prompt) > 50
        assert len(role.critique_prompt) > 50

    def test_bear_role_exists(self):
        assert "bear" in ROLES
        role = ROLES["bear"]
        assert role.id == "bear"
        assert role.name == "质疑者"
        assert "🐻" in role.emoji
        assert len(role.generate_prompt) > 50
        assert len(role.critique_prompt) > 50

    def test_get_role_valid(self):
        bull = get_role("bull")
        assert bull.id == "bull"
        bear = get_role("bear")
        assert bear.id == "bear"

    def test_get_role_invalid_fallback(self):
        """无效角色 ID 应回退到 bear"""
        role = get_role("nonexistent")
        assert role.id == "bear"

    def test_get_role_display_name_with_model(self):
        name = get_role_display_name("bull", "kimi_k2.6")
        assert "支持者" in name
        assert "kimi_k2.6" in name

    def test_get_role_display_name_without_model(self):
        name = get_role_display_name("bear")
        assert "质疑者" in name
