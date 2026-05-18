"""测试 skills/registry.py — SkillRegistry"""

import pytest

from debater.skills.registry import SkillRegistry
from debater.skills.loader import Skill


class TestSkillRegistry:
    def test_get_by_scenario_exact_match(self):
        """精确匹配 scenario_type 作为 skill name"""
        reg = SkillRegistry()
        reg._skills = {
            "financial_interpretation": Skill(
                name="financial_interpretation",
                description="Financial analysis",
                body="body",
                meta={},
                path="",
            )
        }
        reg._loaded = True

        skill = reg.get_by_scenario("financial_interpretation")
        assert skill is not None
        assert skill.name == "financial_interpretation"

    def test_get_by_scenario_name_contains(self):
        """skill name 包含 scenario_type"""
        reg = SkillRegistry()
        reg._skills = {
            "financial-analysis-3-statement-model": Skill(
                name="financial-analysis-3-statement-model",
                description="Desc",
                body="body",
                meta={},
                path="",
            )
        }
        reg._loaded = True

        # financial_analysis -> financial-analysis, which is in the name
        skill = reg.get_by_scenario("financial_analysis")
        assert skill is not None
        assert "financial-analysis" in skill.name

    def test_get_by_scenario_description_match(self):
        """description 包含 scenario_type 关键词"""
        reg = SkillRegistry()
        reg._skills = {
            "some-skill": Skill(
                name="some-skill",
                description="This is about risk assessment techniques",
                body="body",
                meta={},
                path="",
            )
        }
        reg._loaded = True

        skill = reg.get_by_scenario("risk_assessment")
        assert skill is not None
        assert skill.name == "some-skill"

    def test_get_by_scenario_no_match(self):
        """无匹配时应返回 None"""
        reg = SkillRegistry()
        reg._skills = {}
        reg._loaded = True

        assert reg.get_by_scenario("nonexistent") is None

    def test_get_context_for_scenario_returns_empty_on_no_match(self):
        """无匹配时 get_context_for_scenario 应返回空字符串"""
        reg = SkillRegistry()
        reg._skills = {}
        reg._loaded = True

        ctx = reg.get_context_for_scenario("anything")
        assert ctx == ""

    def test_get_context_for_scenario_includes_body(self):
        """匹配时应返回包含 skill body 的 context"""
        reg = SkillRegistry()
        reg._skills = {
            "test": Skill(
                name="test",
                description="A test skill",
                body="Skill body content here",
                meta={},
                path="",
            )
        }
        reg._loaded = True

        ctx = reg.get_context_for_scenario("test")
        assert "专业 Skill 指导" in ctx
        assert "Skill body" in ctx
        assert "A test skill" in ctx

    def test_list_all(self):
        """list_all 应返回所有 skill"""
        reg = SkillRegistry()
        reg._skills = {
            "a": Skill(name="a", description="", body="", meta={}, path=""),
            "b": Skill(name="b", description="", body="", meta={}, path=""),
        }
        reg._loaded = True

        assert len(reg.list_all()) == 2

    def test_lazy_load(self):
        """load 应为延迟加载（多次调用不重复）"""
        reg = SkillRegistry()
        # 第一次加载
        reg.load()
        assert reg._loaded is True
        # 第二次应直接返回
        reg.load()  # 不应抛出异常或重复加载

    def test_scenario_explicit_map_priority(self):
        """显式映射优先级应高于精确匹配和 description 匹配"""
        reg = SkillRegistry()
        reg._skills = {
            # 精确匹配同名但不应被选中（显式映射指向另一个）
            "question_analysis": Skill(
                name="question_analysis",
                description="Old question analysis",
                body="old body",
                meta={},
                path="",
            ),
            # 显式映射目标
            "brainstorming": Skill(
                name="brainstorming",
                description="Brainstorm skill",
                body="brainstorm body",
                meta={},
                path="",
            ),
            # description 中包含 analysis 的干扰项
            "competitive-analysis": Skill(
                name="competitive-analysis",
                description="Framework for competitive analysis",
                body="comp body",
                meta={},
                path="",
            ),
        }
        reg._loaded = True

        skill = reg.get_by_scenario("question_analysis")
        assert skill is not None
        # 显式映射：question_analysis -> brainstorming
        assert skill.name == "brainstorming", f"期望 brainstorming，实际得到 {skill.name}"

    def test_scenario_explicit_map_all_four_scenarios(self):
        """四个场景都应正确匹配到显式映射的 skill"""
        reg = SkillRegistry()
        reg._skills = {
            "brainstorming": Skill(
                name="brainstorming", description="Brainstorm", body="", meta={}, path=""
            ),
            "3-statement-model": Skill(
                name="3-statement-model", description="3SM", body="", meta={}, path=""
            ),
            "competitive-analysis": Skill(
                name="competitive-analysis", description="Comp", body="", meta={}, path=""
            ),
            "systematic-debugging": Skill(
                name="systematic-debugging", description="Debug", body="", meta={}, path=""
            ),
        }
        reg._loaded = True

        assert reg.get_by_scenario("question_analysis").name == "brainstorming"
        assert reg.get_by_scenario("financial_interpretation").name == "3-statement-model"
        assert reg.get_by_scenario("event_analysis").name == "competitive-analysis"
        assert reg.get_by_scenario("risk_assessment").name == "systematic-debugging"

    def test_scenario_fallback_when_no_explicit_map(self):
        """无显式映射时应 fallback 到精确匹配"""
        reg = SkillRegistry()
        reg._skills = {
            "custom_scenario": Skill(
                name="custom_scenario",
                description="Custom",
                body="",
                meta={},
                path="",
            ),
        }
        reg._loaded = True

        # custom_scenario 不在 SCENARIO_SKILL_MAP 中
        skill = reg.get_by_scenario("custom_scenario")
        assert skill is not None
        assert skill.name == "custom_scenario"

    def test_load_excludes_gstack_skills(self):
        """加载后不应包含已删除的 gstack skills"""
        reg = SkillRegistry()
        reg.load()

        gstack_names = [
            "browse", "scrape", "ship", "canary", "freeze", "unfreeze",
            "cso", "guard", "qa", "qa-only", "design-html", "design-review",
            "plan-ceo-review", "land-and-deploy", "setup-deploy",
        ]
        for name in gstack_names:
            assert name not in reg._skills, f"gstack skill '{name}' 不应被加载"

        # 确认有用的 skills 仍然存在
        useful_names = [
            "brainstorming", "systematic-debugging", "3-statement-model",
            "competitive-analysis", "karpathy-guidelines", "test-driven-development",
        ]
        for name in useful_names:
            assert name in reg._skills, f"有用 skill '{name}' 应被加载"
