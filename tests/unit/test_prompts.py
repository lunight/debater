"""测试 prompts.py — 所有 prompt 构建函数"""

import pytest

from debater import prompts


class TestBuildGeneratePrompt:
    def test_round_one_includes_thinking_instruction(self):
        """第一轮应包含 <thinking> 标签说明"""
        result = prompts.build_generate_prompt(
            scenario_type="question_analysis",
            question="test question",
            context="",
            round_num=1,
        )
        assert "<thinking>" in result
        assert "test question" in result
        assert "question_analysis" not in result  # role text, not the key
        assert "资深金融分析师" in result

    def test_round_two_revised_format(self):
        """第二轮应为修订格式"""
        result = prompts.build_generate_prompt(
            scenario_type="risk_assessment",
            question="q",
            context="",
            round_num=2,
        )
        assert "第 2 轮分析" in result
        assert "修正或完善" in result

    def test_context_included_when_provided(self):
        """提供 context 时应包含背景资料部分"""
        result = prompts.build_generate_prompt(
            scenario_type="question_analysis",
            question="q",
            context="some background",
            round_num=1,
        )
        assert "补充背景资料" in result
        assert "some background" in result

    def test_context_omitted_when_empty(self):
        """空 context 时不应包含背景资料部分"""
        result = prompts.build_generate_prompt(
            scenario_type="question_analysis",
            question="q",
            context="  ",
            round_num=1,
        )
        assert "补充背景资料" not in result

    def test_memory_skill_tool_sections(self):
        """memory_context, skill_context, tool_context 应正确注入"""
        result = prompts.build_generate_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            round_num=1,
            memory_context="memory here",
            skill_context="skill here",
            tool_context="tool here",
        )
        assert "memory here" in result
        assert "skill here" in result
        assert "tool here" in result

    def test_invalid_scenario_fallback(self):
        """无效 scenario_type 应回退到默认角色"""
        result = prompts.build_generate_prompt(
            scenario_type="nonexistent",
            question="q",
            context="",
            round_num=1,
        )
        assert "资深金融分析师" in result  # default role


class TestBuildCritiquePrompt:
    def test_includes_other_answers(self):
        """应包含其他模型的分析"""
        result = prompts.build_critique_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            my_answer="my ans",
            other_answers={"deepseek": "ds ans"},
        )
        assert "deepseek 的分析" in result
        assert "ds ans" in result
        assert "my ans" in result

    def test_history_section_when_provided(self):
        """提供 critique_history 时应包含历史评审部分"""
        result = prompts.build_critique_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            my_answer="a",
            other_answers={},
            critique_history="past critiques",
        )
        assert "历史评审记录" in result
        assert "past critiques" in result

    def test_empty_other_answers(self):
        """无其他答案时不应有空的部分"""
        result = prompts.build_critique_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            my_answer="a",
            other_answers={},
        )
        assert "原始问题" in result


class TestBuildAdversarialCritiquePrompt:
    def test_includes_checklist(self):
        """应包含对抗性检查清单"""
        result = prompts.build_adversarial_critique_prompt(
            scenario_type="financial_interpretation",
            question="q",
            context="",
            my_answer="a",
            other_answers={},
        )
        assert "事实核查" in result
        assert "逻辑检查" in result
        assert "跑题检查" in result
        assert "漏答检查" in result
        assert "严格的财务审计师" in result

    def test_adversarial_tone(self):
        """应使用对抗性语气，但基于独立分析"""
        result = prompts.build_adversarial_critique_prompt(
            scenario_type="risk_assessment",
            question="q",
            context="",
            my_answer="a",
            other_answers={},
        )
        assert "你是来挑错的" in result
        assert "独立分析" in result
        assert "刻意差异化" in result
        assert "不是为反对而反对" in result


class TestBuildRevisePrompt:
    def test_includes_critiques(self):
        """应包含收到的评审意见"""
        result = prompts.build_revise_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            my_previous_answer="prev",
            critiques_on_me={"deepseek": "your logic is wrong"},
        )
        assert "deepseek 对你的评审" in result
        assert "your logic is wrong" in result
        assert "prev" in result

    def test_includes_history(self):
        """应包含历史评审演进"""
        result = prompts.build_revise_prompt(
            scenario_type="question_analysis",
            question="q",
            context="",
            my_previous_answer="prev",
            critiques_on_me={},
            critique_history="history of critiques",
        )
        assert "评审演进" in result
        assert "history of critiques" in result


class TestBuildBrainstormPrompt:
    def test_includes_question(self):
        """应包含用户问题"""
        result = prompts.build_brainstorm_prompt(
            question="how to evaluate?",
            context="",
        )
        assert "how to evaluate?" in result
        assert "问题定性" in result
        assert "信息缺口" in result

    def test_context_omitted_when_empty(self):
        """空 context 时不应包含背景资料"""
        result = prompts.build_brainstorm_prompt(
            question="q",
            context="  ",
        )
        assert "已知背景资料" not in result


class TestBuildAggregatePrompt:
    def test_includes_all_answers(self):
        """应包含所有模型的最终观点"""
        result = prompts.build_aggregate_prompt(
            question="q",
            context="ctx",
            all_answers={"kimi": "final1", "deepseek": "final2"},
            all_confidences={"kimi": 0.95, "deepseek": 0.80},
            all_rounds=5,
        )
        assert "final1" in result
        assert "final2" in result
        assert "95%" in result
        assert "80%" in result
        assert "投资委员会主席" in result
