"""
集成测试：验证 DebateNodes 的串行交替行为和 role_id key 约定

关键验证点：
1. generate 结果以 role_id 为 key
2. critique 串行：只有 bear 评审 bull
3. revise 串行：只有 bull 修订
4. judge 使用 LLM-based 判断
5. 状态流转正确
"""

import pytest
from unittest.mock import MagicMock

from debater.nodes import DebateNodes, _extract_confidence
from debater.state import DebateState


@pytest.fixture
def mock_nodes():
    """创建带 mock client 的 DebateNodes"""
    nodes = DebateNodes.__new__(DebateNodes)
    nodes.client = MagicMock()
    nodes.config = MagicMock()
    nodes._tool_registry = MagicMock()

    # mock tool executor
    executor = MagicMock()
    executor.has_tool_calls.return_value = False
    nodes._tool_registry.create_executor.return_value = executor

    # mock config
    mock_model = MagicMock()
    mock_model.id = "kimi_k2.6"
    mock_model.name = "Kimi"
    mock_model2 = MagicMock()
    mock_model2.id = "deepseek_v4"
    mock_model2.name = "DeepSeek"

    nodes.config.get_proposer_models.return_value = {
        "kimi_k2.6": mock_model,
        "deepseek_v4": mock_model2,
    }
    nodes.config.get_aggregator_model.return_value = mock_model
    nodes.config.debate.consensus_threshold = 0.85

    yield nodes


@pytest.fixture
def base_state() -> DebateState:
    return {
        "question": "Test question?",
        "context": "",
        "scenario_type": "question_analysis",
        "round": 0,
        "max_rounds": 3,
        "answers": {},
        "previous_answers": {},
        "critiques": {},
        "confidences": {},
        "consensus_reached": False,
        "final_answer": None,
        "reasoning_process": "",
        "divergence_points": [],
        "role_models": {
            "bull": "kimi_k2.6",
            "bear": "deepseek_v4",
        },
    }


class TestGenerate:
    """验证 generate 节点：并行生成，结果以 role_id 为 key"""

    def test_generate_uses_role_id_as_key(self, mock_nodes, base_state):
        """generate 结果应以 role_id 为 key"""
        def mock_call_parallel(calls):
            results = {}
            for c in calls:
                call_id = c.get("call_id", c["model_id"])
                resp = MagicMock()
                resp.content = f"output_from_{call_id}"
                resp.latency_ms = 500
                results[call_id] = resp
            return results

        mock_nodes.client.call_parallel = mock_call_parallel

        state = mock_nodes.generate(base_state)

        assert "bull" in state["answers"]
        assert "bear" in state["answers"]
        assert state["answers"]["bull"] == "output_from_bull"
        assert state["answers"]["bear"] == "output_from_bear"
        assert "bull" in state["confidences"]
        assert "bear" in state["confidences"]

    def test_generate_call_id_is_role_id(self, mock_nodes, base_state):
        """call_parallel 的 calls 中 call_id 应为 role_id"""
        captured_calls = []

        def mock_call_parallel(calls):
            captured_calls.extend(calls)
            results = {}
            for c in calls:
                call_id = c.get("call_id", c["model_id"])
                resp = MagicMock()
                resp.content = "x"
                resp.latency_ms = 100
                results[call_id] = resp
            return results

        mock_nodes.client.call_parallel = mock_call_parallel
        mock_nodes.generate(base_state)

        call_ids = [c.get("call_id") for c in captured_calls]
        assert "bull" in call_ids
        assert "bear" in call_ids
        # 验证 model_id 与 role_id 映射正确
        for c in captured_calls:
            if c["call_id"] == "bull":
                assert c["model_id"] == "kimi_k2.6"
            elif c["call_id"] == "bear":
                assert c["model_id"] == "deepseek_v4"


class TestCritiqueSerial:
    """验证 critique 节点：串行，只有 bear 评审 bull"""

    def test_only_bear_critiques_bull(self, mock_nodes, base_state):
        """critique 只调用 bear 的模型"""
        call_log = []

        def mock_call(*, model_id, system_prompt, user_prompt):
            call_log.append(model_id)
            resp = MagicMock()
            resp.content = f"critique_by_{model_id}"
            return resp

        mock_nodes.client.call = mock_call

        base_state["answers"] = {"bull": "bull_answer", "bear": "bear_answer"}
        state = mock_nodes.critique(base_state)

        assert call_log == ["deepseek_v4"], f"期望只调用 bear (deepseek_v4)，实际调用: {call_log}"
        assert "bear" in state["critiques"]
        assert "bull" in state["critiques"]["bear"]
        assert state["critiques"]["bear"]["bull"] == "critique_by_deepseek_v4"

    def test_bull_does_not_critique(self, mock_nodes, base_state):
        """bull 不应在 critique 阶段被调用"""
        call_log = []

        def mock_call(*, model_id, system_prompt, user_prompt):
            call_log.append(model_id)
            resp = MagicMock()
            resp.content = "critique"
            return resp

        mock_nodes.client.call = mock_call
        base_state["answers"] = {"bull": "x", "bear": "y"}
        state = mock_nodes.critique(base_state)

        assert "kimi_k2.6" not in call_log, "bull 的模型不应被调用"


class TestReviseSerial:
    """验证 revise 节点：串行，只有 bull 修订"""

    def test_only_bull_revises(self, mock_nodes, base_state):
        """revise 只调用 bull 的模型"""
        call_log = []

        def mock_call(*, model_id, system_prompt, user_prompt):
            call_log.append(model_id)
            resp = MagicMock()
            resp.content = f"revised_by_{model_id}"
            return resp

        mock_nodes.client.call = mock_call

        base_state["answers"] = {"bull": "bull_answer", "bear": "bear_answer"}
        base_state["critiques"] = {"bear": {"bull": "some critique"}}
        state = mock_nodes.revise(base_state)

        assert call_log == ["kimi_k2.6"], f"期望只调用 bull (kimi_k2.6)，实际调用: {call_log}"
        assert state["answers"]["bull"] == "revised_by_kimi_k2.6"

    def test_bear_answer_unchanged(self, mock_nodes, base_state):
        """revise 后 bear 的答案应保持不变"""
        def mock_call(*, model_id, system_prompt, user_prompt):
            resp = MagicMock()
            resp.content = "revised"
            return resp

        mock_nodes.client.call = mock_call

        base_state["answers"] = {"bull": "bull_old", "bear": "bear_old"}
        base_state["critiques"] = {"bear": {"bull": "critique"}}
        state = mock_nodes.revise(base_state)

        assert state["answers"]["bear"] == "bear_old"
        assert state["answers"]["bull"] == "revised"


class TestJudge:
    """验证 judge 节点：LLM-based 判断"""

    def test_judge_parses_json_stop(self, mock_nodes, base_state):
        """judge 应解析 JSON 并返回 stop"""
        def mock_call(*, model_id, system_prompt, user_prompt):
            resp = MagicMock()
            resp.content = '{"action": "stop", "reason": "已统一", "info_needed": ""}'
            return resp

        mock_nodes.client.call = mock_call

        base_state["round"] = 2
        base_state["answers"] = {"bull": "a", "bear": "b"}
        base_state["confidences"] = {"bull": 0.9, "bear": 0.9}
        state = mock_nodes.judge(base_state)

        assert state["consensus_reached"] is True

    def test_judge_parses_json_continue(self, mock_nodes, base_state):
        """judge 应解析 JSON 并继续下一轮"""
        def mock_call(*, model_id, system_prompt, user_prompt):
            resp = MagicMock()
            resp.content = '{"action": "continue", "reason": "还有分歧", "info_needed": ""}'
            return resp

        mock_nodes.client.call = mock_call

        base_state["round"] = 1
        base_state["answers"] = {"bull": "a", "bear": "b"}
        base_state["confidences"] = {"bull": 0.5, "bear": 0.5}
        state = mock_nodes.judge(base_state)

        assert state["consensus_reached"] is False
        assert state["round"] == 2

    def test_judge_fallback_on_api_error(self, mock_nodes, base_state):
        """API 失败时应 fallback 到简单启发式"""
        mock_nodes.client.call.side_effect = Exception("API 挂了")

        base_state["round"] = 3
        base_state["max_rounds"] = 3
        base_state["answers"] = {"bull": "a", "bear": "b"}
        base_state["confidences"] = {"bull": 0.5, "bear": 0.5}
        state = mock_nodes.judge(base_state)

        # 达到最大轮次，fallback 到 stop
        assert state["consensus_reached"] is True


class TestAggregate:
    """验证 aggregate 节点"""

    def test_aggregate_uses_role_id_keys(self, mock_nodes, base_state):
        """aggregate prompt 中的 answers 应以 role_id 为 key"""
        captured_prompt = {}

        def mock_call(*, model_id, system_prompt, user_prompt, temperature=None, max_tokens=None):
            captured_prompt["user"] = user_prompt
            resp = MagicMock()
            resp.content = "final"
            resp.latency_ms = 100
            return resp

        mock_nodes.client.call = mock_call

        base_state["answers"] = {"bull": "bull_final", "bear": "bear_final"}
        base_state["confidences"] = {"bull": 0.8, "bear": 0.85}
        base_state["round"] = 2
        state = mock_nodes.aggregate(base_state)

        assert "bull_final" in captured_prompt.get("user", "")
        assert "bear_final" in captured_prompt.get("user", "")
        assert state["final_answer"] == "final"


class TestInitialize:
    """验证 initialize 节点"""

    def test_initializes_role_models(self, mock_nodes):
        """initialize 应自动设置 role_models"""
        state: DebateState = {
            "question": "q",
            "context": "",
            "scenario_type": "question_analysis",
            "round": 0,
            "max_rounds": 3,
            "answers": {},
            "previous_answers": {},
            "critiques": {},
            "confidences": {},
            "consensus_reached": False,
            "final_answer": None,
            "reasoning_process": "",
            "divergence_points": [],
        }
        result = mock_nodes.initialize(state)
        assert "role_models" in result
        assert result["role_models"]["bull"] == "kimi_k2.6"
        assert result["role_models"]["bear"] == "deepseek_v4"

    def test_preserves_existing_role_models(self, mock_nodes):
        """initialize 不应覆盖已存在的 role_models"""
        state: DebateState = {
            "question": "q",
            "context": "",
            "scenario_type": "question_analysis",
            "round": 0,
            "max_rounds": 3,
            "answers": {},
            "previous_answers": {},
            "critiques": {},
            "confidences": {},
            "consensus_reached": False,
            "final_answer": None,
            "reasoning_process": "",
            "divergence_points": [],
            "role_models": {"bull": "custom_model", "bear": "custom_model"},
        }
        result = mock_nodes.initialize(state)
        assert result["role_models"]["bull"] == "custom_model"


class TestExtractConfidence:
    def test_chinese_format(self):
        assert _extract_confidence("置信度：85%") == 0.85

    def test_no_confidence_returns_default(self):
        assert _extract_confidence("no confidence") == 0.7
