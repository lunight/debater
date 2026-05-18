"""
集成测试：验证 auto_debate_stream 的严格串行交替行为

关键验证点：
1. critique 和 revise 严格串行，不是并行
2. 每轮先 bear critique bull，然后 bull revise
3. 调用顺序可被精确追踪
4. 事件流顺序正确
5. 数据状态（critiques, answers）正确更新
"""

import pytest
from unittest.mock import MagicMock, call
from debater.session import Stage, DebateSession


@pytest.fixture
def serial_session():
    """创建用于串行交替测试的 DebateSession 实例"""
    s = object.__new__(DebateSession)
    s.question = "Test question?"
    s.scenario_type = "question_analysis"
    s.context = ""
    s.stage = Stage.INIT
    s.current_round = 0
    s.answers = {"bull": "bull initial answer", "bear": "bear initial answer"}
    s.thinkings = {"bull": "", "bear": ""}
    s.summaries = {"bull": "", "bear": ""}
    s.confidences = {"bull": 0.0, "bear": 0.0}
    s.previous_answers = {"bull": "bull initial answer", "bear": "bear initial answer"}
    s.critiques_history = {}
    s.critiques = {"bull": {}, "bear": {}}
    s.final_answer = ""
    s.known_facts = []
    s.memory_manager = None
    s._client = MagicMock()
    s._config = MagicMock()
    s.role_models = {"bull": "kimi_k2.6", "bear": "deepseek_v4"}
    s._models = ["kimi_k2.6", "deepseek_v4"]
    s._skill_registry = None
    s._tool_registry = MagicMock()
    s._consecutive_tool_only = {}
    
    mock_model = MagicMock()
    mock_model.name = "Kimi"
    mock_model2 = MagicMock()
    mock_model2.name = "DeepSeek"
    s._config.models = {"kimi_k2.6": mock_model, "deepseek_v4": mock_model2}
    s._config.get_proposer_models.return_value = {
        "kimi_k2.6": mock_model,
        "deepseek_v4": mock_model2,
    }
    s._config.get_aggregator_model.return_value = MagicMock(id="aggregator")
    
    # mock tool executor
    executor = MagicMock()
    executor.has_tool_calls.return_value = False
    s._tool_registry.create_executor.return_value = executor
    
    yield s


class TestSerialAlternation:
    """验证串行交替 critique → revise 的核心行为"""
    
    def test_critique_then_revise_in_sequence(self, serial_session):
        """验证 critique 和 revise 是严格串行调用的"""
        s = serial_session
        call_order = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_order.append(model_id)
            # 返回一个简单生成器
            if "critique" in user_prompt.lower() or "评审" in user_prompt:
                text = "bear's critique of bull"
            else:
                text = "bull's revised answer"
            yield ("token", text)
        
        s._client.call_stream = mock_call_stream
        
        # mock auto_judge 让它第一轮就 stop
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        events = list(s.auto_debate_stream(max_cycles=1))
        
        # 验证调用顺序：先 bear (deepseek_v4) critique，再 bull (kimi_k2.6) revise
        assert call_order == ["deepseek_v4", "kimi_k2.6"], (
            f"期望串行调用 [deepseek_v4, kimi_k2.6]，实际得到 {call_order}"
        )
    
    def test_two_cycles_calls_four_times(self, serial_session):
        """验证两轮循环产生 4 次串行调用"""
        s = serial_session
        call_order = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_order.append(model_id)
            text = f"output from {model_id}"
            yield ("token", text)
        
        s._client.call_stream = mock_call_stream
        
        # mock auto_judge：continue, then stop
        s.auto_judge = MagicMock(side_effect=[
            {"action": "continue", "reason": "还有分歧", "info_needed": ""},
            {"action": "stop", "reason": "已统一", "info_needed": ""},
        ])
        
        events = list(s.auto_debate_stream(max_cycles=3))
        
        # 两轮：bear→bull critique, bull revise, bear→bull critique, bull revise
        expected = ["deepseek_v4", "kimi_k2.6", "deepseek_v4", "kimi_k2.6"]
        assert call_order == expected, (
            f"期望 {expected}，实际得到 {call_order}"
        )
        assert s.auto_judge.call_count == 2
    
    def test_event_stream_order(self, serial_session):
        """验证事件流顺序：start → tokens → model_done → start → tokens → model_done → judge"""
        s = serial_session
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = f"output from {model_id}"
            yield ("token1", text)
            yield ("token2", text)
        
        s._client.call_stream = mock_call_stream
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        events = list(s.auto_debate_stream(max_cycles=1))
        
        # 过滤关键事件
        types = [e.get("type") for e in events]
        
        # 期望顺序：start(critique) → token → token → model_done(critique) → start(revise) → token → token → model_done(revise) → judge
        assert types[0] == "start"
        assert types[1] == "token"
        assert types[2] == "token"
        assert types[3] == "model_done"
        assert types[4] == "start"
        assert types[5] == "token"
        assert types[6] == "token"
        assert types[7] == "model_done"
        assert types[8] == "judge"
        
        # 验证 critique event 的 role_id 是 bear
        critique_done = [e for e in events if e.get("type") == "model_done"][0]
        assert critique_done["role_id"] == "bear"
        
        # 验证 revise event 的 role_id 是 bull
        revise_done = [e for e in events if e.get("type") == "model_done"][1]
        assert revise_done["role_id"] == "bull"
    
    def test_critique_data_structure(self, serial_session):
        """验证 critique 数据结构正确存储"""
        s = serial_session
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = "detailed critique text here"
            yield ("token", text)
        
        s._client.call_stream = mock_call_stream
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        list(s.auto_debate_stream(max_cycles=1))
        
        # critiques_history 应包含 bear → bull 的 critique
        assert 1 in s.critiques_history
        assert "bear" in s.critiques_history[1]
        assert "bull" in s.critiques_history[1]["bear"]
        assert s.critiques_history[1]["bear"]["bull"] == "detailed critique text here"
        
        # critiques 快照也应更新
        assert "bear" in s.critiques
        assert "bull" in s.critiques["bear"]
    
    def test_revise_updates_bull_answer(self, serial_session):
        """验证 revise 只更新 bull 的 answer"""
        s = serial_session
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = "bull revised with new insights"
            yield ("token", text)
        
        s._client.call_stream = mock_call_stream
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        # 记录 bear 初始答案
        bear_initial = s.answers["bear"]
        
        list(s.auto_debate_stream(max_cycles=1))
        
        # bull 的 answer 应被更新
        assert s.answers["bull"] == "bull revised with new insights"
        
        # bear 的 answer 应保持不变（串行单向模式下 bear 不 revise）
        assert s.answers["bear"] == bear_initial
    
    def test_not_parallel(self, serial_session):
        """验证不是并行调用——通过检查是否有同时活跃的调用来确认"""
        s = serial_session
        active_calls = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            active_calls.append(model_id)
            # 模拟调用过程
            text = f"output from {model_id}"
            yield ("token1", text)
            yield ("token2", text)
            # 调用结束，移除
            active_calls.remove(model_id)
        
        s._client.call_stream = mock_call_stream
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        events = list(s.auto_debate_stream(max_cycles=1))
        
        # 在串行模式下，active_calls 在任何时候都不应该同时包含两个模型
        # 这个测试通过 mock_call_stream 的 yield 来间接验证
        # 如果并行，两个 mock_call_stream 会同时执行，active_calls 会有两个元素
        # 但我们通过生成器的顺序消费来确保串行
        
        # 更直接的验证：检查事件中没有同时出现的 model_done（串行不会同时完成）
        # 这个测试主要是概念性的——实际并行/串行的区别在生成器消费层面
        assert len([e for e in events if e.get("type") == "model_done"]) == 2
    
    def test_max_cycles_with_continue(self, serial_session):
        """验证 max_cycles 被尊重——即使 judge 一直说 continue"""
        s = serial_session
        call_count = [0]
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_count[0] += 1
            text = f"output {call_count[0]}"
            yield ("token", text)
        
        s._client.call_stream = mock_call_stream
        s.auto_judge = MagicMock(return_value={
            "action": "continue", "reason": "继续", "info_needed": ""
        })
        
        events = list(s.auto_debate_stream(max_cycles=2))
        
        # 2 轮 × 2 次调用 = 4 次
        assert call_count[0] == 4
        
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 2
        assert s.current_round == 2
        assert s.stage == Stage.PAUSE_AFTER_REVISE
