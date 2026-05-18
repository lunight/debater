"""
全流程集成测试：generate → critique → revise → judge 完整串行交替流程

验证点：
1. generate 阶段双方并行生成初始观点
2. critique 阶段严格串行（bear → bull）
3. revise 阶段严格串行（bull 修订）
4. judge 阶段正确调用
5. 状态流转正确（stage、answers、critiques 等）
6. 多轮循环时 critique/revise 仍保持串行
"""

import pytest
from unittest.mock import MagicMock
from debater.session import Stage, DebateSession


@pytest.fixture
def full_session():
    """创建用于全流程测试的 DebateSession 实例"""
    s = object.__new__(DebateSession)
    s.question = "Test question?"
    s.scenario_type = "question_analysis"
    s.context = ""
    s.stage = Stage.INIT
    s.current_round = 0
    s.answers = {"bull": "", "bear": ""}
    s.thinkings = {"bull": "", "bear": ""}
    s.summaries = {"bull": "", "bear": ""}
    s.confidences = {"bull": 0.0, "bear": 0.0}
    s.previous_answers = {"bull": "", "bear": ""}
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
    
    executor = MagicMock()
    executor.has_tool_calls.return_value = False
    s._tool_registry.create_executor.return_value = executor
    
    yield s


class TestFullDebateFlow:
    """验证从 generate 到 judge 的完整串行交替流程"""
    
    def test_generate_parallel_then_serial_critique_revise(self, full_session):
        """generate 并行 → critique 串行 → revise 串行 → judge"""
        s = full_session
        call_order = []
        
        def mock_call(*, model_id, system_prompt, user_prompt):
            call_order.append(("call", model_id))
            return MagicMock(content=f"sync_output_from_{model_id}")
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_order.append(("stream", model_id))
            text = f"stream_output_from_{model_id}"
            yield ("token", text)
        
        def mock_call_parallel(calls):
            # calls 是 list of dict with "call_id" (role_id)
            return {
                c["call_id"]: MagicMock(content=f"sync_output_from_{c['model_id']}")
                for c in calls
            }
        
        s._client.call = mock_call
        s._client.call_stream = mock_call_stream
        s._client.call_parallel = mock_call_parallel
        
        # mock auto_judge 让它返回 stop
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        # ========== Step 1: GENERATE ==========
        s.generate()
        
        # generate 是并行的，call_parallel 被调用
        # 检查 generate 阶段的调用：call_parallel 一次，包含两个模型
        assert s.current_round == 1
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        assert s.answers["bull"] == "sync_output_from_kimi_k2.6"
        assert s.answers["bear"] == "sync_output_from_deepseek_v4"
        
        # ========== Step 2: CRITIQUE ==========
        s.critique()
        
        # critique 是串行的，只调用 deepseek_v4 (bear)
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert "bear" in s.critiques
        assert "bull" in s.critiques["bear"]
        assert s.critiques["bear"]["bull"] == "sync_output_from_deepseek_v4"
        # bear 不 critique bear 自己
        assert "bear" not in s.critiques.get("bear", {})
        
        # ========== Step 3: REVISE ==========
        s.revise()
        
        # revise 是串行的，只调用 kimi_k2.6 (bull)
        assert s.stage == Stage.PAUSE_AFTER_REVISE
        assert s.answers["bull"] == "sync_output_from_kimi_k2.6"  # bull 被更新
        # bear 的答案保持 generate 阶段的结果
        assert s.answers["bear"] == "sync_output_from_deepseek_v4"
    
    def test_continue_next_flow(self, full_session):
        """测试 continue_next 的完整状态流转"""
        s = full_session
        
        def mock_call(*, model_id, system_prompt, user_prompt):
            return MagicMock(content=f"output_{model_id}")
        
        s._client.call = mock_call
        
        # 初始状态
        assert s.stage == Stage.INIT
        
        # Step 1: generate
        s.generate()
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        
        # Step 2: continue_next 从 PAUSE_AFTER_GENERATE → critique
        s.continue_next()
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert s.current_round == 1
        
        # Step 3: continue_next 从 PAUSE_AFTER_CRITIQUE → revise
        s.continue_next()
        assert s.stage == Stage.PAUSE_AFTER_REVISE
        
        # Step 4: continue_next 从 PAUSE_AFTER_REVISE → 下一轮 critique
        s.continue_next()
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert s.current_round == 2
    
    def test_continue_next_never_regenerates(self, full_session):
        """核心测试：continue_next 在多轮循环时绝不回到 generate"""
        s = full_session
        call_log = []
        
        def mock_call(*, model_id, system_prompt, user_prompt):
            call_log.append(("call", model_id, user_prompt[:50]))
            return MagicMock(content=f"output_{model_id}")
        
        def mock_call_parallel(calls):
            call_log.append(("call_parallel", [c["model_id"] for c in calls]))
            return {
                c["call_id"]: MagicMock(content=f"output_{c['model_id']}")
                for c in calls
            }
        
        s._client.call = mock_call
        s._client.call_parallel = mock_call_parallel
        
        # 第一轮完整流程
        s.generate()                          # → PAUSE_AFTER_GENERATE
        s.continue_next()                     # → PAUSE_AFTER_CRITIQUE (round=1)
        s.continue_next()                     # → PAUSE_AFTER_REVISE (round=1)
        
        # 记录第一轮结束后的调用
        calls_after_round1 = list(call_log)
        
        # 第二轮：continue_next 从 PAUSE_AFTER_REVISE 进入下一轮
        s.continue_next()                     # → PAUSE_AFTER_CRITIQUE (round=2)
        
        # 验证第二轮没有调用 call_parallel（generate 会用 call_parallel）
        second_round_calls = call_log[len(calls_after_round1):]
        parallel_calls_in_round2 = [c for c in second_round_calls if c[0] == "call_parallel"]
        assert len(parallel_calls_in_round2) == 0, (
            f"第二轮不应调用 generate（call_parallel），实际调用了 {parallel_calls_in_round2}"
        )
        
        # 验证第二轮只调用了 critique（bear=deepseek_v4）
        round2_model_calls = [c[1] for c in second_round_calls if c[0] == "call"]
        assert round2_model_calls == ["deepseek_v4"], (
            f"第二轮应只调用 bear critique（deepseek_v4），实际调用了 {round2_model_calls}"
        )
        
        # 继续第二轮 revise
        s.continue_next()                     # → PAUSE_AFTER_REVISE (round=2)
        
        third_round_calls = call_log[len(calls_after_round1) + len(second_round_calls):]
        round2_revise_calls = [c[1] for c in third_round_calls if c[0] == "call"]
        assert round2_revise_calls == ["kimi_k2.6"], (
            f"第二轮 revise 应只调用 bull（kimi_k2.6），实际调用了 {round2_revise_calls}"
        )
        
        # 验证整个流程的调用顺序
        all_model_calls = [c[1] for c in call_log if c[0] == "call"]
        # Round1: critique(bear) → revise(bull)
        # Round2: critique(bear) → revise(bull)
        expected = ["deepseek_v4", "kimi_k2.6", "deepseek_v4", "kimi_k2.6"]
        assert all_model_calls == expected, (
            f"期望 {expected}，实际 {all_model_calls}"
        )
    
    def test_auto_debate_stream_full_flow(self, full_session):
        """auto_debate_stream 完整流程：generate + auto_debate"""
        s = full_session
        call_order = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_order.append(model_id)
            text = f"stream_{model_id}"
            yield ("t1", text)
            yield ("t2", text)
        
        s._client.call_stream = mock_call_stream
        s._client.call = MagicMock(return_value=MagicMock(content="judge_output"))
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        # generate（并行）
        s.generate()
        
        # auto_debate_stream（串行）
        events = list(s.auto_debate_stream(max_cycles=1))
        
        # 验证事件流包含 critique + revise + judge
        types = [e.get("type") for e in events]
        assert "start" in types
        assert "token" in types
        assert "model_done" in types
        assert "judge" in types
        
        # 验证 critique 只产生 bear 的 model_done
        critique_dones = [e for e in events if e.get("type") == "model_done" and e.get("role_id") == "bear"]
        assert len(critique_dones) == 1
        
        # 验证 revise 只产生 bull 的 model_done
        revise_dones = [e for e in events if e.get("type") == "model_done" and e.get("role_id") == "bull"]
        assert len(revise_dones) == 1
        
        # 验证 judge
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 1
        assert judge_events[0]["action"] == "stop"
    
    def test_multiple_cycles_remain_serial(self, full_session):
        """多轮循环时 critique/revise 仍保持串行"""
        s = full_session
        call_order = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_order.append(model_id)
            text = f"stream_{model_id}"
            yield ("t", text)
        
        s._client.call_stream = mock_call_stream
        s._client.call = MagicMock(return_value=MagicMock(content="judge"))
        s.auto_judge = MagicMock(side_effect=[
            {"action": "continue", "reason": "继续", "info_needed": ""},
            {"action": "stop", "reason": "已统一", "info_needed": ""},
        ])
        
        s.generate()
        events = list(s.auto_debate_stream(max_cycles=3))
        
        # 两轮：每轮 [bear critique, bull revise]
        expected = ["deepseek_v4", "kimi_k2.6", "deepseek_v4", "kimi_k2.6"]
        assert call_order == expected, f"期望 {expected}，实际 {call_order}"
        
        # 两个 judge 事件
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 2
    
    def test_bear_does_not_revise_in_serial_mode(self, full_session):
        """串行模式下 bear 不应该 revise"""
        s = full_session
        
        def mock_call(*, model_id, system_prompt, user_prompt):
            return MagicMock(content=f"output_{model_id}")
        
        s._client.call = mock_call
        
        s.generate()
        
        # 记录 bear 的初始答案
        bear_initial = s.answers["bear"]
        
        s.critique()
        s.revise()
        
        # bear 的答案不应该被 revise 改变
        assert s.answers["bear"] == bear_initial
        # bull 的答案应该被更新（虽然 mock 返回相同格式，但逻辑上应该更新了）
        assert s.previous_answers["bull"] == "output_kimi_k2.6"
    
    def test_critiques_history_single_direction(self, full_session):
        """critiques_history 只记录 bear→bull 的单向关系"""
        s = full_session
        
        s._client.call = MagicMock(return_value=MagicMock(content="critique_text"))
        
        s.generate()
        s.critique()
        
        # critiques_history 应该有 bear → bull
        assert 1 in s.critiques_history
        assert "bear" in s.critiques_history[1]
        assert "bull" in s.critiques_history[1]["bear"]
        assert s.critiques_history[1]["bear"]["bull"] == "critique_text"
        
        # 不应该有 bull → bear（串行模式下 bull 不 critique）
        assert "bull" not in s.critiques_history[1]
    
    def test_no_bull_critique_in_stream_events(self, full_session):
        """事件流中不应出现 bull 的 critique 事件"""
        s = full_session
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = f"stream_{model_id}"
            yield ("t", text)
        
        s._client.call_stream = mock_call_stream
        s._client.call = MagicMock(return_value=MagicMock(content="judge"))
        s.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        s.generate()
        events = list(s.auto_debate_stream(max_cycles=1))
        
        # 所有 model_done 事件
        model_dones = [e for e in events if e.get("type") == "model_done"]
        
        # 不应该有 bull 的 critique（即 bull 的 model_done 在 critique 阶段）
        # 实际上 auto_debate_stream 中 model_done 的顺序是：bear(critique) → bull(revise)
        # 所以我们验证：第一个 model_done 是 bear，第二个是 bull
        assert model_dones[0]["role_id"] == "bear"
        assert model_dones[1]["role_id"] == "bull"
    
    def test_stage_transitions(self, full_session):
        """验证所有 stage 状态流转正确"""
        s = full_session
        s._client.call = MagicMock(return_value=MagicMock(content="x"))
        
        stages = []
        
        stages.append(("init", s.stage))
        s.generate()
        stages.append(("after_generate", s.stage))
        s.critique()
        stages.append(("after_critique", s.stage))
        s.revise()
        stages.append(("after_revise", s.stage))
        
        assert stages == [
            ("init", Stage.INIT),
            ("after_generate", Stage.PAUSE_AFTER_GENERATE),
            ("after_critique", Stage.PAUSE_AFTER_CRITIQUE),
            ("after_revise", Stage.PAUSE_AFTER_REVISE),
        ]
    
    def test_step_stream_init_to_generate(self, full_session):
        """step_stream 从 INIT 开始时应执行 generate"""
        s = full_session
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = f"stream_{model_id}"
            yield ("t1", text)
            yield ("t2", text)
        
        s._client.call_stream = mock_call_stream
        
        events = list(s.step_stream())
        
        # 验证 stage 变为 PAUSE_AFTER_GENERATE
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        assert s.current_round == 1
        
        # 验证所有事件都有 step_type="generate"
        for e in events:
            assert e.get("step_type") == "generate", f"事件缺少 step_type='generate': {e}"
    
    def test_step_stream_full_flow(self, full_session):
        """step_stream 完整流程：generate → critique → revise → 下一轮 critique"""
        s = full_session
        call_log = []
        
        def mock_call(*, model_id, system_prompt, user_prompt):
            call_log.append(("call", model_id))
            return MagicMock(content=f"output_{model_id}")
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_log.append(("stream", model_id))
            text = f"stream_{model_id}"
            yield ("t", text)
        
        s._client.call = mock_call
        s._client.call_stream = mock_call_stream
        
        # Step 1: INIT → generate
        events = list(s.step_stream())
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        assert s.current_round == 1
        for e in events:
            assert e.get("step_type") == "generate"
        
        # Step 2: PAUSE_AFTER_GENERATE → critique
        events = list(s.step_stream())
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert s.current_round == 1
        for e in events:
            assert e.get("step_type") == "critique"
        
        # Step 3: PAUSE_AFTER_CRITIQUE → revise
        events = list(s.step_stream())
        assert s.stage == Stage.PAUSE_AFTER_REVISE
        assert s.current_round == 1
        for e in events:
            assert e.get("step_type") == "revise"
        
        # Step 4: PAUSE_AFTER_REVISE → 下一轮 critique (round += 1)
        events = list(s.step_stream())
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert s.current_round == 2
        for e in events:
            assert e.get("step_type") == "critique"
        
        # 验证调用顺序：generate/critique/revise 都使用 call_stream
        # generate: 2 个模型
        # critique: deepseek_v4 (bear)
        # revise: kimi_k2.6 (bull)
        # next critique: deepseek_v4 (bear)
        stream_calls = [c for c in call_log if c[0] == "stream"]
        expected_stream = [
            "kimi_k2.6", "deepseek_v4",    # generate
            "deepseek_v4",                  # critique
            "kimi_k2.6",                    # revise
            "deepseek_v4",                  # next critique
        ]
        assert [c[1] for c in stream_calls] == expected_stream
    
    def test_step_sync_version(self, full_session):
        """step() 同步版应正确推进状态并返回 ui_state"""
        s = full_session
        
        s._client.call = MagicMock(return_value=MagicMock(content="output"))
        
        # 初始状态
        assert s.stage == Stage.INIT
        
        # step() 应执行 generate
        ui = s.step()
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        assert s.current_round == 1
        assert "stage_info" in ui
        
        # step() 应执行 critique
        ui = s.step()
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        
        # step() 应执行 revise
        ui = s.step()
        assert s.stage == Stage.PAUSE_AFTER_REVISE
        
        # step() 应进入下一轮 critique
        ui = s.step()
        assert s.stage == Stage.PAUSE_AFTER_CRITIQUE
        assert s.current_round == 2
    
    def test_step_stream_never_regenerates(self, full_session):
        """核心测试：step_stream 在多轮循环时绝不回到 generate"""
        s = full_session
        call_log = []
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_log.append(("stream", model_id))
            text = f"stream_{model_id}"
            yield ("t", text)
        
        def mock_call_parallel(calls):
            call_log.append(("call_parallel", [c["model_id"] for c in calls]))
            return {
                c["call_id"]: MagicMock(content=f"output_{c['model_id']}")
                for c in calls
            }
        
        s._client.call_stream = mock_call_stream
        s._client.call_parallel = mock_call_parallel
        
        # 第一轮：用 generate() 初始化（模拟正常入口）
        s.generate()
        assert s.stage == Stage.PAUSE_AFTER_GENERATE
        
        # 用 step_stream 走完第一轮
        list(s.step_stream())  # critique
        list(s.step_stream())  # revise
        assert s.stage == Stage.PAUSE_AFTER_REVISE
        
        calls_after_generate = list(call_log)
        
        # 第二轮：step_stream 从 PAUSE_AFTER_REVISE 进入
        list(s.step_stream())
        
        second_round_calls = call_log[len(calls_after_generate):]
        parallel_calls = [c for c in second_round_calls if c[0] == "call_parallel"]
        assert len(parallel_calls) == 0, (
            f"第二轮不应调用 generate（call_parallel），实际调用了 {parallel_calls}"
        )
        
        # 第二轮只应调用 deepseek_v4（bear critique）通过 call_stream
        round2_stream_calls = [c[1] for c in second_round_calls if c[0] == "stream"]
        assert round2_stream_calls == ["deepseek_v4"], (
            f"第二轮应只调用 bear critique（deepseek_v4），实际调用了 {round2_stream_calls}"
        )
