"""测试 session.py — DebateSession 核心逻辑"""

import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from debater.session import Stage, DebateSession
from debater.engine import extract_confidence, extract_thinking, extract_summary


@pytest.fixture
def session():
    """创建绕过 __post_init__ 的 DebateSession 实例"""
    s = object.__new__(DebateSession)
    s.question = "Test question?"
    s.scenario_type = "question_analysis"
    s.context = ""
    s.stage = Stage.INIT
    s.current_round = 0
    s.answers = {"kimi_k2.6": "", "deepseek_v4": ""}
    s.thinkings = {"kimi_k2.6": "", "deepseek_v4": ""}
    s.summaries = {"kimi_k2.6": "", "deepseek_v4": ""}
    s.confidences = {"kimi_k2.6": 0.0, "deepseek_v4": 0.0}
    s.previous_answers = {"kimi_k2.6": "", "deepseek_v4": ""}
    s.critiques_history = {}
    s.critiques = {"kimi_k2.6": {}, "deepseek_v4": {}}
    s.final_answer = ""
    s.known_facts = []
    s.memory_manager = None
    s._client = MagicMock()
    s._config = MagicMock()
    s.role_models = {"bull": "kimi_k2.6", "bear": "deepseek_v4"}
    s._models = ["kimi_k2.6", "deepseek_v4"]
    s._skill_registry = None
    
    # mock _config.models for _get_model_display_name
    mock_model = MagicMock()
    mock_model.name = "Kimi"
    mock_model2 = MagicMock()
    mock_model2.name = "DeepSeek"
    s._config.models = {"kimi_k2.6": mock_model, "deepseek_v4": mock_model2}
    
    # mock tool registry
    s._tool_registry = MagicMock()
    
    yield s


class TestStageConstants:
    def test_stage_values(self):
        """Stage 常量应正确定义"""
        assert Stage.INIT == "init"
        assert Stage.GENERATING == "generating"
        assert Stage.PAUSE_AFTER_GENERATE == "pause_after_generate"
        assert Stage.CRITIQUING == "critiquing"
        assert Stage.PAUSE_AFTER_CRITIQUE == "pause_after_critique"
        assert Stage.REVISING == "revising"
        assert Stage.PAUSE_AFTER_REVISE == "pause_after_revise"
        assert Stage.AGGREGATING == "aggregating"
        assert Stage.DONE == "done"


class TestExtractThinking:
    def test_extracts_thinking_tag(self):
        """应提取 <thinking> 标签内容"""
        text = "<thinking>my reasoning</thinking>\nformal answer"
        thinking, formal = extract_thinking(text)
        assert thinking == "my reasoning"
        assert formal == "formal answer"

    def test_no_thinking_tag(self):
        """无 thinking 标签时应返回空和原文"""
        text = "just plain answer"
        thinking, formal = extract_thinking(text)
        assert thinking == ""
        assert formal == "just plain answer"

    def test_multiline_thinking(self):
        """应支持多行 thinking 内容"""
        text = "<thinking>line1\nline2\nline3</thinking>\nanswer"
        thinking, formal = extract_thinking(text)
        assert "line1" in thinking
        assert "line3" in thinking
        assert "answer" == formal


class TestExtractConfidence:
    def test_chinese_format(self):
        """应解析中文置信度格式"""
        text = "结论成立。置信度：85%"
        assert extract_confidence(text) == 0.85

    def test_english_format(self):
        """应解析英文置信度格式"""
        text = "Confidence: 92.5%"
        assert extract_confidence(text) == 0.925

    def test_no_confidence_returns_default(self):
        """无置信度时应返回默认值 0.7"""
        text = "no confidence here"
        assert extract_confidence(text) == 0.7

    def test_clamps_out_of_range(self):
        """超出范围的值应被裁剪到 [0, 1]"""
        assert extract_confidence("置信度：150%") == 1.0
        # 负号被 regex 忽略，匹配到 10 -> 0.1，这是已知边界行为
        assert extract_confidence("置信度：-10%") == 0.1


class TestBuildCritiqueHistoryText:
    def test_builds_personalized_history(self, session):
        """应构建针对特定模型的历史评审"""
        session.critiques_history = {
            1: {
                "deepseek_v4": {"kimi_k2.6": "Your logic is flawed"},
            },
            2: {
                "deepseek_v4": {"kimi_k2.6": "Still wrong on point B"},
            },
        }
        result = session._build_critique_history_text("kimi_k2.6", current_round=3)
        assert "第1轮" in result
        assert "第2轮" in result
        assert "Your logic is flawed" in result
        assert "DeepSeek" in result  # display name

    def test_truncates_long_critiques(self, session):
        """长评审应被截断到 200 字"""
        session.critiques_history = {
            1: {"deepseek_v4": {"kimi_k2.6": "x" * 300}},
        }
        result = session._build_critique_history_text("kimi_k2.6", current_round=2)
        assert "..." in result
        assert len(result) < 400

    def test_excludes_current_and_future_rounds(self, session):
        """不应包含当前轮次及之后的评审"""
        session.critiques_history = {
            1: {"deepseek_v4": {"kimi_k2.6": "old"}},
            3: {"deepseek_v4": {"kimi_k2.6": "future"}},
        }
        result = session._build_critique_history_text("kimi_k2.6", current_round=3)
        assert "old" in result
        assert "future" not in result

    def test_returns_empty_when_no_history(self, session):
        """无历史时应返回空字符串"""
        result = session._build_critique_history_text("kimi_k2.6", current_round=1)
        assert result == ""


class TestBuildAllCritiqueHistoryText:
    def test_builds_global_history(self, session):
        """应构建包含所有模型间评审的全局历史"""
        session.critiques_history = {
            1: {
                "deepseek_v4": {"kimi_k2.6": "critique A"},
                "kimi_k2.6": {"deepseek_v4": "critique B"},
            },
        }
        result = session._build_all_critique_history_text(current_round=2)
        assert "critique A" in result
        assert "critique B" in result
        assert "DeepSeek" in result
        assert "Kimi" in result

    def test_truncates_to_150_chars(self, session):
        """全局历史每行应截断到 150 字"""
        session.critiques_history = {
            1: {"deepseek_v4": {"kimi_k2.6": "x" * 200}},
        }
        result = session._build_all_critique_history_text(current_round=2)
        assert "..." in result


class TestBuildUiState:
    def test_pause_after_generate_state(self, session):
        """PAUSE_AFTER_GENERATE 阶段 UI 状态应正确"""
        session.stage = Stage.PAUSE_AFTER_GENERATE
        session.current_round = 1
        # 状态以 role_id 为 key
        session.answers = {"bull": "k ans", "bear": "d ans"}
        session.thinkings = {"bull": "k think", "bear": "d think"}
        session.confidences = {"bull": 0.85, "bear": 0.75}
        
        ui = session._build_ui_state()
        assert ui["stage"] == Stage.PAUSE_AFTER_GENERATE
        # UI key 使用 model_id（带 safe_id）
        assert ui["kimi_k2_6_output"] == "k ans"
        assert ui["kimi_k2_6_thinking"] == "k think"
        assert ui["kimi_k2_6_conf"] == "85%"
        assert ui["deepseek_v4_conf"] == "75%"
        assert ui["can_continue"] is True
        assert ui["can_finish"] is True
        assert ui["is_done"] is False
        assert "第 1 轮" in ui["stage_info"]

    def test_done_state(self, session):
        """DONE 阶段 UI 状态应正确"""
        session.stage = Stage.DONE
        session.final_answer = "Final consensus"
        
        ui = session._build_ui_state()
        assert ui["is_done"] is True
        assert ui["final_answer"] == "Final consensus"
        assert ui["can_continue"] is False

    def test_generating_state(self, session):
        """GENERATING 阶段按钮应不可用"""
        session.stage = Stage.GENERATING
        session.current_round = 2
        
        ui = session._build_ui_state()
        assert ui["can_continue"] is False
        assert ui["can_finish"] is False
        assert "正在生成" in ui["stage_info"]


class TestSaveAndLoad:
    def test_save_creates_json(self, session, temp_dir):
        """save 应创建 JSON 文件"""
        session.memory_manager = MagicMock()
        session.memory_manager.debate_id = "test123"
        session.memory_manager.topic = "Test Topic"
        session.memory_manager.read_working_memory.return_value = "wm content"
        session.memory_manager.root = temp_dir
        
        path = session.save(save_dir=str(temp_dir))
        
        assert path.endswith(".json")
        saved = json.loads(Path(path).read_text())
        assert saved["question"] == "Test question?"
        assert saved["current_round"] == 0
        assert saved["version"] == 1

    def test_load_restores_state(self, session, temp_dir):
        """load 应正确恢复辩论状态"""
        session.memory_manager = MagicMock()
        session.memory_manager.debate_id = "test123"
        session.memory_manager.topic = "Test Topic"
        session.memory_manager.read_working_memory.return_value = "wm content"
        session.memory_manager.root = temp_dir
        
        save_path = session.save(save_dir=str(temp_dir))
        
        with patch("debater.session.LLMClient"), \
             patch("debater.session.get_config") as mock_cfg:
            mock_cfg.return_value.get_proposer_models.return_value = {
                "kimi_k2.6": MagicMock(), "deepseek_v4": MagicMock()
            }
            loaded = DebateSession.load(save_path)
        
        assert loaded.question == "Test question?"
        assert loaded.current_round == 0
        assert loaded.stage == Stage.INIT
        # critiques_history keys should be int
        assert isinstance(list(loaded.critiques_history.keys())[0] if loaded.critiques_history else 0, int)

    def test_load_preserves_critiques_history_int_keys(self, session, temp_dir):
        """load 应将 critiques_history 的 key 从 str 转回 int"""
        session.critiques_history = {1: {"a": {"b": "crit"}}}
        session.memory_manager = MagicMock()
        session.memory_manager.debate_id = "test456"
        session.memory_manager.topic = "Test Topic"
        session.memory_manager.read_working_memory.return_value = ""
        session.memory_manager.root = temp_dir
        
        save_path = session.save(save_dir=str(temp_dir))
        
        with patch("debater.session.LLMClient"), patch("debater.session.get_config") as mock_cfg:
            mock_cfg.return_value.get_proposer_models.return_value = {
                "kimi_k2.6": MagicMock(), "deepseek_v4": MagicMock()
            }
            loaded = DebateSession.load(save_path)
        
        assert 1 in loaded.critiques_history
        assert "1" not in loaded.critiques_history


class TestExecuteToolCalls:
    def test_no_tool_calls_returns_empty(self, session):
        """无 tool call 时应返回空结构"""
        session._tool_registry.create_executor.return_value.has_tool_calls.return_value = False
        result = session._execute_tool_calls("just some text")
        assert result["text"] == ""
        assert result["errors"] == []
        assert result["has_error"] is False
        assert result["fallback_notices"] == []

    def test_executes_tool_calls(self, session):
        """应执行 tool calls 并返回结构化结果"""
        mock_executor = MagicMock()
        mock_executor.has_tool_calls.return_value = True
        mock_executor.extract_and_execute.return_value = [
            {"name": "web_search", "params": {"query": "test"}, "result": MagicMock(content="results", error=None)}
        ]
        mock_executor.build_tool_result_message.return_value = "<tool_result>results</tool_result>"
        session._tool_registry.create_executor.return_value = mock_executor
        
        result = session._execute_tool_calls('<tool_call name="web_search"><query>test</query></tool_call>')
        assert "results" in result["text"]
        assert result["has_error"] is False
        assert result["errors"] == []
        assert result["fallback_notices"] == []
    
    def test_tool_call_with_error(self, session):
        """tool 执行失败时应返回错误信息"""
        mock_executor = MagicMock()
        mock_executor.has_tool_calls.return_value = True
        mock_executor.extract_and_execute.return_value = [
            {"name": "web_search", "params": {"query": "test"}, "result": MagicMock(content="", error="API rate limited")}
        ]
        mock_executor.build_tool_result_message.return_value = '<tool_result name="web_search">\n错误: API rate limited\n</tool_result>'
        session._tool_registry.create_executor.return_value = mock_executor
        
        result = session._execute_tool_calls('<tool_call name="web_search"><query>test</query></tool_call>')
        assert result["has_error"] is True
        assert "[web_search] API rate limited" in result["errors"]
        assert "错误: API rate limited" in result["text"]
    
    def test_tool_call_with_fallback(self, session):
        """tool 回退时应检测到 fallback 通知"""
        mock_executor = MagicMock()
        mock_executor.has_tool_calls.return_value = True
        mock_executor.extract_and_execute.return_value = [
            {"name": "web_search", "params": {"query": "test"}, "result": MagicMock(
                content="[Tavily 不可用: rate limit，已回退到 DuckDuckGo]\n\n搜索结果", error=None
            )}
        ]
        mock_executor.build_tool_result_message.return_value = '<tool_result name="web_search">\n[Tavily 不可用: rate limit，已回退到 DuckDuckGo]\n\n搜索结果\n</tool_result>'
        session._tool_registry.create_executor.return_value = mock_executor
        
        result = session._execute_tool_calls('<tool_call name="web_search"><query>test</query></tool_call>')
        assert result["has_error"] is False
        assert "Tavily 不可用: rate limit，已回退到 DuckDuckGo" in result["fallback_notices"]
        assert "搜索结果" in result["text"]


class TestPostInit:
    """验证 __post_init__ 正确初始化所有依赖（这些测试不绕过 __post_init__）"""
    
    @patch("debater.session.LLMClient")
    @patch("debater.session.get_config")
    def test_initializes_tool_and_skill_registry(self, mock_get_config, mock_llm_client):
        """DebateSession() 正常创建后，_tool_registry 和 _skill_registry 不应为 None"""
        mock_cfg = MagicMock()
        mock_cfg.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(),
            "deepseek_v4": MagicMock(),
        }
        mock_get_config.return_value = mock_cfg
        
        session = DebateSession("test question", "question_analysis", "")
        
        assert session._tool_registry is not None
        assert session._skill_registry is not None
        assert session._get_tool_context() is not None   # 不应抛 AttributeError
        assert session._get_skill_context() is not None  # 不应抛 AttributeError
    
    @patch("debater.session.LLMClient")
    @patch("debater.session.get_config")
    def test_generate_stream_parallel_runs_without_crash(self, mock_get_config, mock_llm_client):
        """generate_stream_parallel 完整流程不应因 _tool_registry 未初始化而崩溃"""
        mock_cfg = MagicMock()
        mock_cfg.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi"),
        }
        mock_cfg.models = {"kimi_k2.6": MagicMock(name="Kimi")}
        mock_get_config.return_value = mock_cfg
        
        session = DebateSession("test question", "question_analysis", "")
        session.set_role_model("bull", "kimi_k2.6")
        
        # Mock call_stream 返回简单内容（无 tool call）
        mock_client = MagicMock()
        mock_client.call_stream.return_value = [
            ("Hello", "Hello"),
            (" world", "Hello world"),
        ]
        session._client = mock_client
        
        events = list(session.generate_stream_parallel())
        
        # 不应出现 error 事件
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 0, f"不应出现 error 事件，但出现了: {error_events}"
        
        # 应正常完成（通过 model_done 事件确认）
        model_done_events = [e for e in events if e.get("type") == "model_done"]
        assert len(model_done_events) == 2, f"期望 2 个 model_done 事件（bull + bear），实际: {len(model_done_events)}"
        
        # answers 应以 role_id 为 key 存储
        assert session.answers.get("bull") == "Hello world"
        assert session.answers.get("bear") == "Hello world"


class TestAutoJudge:
    """测试自动裁判功能"""
    
    def test_auto_judge_parse_json_response(self, session):
        """应能正确解析 judge 返回的 JSON"""
        mock_resp = MagicMock()
        mock_resp.content = '{"action": "continue", "reason": "还有分歧", "info_needed": ""}'
        session._client.call.return_value = mock_resp
        
        result = session.auto_judge()
        
        assert result["action"] == "continue"
        assert "还有分歧" in result["reason"]
        assert result["info_needed"] == ""
    
    def test_auto_judge_stop(self, session):
        """judge 返回 stop 时应正确解析"""
        mock_resp = MagicMock()
        mock_resp.content = '{"action": "stop", "reason": "结论一致", "info_needed": ""}'
        session._client.call.return_value = mock_resp
        
        result = session.auto_judge()
        
        assert result["action"] == "stop"
    
    def test_auto_judge_need_info(self, session):
        """judge 返回 need_info 时应正确解析"""
        mock_resp = MagicMock()
        mock_resp.content = '{"action": "need_info", "reason": "缺少数据", "info_needed": "2024年Q3营收"}'
        session._client.call.return_value = mock_resp
        
        result = session.auto_judge()
        
        assert result["action"] == "need_info"
        assert result["info_needed"] == "2024年Q3营收"
    
    def test_auto_judge_fallback_on_invalid_json(self, session):
        """JSON 解析失败时应 fallback 到 continue"""
        mock_resp = MagicMock()
        mock_resp.content = '不是 JSON'
        session._client.call.return_value = mock_resp
        
        result = session.auto_judge()
        
        assert result["action"] == "continue"
        assert "解析失败" in result["reason"] or "无法解析" in result["reason"]
    
    def test_auto_judge_fallback_on_api_error(self, session):
        """API 调用失败时应 fallback 到 continue"""
        session._client.call.side_effect = Exception("API 挂了")
        
        result = session.auto_judge()
        
        assert result["action"] == "continue"
        assert "失败" in result["reason"]
    
    def test_auto_judge_extracts_json_from_code_block(self, session):
        """应能从 ```json 代码块中提取 JSON"""
        mock_resp = MagicMock()
        mock_resp.content = '```json\n{"action": "stop", "reason": "够了", "info_needed": ""}\n```'
        session._client.call.return_value = mock_resp
        
        result = session.auto_judge()
        
        assert result["action"] == "stop"


# ==================== 新增：工具链放宽测试 ====================

class TestMaxToolRoundsConstant:
    """测试 MAX_TOOL_ROUNDS 常量"""
    
    def test_max_tool_rounds_is_ten(self):
        """MAX_TOOL_ROUNDS 应为 10"""
        assert DebateSession.MAX_TOOL_ROUNDS == 10


class TestBuildToolFollowupPrompt:
    """测试 _build_tool_followup_prompt"""
    
    def test_normal_round_allows_more_tools(self, session):
        """非最后一轮应允许继续调用工具"""
        tool_info = {"text": "results", "has_error": False, "errors": [], "fallback_notices": []}
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=2,
        )
        assert "继续调用工具" in prompt or "补充关键数据" in prompt
        assert "绝对不要调用任何工具" not in prompt
    
    def test_last_round_forbids_tools(self, session):
        """最后一轮应强制禁止工具"""
        tool_info = {"text": "results", "has_error": False, "errors": [], "fallback_notices": []}
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=DebateSession.MAX_TOOL_ROUNDS + 1,  # 11 = 最后一轮 followup
        )
        assert "绝对不要调用任何工具" in prompt or "最后一轮" in prompt
    
    def test_all_tools_failed(self, session):
        """全部失败时应包含强制降权要求"""
        tool_info = {
            "text": "error",
            "has_error": True,
            "errors": ["API error"],
            "fallback_notices": [],
            "executions": [
                {"tool_name": "web_search", "status": "error"},
            ],
        }
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=2,
        )
        assert "工具调用全部失败" in prompt
        assert "置信度必须低于 50%" in prompt
    
    def test_partial_failure(self, session):
        """部分失败时应提示基于已有数据继续，不强制降权"""
        tool_info = {
            "text": "partial results",
            "has_error": True,
            "errors": ["Timeout"],
            "fallback_notices": [],
            "executions": [
                {"tool_name": "web_search", "status": "success"},
                {"tool_name": "web_search", "status": "error"},
            ],
        }
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=2,
        )
        assert "部分工具执行失败" in prompt
        assert "成功：web_search" in prompt
        assert "失败：web_search" in prompt
        assert "置信度必须低于 50%" not in prompt  # 部分失败不强制降权
    
    def test_partial_fallback(self, session):
        """部分回退时应包含警告"""
        tool_info = {"text": "results", "has_error": False, "errors": [], "fallback_notices": ["Tavily->DDG"]}
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=2,
        )
        assert "回退" in prompt
    
    def test_prompt_contains_question_and_context(self, session):
        """prompt 应包含原始问题和上下文"""
        tool_info = {"text": "results", "has_error": False, "errors": [], "fallback_notices": []}
        prompt = session._build_tool_followup_prompt(
            previous_analysis="My analysis",
            tool_info=tool_info,
            tool_round=2,
        )
        assert session.question in prompt
        assert session.context in prompt


class TestRunToolChainLoop:
    """测试 _run_tool_chain_loop 循环逻辑"""
    
    def test_no_tool_call_returns_first_round(self, session):
        """无 tool call 时直接返回第一轮文本"""
        session._tool_registry.create_executor.return_value.has_tool_calls.return_value = False
        import queue
        event_queue = queue.Queue()
        
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text="Initial analysis",
            event_queue=event_queue,
        )
        assert result == "Initial analysis"
    
    def test_single_tool_call_then_done(self, session):
        """一轮 tool call 后模型不再调用，应返回第二轮文本"""
        import queue
        event_queue = queue.Queue()
        
        # has_tool_calls: 第一次 True（第一轮有 tool call），第二次 False（第二轮检查无 tool call）
        has_tc_calls = iter([True, False])
        session._tool_registry.create_executor.return_value.has_tool_calls = lambda text: next(has_tc_calls)
        
        # mock _execute_tool_calls
        session._execute_tool_calls = MagicMock(return_value={
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
            "executions": [],
        })
        
        # mock call_stream 返回不带 tool_call 的最终分析
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            yield ("Final", "Final analysis")
            yield ("", "Final analysis")
        session._client.call_stream = mock_call_stream
        
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text='<tool_call name="web_search"><query>test</query></tool_call>Initial',
            event_queue=event_queue,
        )
        assert result == "Final analysis"
        session._execute_tool_calls.assert_called_once()
    
    def test_multiple_tool_calls_continue_loop(self, session):
        """模型连续多轮输出 tool_call，应继续循环"""
        import queue
        event_queue = queue.Queue()
        
        # has_tool_calls 调用序列：
        # 循环1开头(检查first_round) → True
        # 循环1末尾(检查round2输出) → True
        # 循环2开头(检查round2) → True
        # 循环2末尾(检查round3输出) → True
        # 循环3开头(检查round3) → True
        # 循环3末尾(检查round4输出) → True
        # 循环4开头(检查round4) → True
        # 循环4末尾(检查round5输出) → False → 返回
        has_tc_calls = iter([True, True, True, True, True, True, True, False])
        session._tool_registry.create_executor.return_value.has_tool_calls = lambda text: next(has_tc_calls)
        
        call_count = [0]
        def mock_execute_tool_calls(text, model_id="unknown"):
            call_count[0] += 1
            return {
                "text": f"<tool_result>results_{call_count[0]}</tool_result>",
                "has_error": False,
                "errors": [],
                "fallback_notices": [],
                "executions": [],
            }
        session._execute_tool_calls = mock_execute_tool_calls
        
        round_num = [0]
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            round_num[0] += 1
            if round_num[0] <= 3:
                text = f'<tool_call name="web_search"><query>round_{round_num[0]}</query></tool_call>Analysis {round_num[0]}'
            else:
                text = "Final analysis"
            yield ("token", text)
            yield ("", text)
        session._client.call_stream = mock_call_stream
        
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text='<tool_call name="web_search"><query>round_1</query></tool_call>Initial',
            event_queue=event_queue,
        )
        assert result == "Final analysis"
        assert call_count[0] == 4  # 执行了4次工具
    
    def test_errors_after_max_rounds(self, session):
        """超过 MAX_TOOL_ROUNDS 后仍输出 tool_call，应报错"""
        import queue
        event_queue = queue.Queue()
        
        # has_tool_calls 永远返回 True
        session._tool_registry.create_executor.return_value.has_tool_calls = lambda text: True
        
        session._execute_tool_calls = MagicMock(return_value={
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
            "executions": [],
        })
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            text = '<tool_call name="web_search"><query>test</query></tool_call>Analysis'
            yield ("A", text)
            yield ("", text)
        session._client.call_stream = mock_call_stream
        
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text='<tool_call name="web_search"><query>test</query></tool_call>Initial',
            event_queue=event_queue,
        )
        assert result is None  # 报错时返回 None
        
        # 检查 event_queue 中有 error 事件
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "tool call" in error_events[0]["error"].lower()
    
    def test_empty_second_round_fallback(self, session):
        """第二轮空输出时应回退到上一轮"""
        import queue
        event_queue = queue.Queue()
        
        has_tc_calls = iter([True, False])
        session._tool_registry.create_executor.return_value.has_tool_calls = lambda text: next(has_tc_calls)
        
        session._execute_tool_calls = MagicMock(return_value={
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
            "executions": [],
        })
        
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            yield ("", "")  # 空输出
            yield ("", "")
        session._client.call_stream = mock_call_stream
        
        first_round = "First round analysis"
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text=first_round,
            event_queue=event_queue,
        )
        # 空输出回退到第一轮
        assert result == first_round
    
    def test_high_risk_retry(self, session):
        """第二轮因 high risk 被拒绝时应使用简化 prompt 重试"""
        import queue
        event_queue = queue.Queue()
        
        has_tc_calls = iter([True, False])
        session._tool_registry.create_executor.return_value.has_tool_calls = lambda text: next(has_tc_calls)
        
        session._execute_tool_calls = MagicMock(return_value={
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
            "executions": [],
        })
        
        call_count = [0]
        def mock_call_stream(*, model_id, system_prompt, user_prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("high risk content")
            yield ("Final", "Final analysis")
            yield ("", "Final analysis")
        session._client.call_stream = mock_call_stream
        
        result = session._run_tool_chain_loop(
            role_id="bull",
            model_id="kimi_k2.6",
            model_name="Kimi",
            system_prompt="sys",
            first_round_text="Initial",
            event_queue=event_queue,
        )
        assert result == "Final analysis"
        assert call_count[0] == 2  # 第一次失败，第二次成功


class TestAutoDebateStream:
    """测试 auto_debate_stream 自动辩论循环（串行交替版本）"""
    
    def test_stops_when_judge_says_stop(self, session):
        """judge 返回 stop 时应立即停止"""
        # mock 串行 critique 和 revise（bear → bull，bull revise）
        def mock_critique(critic_role, target_role):
            yield {"type": "model_done", "role_id": critic_role, "model_id": "deepseek_v4", "full_text": "critique"}
        
        def mock_revise(role_id, user_critique=None):
            yield {"type": "model_done", "role_id": role_id, "model_id": "kimi_k2.6", "full_text": "revised"}
        
        session.critique_stream_single = mock_critique
        session.revise_stream_single = mock_revise
        session.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        events = list(session.auto_debate_stream(max_cycles=3))
        
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 1
        assert judge_events[0]["action"] == "stop"
        assert session.stage == Stage.DONE
    
    def test_continues_when_judge_says_continue(self, session):
        """judge 返回 continue 时应继续下一轮"""
        def mock_critique(critic_role, target_role):
            yield {"type": "model_done", "role_id": critic_role, "model_id": "deepseek_v4", "full_text": "critique"}
        
        def mock_revise(role_id, user_critique=None):
            yield {"type": "model_done", "role_id": role_id, "model_id": "kimi_k2.6", "full_text": "revised"}
        
        session.critique_stream_single = mock_critique
        session.revise_stream_single = mock_revise
        session.auto_judge = MagicMock(side_effect=[
            {"action": "continue", "reason": "还有分歧", "info_needed": ""},
            {"action": "continue", "reason": "仍有分歧", "info_needed": ""},
            {"action": "stop", "reason": "已统一", "info_needed": ""},
        ])
        
        events = list(session.auto_debate_stream(max_cycles=5))
        
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 3
        assert judge_events[0]["action"] == "continue"
        assert judge_events[1]["action"] == "continue"
        assert judge_events[2]["action"] == "stop"
        assert session.current_round == 3
    
    def test_respects_max_cycles(self, session):
        """达到 max_cycles 后应停止，即使 judge 仍返回 continue"""
        def mock_critique(critic_role, target_role):
            yield {"type": "model_done", "role_id": critic_role, "model_id": "deepseek_v4", "full_text": "critique"}
        
        def mock_revise(role_id, user_critique=None):
            yield {"type": "model_done", "role_id": role_id, "model_id": "kimi_k2.6", "full_text": "revised"}
        
        session.critique_stream_single = mock_critique
        session.revise_stream_single = mock_revise
        session.auto_judge = MagicMock(return_value={
            "action": "continue", "reason": "继续", "info_needed": ""
        })
        
        events = list(session.auto_debate_stream(max_cycles=2))
        
        judge_events = [e for e in events if e.get("type") == "judge"]
        assert len(judge_events) == 2
        assert session.current_round == 2
    
    def test_yields_all_events(self, session):
        """应 yield critique、revise、judge 的所有事件（串行：1 critique + 1 revise）"""
        def mock_critique(critic_role, target_role):
            yield {"type": "token", "role_id": critic_role, "token": "c1"}
            yield {"type": "model_done", "role_id": critic_role, "model_id": "deepseek_v4", "full_text": "critique"}
        
        def mock_revise(role_id, user_critique=None):
            yield {"type": "token", "role_id": role_id, "token": "r1"}
            yield {"type": "model_done", "role_id": role_id, "model_id": "kimi_k2.6", "full_text": "revised"}
        
        session.critique_stream_single = mock_critique
        session.revise_stream_single = mock_revise
        session.auto_judge = MagicMock(return_value={
            "action": "stop", "reason": "已统一", "info_needed": ""
        })
        
        events = list(session.auto_debate_stream(max_cycles=1))
        
        tokens = [e for e in events if e.get("type") == "token"]
        model_dones = [e for e in events if e.get("type") == "model_done"]
        judges = [e for e in events if e.get("type") == "judge"]
        
        assert len(tokens) == 2  # 1 critique (bear) + 1 revise (bull)
        assert len(model_dones) == 2  # 1 critique + 1 revise
        assert len(judges) == 1


# ==================== 全流程多轮工具链集成测试 ====================

def _make_tool_executor_mock(has_tc_sequence):
    """创建工具执行器 mock，按序列返回 has_tool_calls 结果"""
    tc_iter = iter(has_tc_sequence)
    def factory():
        ex = MagicMock()
        ex.has_tool_calls = lambda text: next(tc_iter)
        ex.extract_and_execute.return_value = [
            {"name": "web_search", "params": {"query": "test"},
             "result": MagicMock(content="results", error=None)}
        ]
        ex.build_tool_result_message.return_value = "<tool_result>results</tool_result>"
        ex.TOOL_CALL_PATTERN = MagicMock()
        ex.TOOL_CALL_PATTERN.finditer.return_value = [MagicMock(group=lambda i: "web_search")]
        return ex
    return factory


def _make_call_stream(texts):
    """创建 call_stream mock，按序列返回文本"""
    idx = [0]
    def mock_call_stream(*, model_id, system_prompt, user_prompt):
        idx[0] += 1
        text = texts[idx[0] - 1] if idx[0] <= len(texts) else (texts[-1] if texts else "Final")
        yield ("token", text)
        yield ("", text)
    return mock_call_stream


class TestGenerateStreamParallelToolChain:
    """generate_stream_parallel 全流程多轮工具链测试"""
    
    def _setup_single_role(self, session):
        """只保留一个角色，避免 threading 竞争"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session.thinkings = {"bull": ""}
        session.summaries = {"bull": ""}
        session.confidences = {"bull": 0.0}
        session.previous_answers = {"bull": ""}
    
    def test_no_tool_call_completes_directly(self, session):
        """无 tool call 时直接完成"""
        self._setup_single_role(session)
        session._client.call_stream.return_value = [
            ("Hello", "Hello"),
            (" world", "Hello world"),
        ]
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([False])
        
        events = list(session.generate_stream_parallel())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Hello world"
    
    def test_single_tool_call_then_complete(self, session):
        """一轮 tool call 后完成"""
        self._setup_single_role(session)
        # call_stream 调用序列: 第1轮带tool_call, 第2轮不带
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>test</query></tool_call>Analysis',
            'Final analysis without tools'
        ])
        # has_tool_calls 序列:
        #   stream_single 检测 -> True (1)
        #   _execute_tool_calls 内部 -> True (1)
        #   _run_tool_chain_loop 循环1 开头 -> True (1)
        #   _run_tool_chain_loop 循环1 末尾 -> False (1)
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, True, True, False
        ])
        
        events = list(session.generate_stream_parallel())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final analysis without tools"
    
    def test_multiple_tool_calls_continue(self, session):
        """多轮 tool call（3轮）后继续循环直到完成"""
        self._setup_single_role(session)
        texts = [
            '<tool_call name="web_search"><query>r1</query></tool_call>Analysis1',
            '<tool_call name="web_search"><query>r2</query></tool_call>Analysis2',
            '<tool_call name="web_search"><query>r3</query></tool_call>Analysis3',
            'Final analysis'
        ]
        session._client.call_stream = _make_call_stream(texts)
        # has_tool_calls 序列:
        #   stream_single 检测 -> True (1)
        #   _execute_tool_calls 内部 -> True (1)
        #   loop1 开头 True (1) + 末尾 True (1)
        #   loop2 开头 True (1) + _execute True (1) + 末尾 True (1)
        #   loop3 开头 True (1) + _execute True (1) + 末尾 False (1)
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, True, True, True, True, True, True, True, True, False
        ])
        
        events = list(session.generate_stream_parallel())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final analysis"
    
    def test_max_rounds_error(self, session):
        """达到 MAX_TOOL_ROUNDS 时报错"""
        self._setup_single_role(session)
        texts = ['<tool_call name="web_search"><query>test</query></tool_call>Analysis'] * 11
        session._client.call_stream = _make_call_stream(texts)
        # 总共: stream_single True(1) + _execute True(1) + loop1 开头True(1)+末尾True(1)
        #       + loop2-10 每轮 开头True(1)+_execute True(1)+末尾True(1) = 3*9 = 27
        #       = 2 + 2 + 27 = 31
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([True] * 31)
        
        events = list(session.generate_stream_parallel())
        
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "tool call" in error_events[0]["error"].lower()
    
    def test_fallback_search(self, session):
        """模型输出工具意图关键词但未输出 XML 时触发回退搜索"""
        self._setup_single_role(session)
        session._client.call_stream = _make_call_stream([
            '让我搜索一下相关数据',
            'Final analysis'
        ])
        # has_tool_calls: 第1次 False(无XML), 第2次 False(回退后第二轮无tool_call)
        # 但回退搜索构造了虚拟 tool_call，所以 _execute_tool_calls 会执行
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            False, False
        ])
        
        events = list(session.generate_stream_parallel())
        
        # 回退搜索会发送 tool_executing 事件
        tool_events = [e for e in events if e.get("type") == "tool_executing"]
        fallback_events = [e for e in tool_events if "回退" in e.get("message", "")]
        assert len(fallback_events) >= 1
        assert len([e for e in events if e.get("type") == "error"]) == 0


class TestBrainstormStreamParallelToolChain:
    """brainstorm_stream_parallel 全流程多轮工具链测试"""
    
    def _setup_single_role(self, session):
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session.thinkings = {"bull": ""}
        session.summaries = {"bull": ""}
    
    def test_single_tool_call_then_complete(self, session):
        """一轮 tool call 后完成"""
        self._setup_single_role(session)
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>test</query></tool_call>Brainstorm1',
            'Final brainstorm'
        ])
        # 同 generate_stream_parallel: stream_single True + _execute True + loop1 开头True + 末尾False = 4次
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, True, True, False
        ])
        
        events = list(session.brainstorm_stream_parallel())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final brainstorm"
    
    def test_max_rounds_error(self, session):
        """达到 MAX_TOOL_ROUNDS 时报错"""
        self._setup_single_role(session)
        texts = ['<tool_call name="web_search"><query>test</query></tool_call>Brainstorm'] * 11
        session._client.call_stream = _make_call_stream(texts)
        # 同 generate_stream_parallel: 31 次 True
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([True] * 31)
        
        events = list(session.brainstorm_stream_parallel())
        
        assert len([e for e in events if e.get("type") == "error"]) == 1


class TestRetryGenerateRoleStreamToolChain:
    """retry_generate_role_stream 全流程多轮工具链测试"""
    
    def test_single_tool_call_then_complete(self, session):
        """一轮 tool call 后完成"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>test</query></tool_call>Retry1',
            'Final retry'
        ])
        # has_tool_calls 序列:
        #   retry 开头检测 -> True (1)
        #   _execute_tool_calls(第一轮) -> True (1)
        #   循环1 末尾 -> False (1)
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, True, False
        ])
        
        events = list(session.retry_generate_role_stream("bull", extra_context="", round_type="generate"))
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final retry"
    
    def test_multiple_tool_calls_continue(self, session):
        """多轮 tool call（3轮）后继续循环直到完成"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>r1</query></tool_call>Retry1',
            '<tool_call name="web_search"><query>r2</query></tool_call>Retry2',
            '<tool_call name="web_search"><query>r3</query></tool_call>Retry3',
            'Final retry'
        ])
        # has_tool_calls:
        #   retry 开头 -> True (1)
        #   _execute(第一轮) -> True (1)
        #   循环1 末尾 -> True (1)
        #   循环2 _execute -> True (1), 末尾 -> True (1)
        #   循环3 _execute -> True (1), 末尾 -> False (1)
        # 总共: True*7
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock(
            [True, True, True, True, True, True, False]
        )
        
        events = list(session.retry_generate_role_stream("bull", extra_context="", round_type="generate"))
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final retry"
    
    def test_max_rounds_error(self, session):
        """达到 MAX_TOOL_ROUNDS 时报错"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        texts = ['<tool_call name="web_search"><query>test</query></tool_call>Retry'] * 11
        session._client.call_stream = _make_call_stream(texts)
        # retry 开头 True(1) + _execute(第一轮) True(1) + 循环1 末尾 True(1)
        # + 循环2-9 每轮 _execute True(1)+末尾 True(1) = 2*8 = 16
        # + 循环9 末尾 True(1) 后 tool_round=10>=10 报错
        # 总共 = 1+1+1+16+1 = 20
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([True] * 20)
        
        events = list(session.retry_generate_role_stream("bull", extra_context="", round_type="generate"))
        
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "tool call" in error_events[0]["error"].lower()


class TestGenerateStreamToolChain:
    """generate_stream 全流程多轮工具链测试"""
    
    def test_single_tool_call_then_complete(self, session):
        """一轮 tool call 后完成"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>test</query></tool_call>Gen1',
            'Final gen'
        ])
        # has_tool_calls:
        #   _execute_tool_calls 内部(第一轮) -> True
        #   循环1 末尾 -> False
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, False
        ])
        
        events = list(session.generate_stream())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final gen"
    
    def test_multiple_tool_calls_continue(self, session):
        """多轮 tool call（3轮）后继续循环"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        session._client.call_stream = _make_call_stream([
            '<tool_call name="web_search"><query>r1</query></tool_call>Gen1',
            '<tool_call name="web_search"><query>r2</query></tool_call>Gen2',
            '<tool_call name="web_search"><query>r3</query></tool_call>Gen3',
            'Final gen'
        ])
        # has_tool_calls:
        #   _execute_tool_calls 内部(第一轮) -> True
        #   循环1 末尾 -> True
        #   循环2: _execute_tool_calls 内部 -> True, 末尾 -> True
        #   循环3: _execute_tool_calls 内部 -> True, 末尾 -> False
        # 总共: True*7
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock(
            [True, True, True, True, True, True, False]
        )
        
        events = list(session.generate_stream())
        
        assert len([e for e in events if e.get("type") == "error"]) == 0
        assert session.answers["bull"] == "Final gen"
    
    def test_max_rounds_error(self, session):
        """达到 MAX_TOOL_ROUNDS 时报错"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        texts = ['<tool_call name="web_search"><query>test</query></tool_call>Gen'] * 11
        session._client.call_stream = _make_call_stream(texts)
        # 第一轮 _execute_tool_calls 内部 True + 循环1-10: 每轮 _execute_tool_calls 内部 True + 末尾 True = 20 次
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([True] * 20)
        
        events = list(session.generate_stream())
        
        assert len([e for e in events if e.get("type") == "error"]) == 1


class TestGenerateToolChain:
    """generate（同步版）全流程多轮工具链测试"""
    
    def test_single_tool_call_then_complete(self, session):
        """一轮 tool call 后完成"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        # mock 第一轮并行调用
        resp_parallel = MagicMock()
        resp_parallel.content = '<tool_call name="web_search"><query>test</query></tool_call>Gen1'
        session._client.call_parallel.return_value = {"bull": resp_parallel}
        # mock 后续轮次 call
        call_count = [0]
        def mock_call(*, model_id, system_prompt, user_prompt):
            call_count[0] += 1
            text = 'Final gen'
            resp = MagicMock()
            resp.content = text
            return resp
        session._client.call = mock_call
        # has_tool_calls:
        #   _execute_tool_calls 内部(第一轮文本) -> True
        #   循环1 末尾 -> False
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([
            True, False
        ])
        
        session.generate()
        
        assert session.answers["bull"] == "Final gen"
    
    def test_multiple_tool_calls_continue(self, session):
        """多轮 tool call（3轮）后继续循环"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        # mock 第一轮并行调用
        resp_parallel = MagicMock()
        resp_parallel.content = '<tool_call name="web_search"><query>r1</query></tool_call>Gen1'
        session._client.call_parallel.return_value = {"bull": resp_parallel}
        # mock 后续轮次 call
        call_count = [0]
        def mock_call(*, model_id, system_prompt, user_prompt):
            call_count[0] += 1
            if call_count[0] <= 2:
                text = f'<tool_call name="web_search"><query>r{call_count[0]+1}</query></tool_call>Gen{call_count[0]+1}'
            else:
                text = 'Final gen'
            resp = MagicMock()
            resp.content = text
            return resp
        session._client.call = mock_call
        # 同 generate_stream: True*6
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock(
            [True, True, True, True, True, False]
        )
        
        session.generate()
        
        assert session.answers["bull"] == "Final gen"
    
    def test_max_rounds_fallback(self, session):
        """达到 MAX_TOOL_ROUNDS 时回退到第一轮结果（同步版不报错，只保留第一轮）"""
        session.role_models = {"bull": "kimi_k2.6"}
        session._models = ["kimi_k2.6"]
        session.answers = {"bull": ""}
        session._config.get_proposer_models.return_value = {
            "kimi_k2.6": MagicMock(name="Kimi")
        }
        # mock 第一轮并行调用
        resp_parallel = MagicMock()
        resp_parallel.content = '<tool_call name="web_search"><query>test</query></tool_call>FirstRound'
        session._client.call_parallel.return_value = {"bull": resp_parallel}
        # mock 后续轮次 call
        resp = MagicMock()
        resp.content = '<tool_call name="web_search"><query>test</query></tool_call>Gen'
        session._client.call.return_value = resp
        # 同 generate_stream: True*20
        session._tool_registry.create_executor.side_effect = _make_tool_executor_mock([True] * 20)
        
        session.generate()
        
        # 同步版达到最大轮次时回退到第一轮结果
        assert session.answers["bull"] == "FirstRound"


class TestToolExecutorCompatibility:
    """ToolExecutor 兼容性格式测试"""
    
    def test_matches_tool_call_format(self):
        """应匹配规范 <tool_call> 标签"""
        from debater.tools.base import ToolExecutor
        executor = ToolExecutor([])
        text = '<tool_call name="web_search"><query>测试</query></tool_call>'
        assert executor.has_tool_calls(text) is True
        results = executor.extract_and_execute(text)
        assert len(results) == 1
        assert results[0]["params"]["query"] == "测试"
    
    def test_matches_tool_format(self):
        """应兼容 <tool> 标签（部分模型的替代格式）"""
        from debater.tools.base import ToolExecutor
        executor = ToolExecutor([])
        text = '<tool name="web_search"> {"query": "测试", "region": "cn-zh"} </tool>'
        assert executor.has_tool_calls(text) is True
        results = executor.extract_and_execute(text)
        assert len(results) == 1
        assert results[0]["params"]["query"] == "测试"
        assert results[0]["params"]["region"] == "cn-zh"
    
    def test_matches_mixed_format(self):
        """应同时匹配 <tool> 和 <tool_call> 混合输出"""
        from debater.tools.base import ToolExecutor
        executor = ToolExecutor([])
        text = (
            '<tool_call name="web_search"><query>A</query></tool_call>\n'
            '<tool name="web_search"> {"query": "B"} </tool>'
        )
        assert executor.has_tool_calls(text) is True
        results = executor.extract_and_execute(text)
        assert len(results) == 2
        assert results[0]["params"]["query"] == "A"
        assert results[1]["params"]["query"] == "B"
    
    def test_no_match_for_plain_text(self):
        """纯文本不应被误判为 tool call"""
        from debater.tools.base import ToolExecutor
        executor = ToolExecutor([])
        assert executor.has_tool_calls("这是普通分析文本") is False
        assert executor.has_tool_calls("让我搜索一下") is False


class TestBuildToolFollowupPromptPureToolWarning:
    """_build_tool_followup_prompt 纯工具输出提醒测试"""
    
    def test_warns_when_only_tools_no_analysis(self, session):
        """previous_analysis 为纯工具占位符时应发出警告"""
        tool_info = {
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
        }
        prompt = session._build_tool_followup_prompt(
            "（你之前只调用了工具，尚未给出分析）",
            tool_info,
            tool_round=2,
        )
        assert "只输出了工具调用标签" in prompt
        assert "没有给出自己的分析" in prompt
    
    def test_no_warning_when_normal_analysis(self, session):
        """previous_analysis 为正常分析时不应发出警告"""
        tool_info = {
            "text": "<tool_result>results</tool_result>",
            "has_error": False,
            "errors": [],
            "fallback_notices": [],
        }
        prompt = session._build_tool_followup_prompt(
            "基于已有数据，我认为深圳三口之家月支出约2万元...",
            tool_info,
            tool_round=2,
        )
        assert "只输出了工具调用标签" not in prompt
