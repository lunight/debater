"""测试 memory.py — MemoryManager"""

import pytest
from unittest.mock import patch, MagicMock

from debater.memory import MemoryManager


class MockSession:
    """模拟 DebateSession"""
    def __init__(self):
        self.debate_id = "test-debate"
        self.topic = "Test Topic"
        self.scenario_type = "question_analysis"
        self.question = "新能源行业前景如何？"
        self.current_round = 5
        self._models = ["kimi_k2.6", "deepseek_v4"]
        self.role_models = {"bull": "kimi_k2.6", "bear": "deepseek_v4"}
        self.answers = {"bull": "乐观", "bear": "谨慎"}
        self.confidences = {"bull": 0.8, "bear": 0.6}
        self.critiques_history = {
            1: {"bull": {"bear": "数据不够"}},
        }


@pytest.fixture
def mm(temp_dir):
    """创建 MemoryManager，使用临时目录"""
    with patch("debater.memory.LLMClient") as MockLLM, \
         patch("debater.memory.get_config") as mock_cfg:
        
        mock_client = MagicMock()
        MockLLM.return_value = mock_client
        
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "kimi"
        mock_config.get_aggregator_model.return_value = mock_model
        mock_cfg.return_value = mock_config
        
        # 使用临时目录覆盖默认路径
        mm = MemoryManager(debate_id="test-debate", topic="Test Topic")
        mm.root = temp_dir / "memories"
        mm.root.mkdir(parents=True, exist_ok=True)
        (mm.root / "stage_memory").mkdir(exist_ok=True)
        (mm.root / "obsidian").mkdir(exist_ok=True)
        mm.working_file = mm.root / "working_memory.md"
        mm.facts_file = mm.root / "facts.json"
        mm._init_working_memory()
        mm._save_facts({"facts": [], "version": 1})
        
        yield mm


class TestMemoryManagerInit:
    def test_init_creates_directories(self, mm):
        """初始化应创建必要的目录"""
        assert mm.root.exists()
        assert (mm.root / "stage_memory").exists()
        assert (mm.root / "obsidian").exists()

    def test_init_creates_working_memory(self, mm):
        """初始化应创建 working memory 文件"""
        assert mm.working_file.exists()
        content = mm.working_file.read_text()
        assert "test-debate" in content
        assert "Test Topic" in content
        assert "Working Memory" in content

    def test_init_creates_facts_file(self, mm):
        """初始化应创建 facts.json"""
        assert mm.facts_file.exists()
        data = mm._load_facts()
        assert data["facts"] == []
        assert data["version"] == 1


class TestFactsRegistry:
    """测试 Facts Registry 独立维护"""
    
    def test_add_fact(self, mm):
        """应能添加事实"""
        assert mm.add_fact("该公司2024Q3营收412.7亿", source_round=1, confidence=0.9) is True
        facts = mm.get_all_facts()
        assert len(facts) == 1
        assert facts[0]["text"] == "该公司2024Q3营收412.7亿"
        assert facts[0]["source_round"] == 1
        assert facts[0]["confidence"] == 0.9
        assert facts[0]["stale"] is False

    def test_add_fact_dedup_exact(self, mm):
        """完全相同的事实不应重复添加"""
        mm.add_fact("营收412.7亿", source_round=1)
        result = mm.add_fact("营收412.7亿", source_round=2)
        assert result is False
        assert len(mm.get_all_facts()) == 1

    def test_add_fact_dedup_contained(self, mm):
        """包含关系的事实不应重复添加"""
        mm.add_fact("该公司2024Q3营收412.7亿元", source_round=1)
        result = mm.add_fact("营收412.7亿元", source_round=2)
        assert result is False

    def test_add_fact_empty_rejected(self, mm):
        """空事实或太短的事实应被拒绝"""
        assert mm.add_fact("", source_round=1) is False
        assert mm.add_fact("短", source_round=1) is False

    def test_mark_fact_stale_by_index(self, mm):
        """应能通过索引标记事实为失效"""
        mm.add_fact("事实A", source_round=1)
        mm.add_fact("事实B", source_round=2)
        mm.mark_fact_stale(0, reason="数据更新")
        
        facts = mm.get_all_facts(include_stale=False)
        assert len(facts) == 1
        assert facts[0]["text"] == "事实B"
        
        all_facts = mm.get_all_facts(include_stale=True)
        assert all_facts[0]["stale"] is True
        assert all_facts[0]["stale_reason"] == "数据更新"

    def test_mark_fact_stale_by_text(self, mm):
        """应能通过文本匹配标记事实为失效"""
        mm.add_fact("毛利率19.8%", source_round=1)
        mm.mark_fact_stale("毛利率", reason="Q4已更新")
        
        facts = mm.get_all_facts()
        assert len(facts) == 0

    def test_get_facts_context_format(self, mm):
        """get_facts_context 应返回 Markdown 格式"""
        mm.add_fact("营收412.7亿", source_round=1)
        mm.add_fact("毛利率19.8%", source_round=2, confidence=0.85)
        
        ctx = mm.get_facts_context()
        assert "## 已验证核心事实" in ctx
        assert "营收412.7亿" in ctx
        assert "毛利率19.8%" in ctx
        assert "置信度85%" in ctx
        assert "[R1]" in ctx
        assert "[R2]" in ctx

    def test_get_facts_context_excludes_stale(self, mm):
        """默认不应包含已失效的事实"""
        mm.add_fact("事实A", source_round=1)
        mm.add_fact("事实B", source_round=2)
        mm.mark_fact_stale(0)
        
        ctx = mm.get_facts_context()
        assert "事实A" not in ctx
        assert "事实B" in ctx

    def test_get_facts_context_respects_max_items(self, mm):
        """应限制返回的事实数量"""
        for i in range(35):
            mm.add_fact(f"事实{i}", source_round=1)
        ctx = mm.get_facts_context(max_items=5)
        # 只保留最近 5 条
        assert "事实30" in ctx
        assert "事实0" not in ctx

    def test_get_facts_context_empty(self, mm):
        """无事实时应返回空字符串"""
        assert mm.get_facts_context() == ""


class TestGetContextForPrompt:
    def test_returns_empty_when_no_file(self, mm):
        """文件不存在时应返回空字符串"""
        mm.working_file.unlink()
        assert mm.get_context_for_prompt() == ""

    def test_returns_empty_for_initial_template(self, mm):
        """初始模板未填充时应返回空字符串"""
        ctx = mm.get_context_for_prompt()
        assert ctx == ""

    def test_returns_content_after_update(self, mm):
        """更新后的内容应返回实质内容"""
        mm.working_file.write_text("""---
debate_id: x
topic: y
created: 2024-01-01
rounds_covered: 5
---

# Updated Content

Some real analysis here.
""")
        ctx = mm.get_context_for_prompt()
        assert "Updated Content" in ctx
        assert "Some real analysis" in ctx
        # frontmatter 应被移除
        assert "debate_id:" not in ctx

    def test_includes_facts_and_working_memory(self, mm):
        """应同时包含 facts 和 working memory"""
        mm.add_fact("营收412.7亿", source_round=1)
        mm.working_file.write_text("""---
debate_id: x
topic: y
---

# Analysis

Key insight here.
""")
        ctx = mm.get_context_for_prompt()
        assert "已验证核心事实" in ctx
        assert "营收412.7亿" in ctx
        assert "Key insight" in ctx

    def test_facts_come_before_working_memory(self, mm):
        """facts 应排在 working memory 前面"""
        mm.add_fact("事实A", source_round=1)
        mm.working_file.write_text("""---
debate_id: x
---

# Analysis

Insight.
""")
        ctx = mm.get_context_for_prompt()
        facts_pos = ctx.index("已验证核心事实")
        wm_pos = ctx.index("辩论演进摘要")
        assert facts_pos < wm_pos


class TestShouldUpdate:
    def test_should_update_round_zero(self, mm):
        """第0轮不应更新"""
        assert mm.should_update(0) is False

    def test_should_update_round_two(self, mm):
        """第2轮应提前激活"""
        assert mm.should_update(2) is True

    def test_should_update_every_five(self, mm):
        """每5轮应更新"""
        assert mm.should_update(5) is True
        assert mm.should_update(10) is True
        assert mm.should_update(15) is True

    def test_should_update_other_rounds(self, mm):
        """非更新轮次不应更新"""
        assert mm.should_update(1) is False
        assert mm.should_update(3) is False
        assert mm.should_update(4) is False
        assert mm.should_update(6) is False


class TestShouldSaveStage:
    def test_should_save_stage_round_zero(self, mm):
        """第0轮不应落地"""
        assert mm.should_save_stage(0) is False

    def test_should_save_stage_every_ten(self, mm):
        """每10轮应落地"""
        assert mm.should_save_stage(10) is True
        assert mm.should_save_stage(20) is True

    def test_should_save_stage_other_rounds(self, mm):
        """非落地轮次不应落地"""
        assert mm.should_save_stage(5) is False
        assert mm.should_save_stage(7) is False
        assert mm.should_save_stage(15) is False


class TestSaveStageMemory:
    def test_saves_snapshot(self, mm):
        """应保存工作记忆的完整快照"""
        mm.working_file.write_text("test content")
        mm.save_stage_memory(round_num=10)
        
        stage_file = mm.root / "stage_memory" / "stage_10.md"
        assert stage_file.exists()
        assert stage_file.read_text() == "test content"

    def test_facts_not_included_in_skeleton(self, mm):
        """stage 压缩时 facts 不应被包含在骨架中"""
        mm.add_fact("营收412.7亿", source_round=1)
        mm.working_file.write_text("some analysis content")
        
        # mock LLM 调用，让它返回骨架内容
        with patch.object(mm.client, "call") as mock_call:
            mock_call.return_value = MagicMock(content="# Skeleton\n\nJust framework.")
            mm.save_stage_memory(round_num=10)
        
        # 验证压缩 prompt 中明确提到 facts 已单独维护
        prompt = mock_call.call_args[1]["user_prompt"]
        assert "核心事实已单独维护" in prompt
        assert "facts" in prompt or "事实" in prompt


class TestReadWorkingMemory:
    def test_reads_content(self, mm):
        """应正确读取 working memory 内容"""
        mm.working_file.write_text("custom content")
        assert mm.read_working_memory() == "custom content"

    def test_returns_empty_when_missing(self, mm):
        """文件不存在时应返回空字符串"""
        mm.working_file.unlink()
        assert mm.read_working_memory() == ""


class TestExtractTags:
    def test_scenario_tag(self, mm):
        """应从场景类型提取标签"""
        session = MockSession()
        session.scenario_type = "financial_interpretation"
        tags = mm._extract_tags(session)
        assert "财务分析" in tags

    def test_industry_tags_from_question(self, mm):
        """应从问题中提取行业标签"""
        session = MockSession()
        session.question = "新能源电池技术发展如何？"
        tags = mm._extract_tags(session)
        assert "新能源" in tags

    def test_multiple_industry_tags(self, mm):
        """应能提取多个行业标签"""
        session = MockSession()
        session.question = "新能源和金融行业的交叉机会？"
        tags = mm._extract_tags(session)
        assert "新能源" in tags
        assert "金融" in tags


class TestGenerateMidObsidian:
    def test_generates_file(self, mm):
        """应生成中途总结文件"""
        session = MockSession()
        path = mm.generate_mid_obsidian(session, current_round=3)
        
        assert path != ""
        output_file = mm.root / "obsidian" / path.split("/")[-1]
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "辩论进行中总结" in content
        assert "进行中" in content
        assert "第 3 轮" in content
        assert "in_progress" in content

    def test_includes_stance_table(self, mm):
        """应包含各模型当前立场表"""
        session = MockSession()
        path = mm.generate_mid_obsidian(session, current_round=3)
        
        output_file = mm.root / "obsidian" / path.split("/")[-1]
        content = output_file.read_text()
        assert "当前立场" in content
        assert "乐观" in content
        assert "80%" in content

    def test_includes_critique_history(self, mm):
        """应包含评审历史"""
        session = MockSession()
        path = mm.generate_mid_obsidian(session, current_round=3)
        
        output_file = mm.root / "obsidian" / path.split("/")[-1]
        content = output_file.read_text()
        assert "评审历史" in content
        assert "数据不够" in content

    def test_includes_facts_section(self, mm):
        """应包含核心事实章节"""
        mm.add_fact("营收412.7亿", source_round=1)
        session = MockSession()
        path = mm.generate_mid_obsidian(session, current_round=3)
        
        output_file = mm.root / "obsidian" / path.split("/")[-1]
        content = output_file.read_text()
        assert "核心事实" in content
        assert "营收412.7亿" in content
