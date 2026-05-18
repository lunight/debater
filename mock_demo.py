"""Mock 演示：不调用真实 API，模拟完整的多模型辩论流程

用于快速验证系统逻辑和展示效果。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Dict, Any, List
from debater.state import DebateState
from debater.nodes import DebateNodes
from debater.llm_client import LLMClient, LLMResponse
from debater.graph import create_debate_graph


class MockLLMClient(LLMClient):
    """模拟 LLM 客户端，返回预设回答"""
    
    # 模拟两个模型的角色性格
    MOCK_RESPONSES = {
        "kimi_k2.6": {
            "generate": """**分析过程**：
从财务数据看，该公司营收三年持续增长（156.8→289.4→412.7亿元），增速虽然放缓但绝对值仍在提升。净利润同步增长（12.3→28.6→35.2亿元），但增速从132.5%骤降至23.1%，这个下滑幅度需要警惕。

毛利率在2022年改善至22.3%后，2023年回落至19.8%，说明行业价格战已经传导至公司层面。不过研发费用持续高增长（15.2→28.4→42.6亿元），表明公司仍在加大技术投入，这对长期竞争力是正面的。

**结论**：盈利能力整体改善但动能减弱，2023年呈现"增收不增利"的早期信号。

**置信度**：75%
（毛利率下滑和利润增速放缓让我无法给出更高置信度）

**不确定点**：2024年一季度数据未披露，无法判断趋势是否延续。""",
            
            "critique": """对 deepseek_v4 的评审：
1. **认同点**：你提到规模效应带来的成本下降是正确的，2022年毛利率提升确实印证了这一点。
2. **质疑点**：你说"盈利质量在提升"，但我注意到应收账款周转天数从2021年的68天增加到2023年的95天，回款变慢可能意味着盈利质量其实在下降，而非提升。你没有充分讨论现金流质量。
3. **补充点**：研发费用率从9.7%提升到10.3%，虽然绝对值增加，但相对营收的增长在放缓，这是否意味着研发投入强度在边际递减？
4. **对结论的影响**：你的乐观判断让我重新考虑，但现金流质量的隐忧使我维持谨慎态度。""",
            
            "revise": """**修订说明**：
收到 deepseek_v4 的评审后，我重新审视了现金流数据。确实，虽然净利润在增长，但经营现金流净额从2022年的31.2亿元下降至2023年的18.7亿元，说明利润增长并没有完全转化为现金。这是一个重要修正。

但我不完全认同对方"盈利质量提升"的判断，现金流恶化恰恰说明质量在下降。

**更新后的分析**：
1. 营收增长持续但放缓，符合行业从高速扩张转向稳态发展的规律
2. 净利润增速大幅回落（132.5%→23.1%），且经营现金流同比下降40%，盈利质量实际在恶化
3. 毛利率19.8%已回到2021年水平，价格战对利润的侵蚀已被证实
4. 研发费用绝对值继续增长，但研发费用率增速放缓，需关注后续投入强度

**更新后的结论**：盈利能力在2023年出现拐点，从"改善"转向"承压"。虽然基本面仍稳健，但趋势已变。

**更新后的置信度**：85%
（现金流数据支撑了更明确的判断）""",
        },
        
        "deepseek_v4": {
            "generate": """**分析过程**：
营收三年复合增速约62%，在新能源电池行业属于中上水平。2023年增速42.6%虽较2022年放缓，但考虑到基数变大，这个增速仍相当可观。

净利润增速从132.5%降至23.1%，看起来大幅放缓，但需要注意2022年存在原材料成本下降的额外利好（碳酸锂价格从60万/吨跌至15万/吨），这个因素在2023年已不存在。如果剔除这个一次性因素，2022年真实增速应在50-60%左右，2023年23.1%的增速就显得没那么糟糕了。

毛利率19.8%较2022年下降2.5个百分点，但相比2021年仍高1.3个百分点，说明公司的规模效应和成本控制能力在持续提升，只是被行业价格战部分抵消。

研发费用三年增长180%，在技术创新密集期的投入是合理的，且研发费用率稳定在10%左右，并未失控。

**结论**：盈利能力持续改善，2023年的数据看起来放缓，但很大程度上是因为2022年有异常利好。剔除异常因素后，真实改善趋势仍在。

**置信度**：82%
（需要2024年数据验证我的"剔除异常因素"假设）

**不确定点**：如果2024年毛利率继续下滑至18%以下，我的判断可能需要修正。""",
            
            "critique": """对 kimi_k2.6 的评审：
1. **认同点**：你指出利润增速放缓和毛利率回落是正确的观察，这些数据确实需要警惕。
2. **质疑点**：你说"增收不增利"，但净利润从28.6亿增长到35.2亿，绝对值仍在增加，只是增速放缓。"增收不增利"通常指营收增但利润降，你的表述不够准确。
3. **补充点**：你没有考虑2022年原材料成本异常下降带来的高基数效应。如果正常化2022年利润至20-22亿元，那么2023年35.2亿反而显示出很强的增长韧性。
4. **对结论的影响**：我认为你的判断偏悲观，可能高估了短期波动，低估了行业竞争格局优化后的长期盈利稳定性。""",
            
            "revise": """**修订说明**：
kimi_k2.6 指出我的"盈利质量提升"判断可能忽略了现金流恶化的问题。我重新检查了经营现金流数据，确实从31.2亿降至18.7亿，这是个重要问题。但同时我也坚持我的核心观点：2022年的高基数效应被忽视了。

关于"增收不增利"的表述，对方说得对，我的措辞不够严谨。应该改为"增收但利润增速显著放缓"。

**更新后的分析**：
1. 营收增长健康，行业地位稳固
2. 净利润增速放缓是事实，但需考虑2022年异常高基数。正常化后，2023年增速约40-50%，仍属良好
3. ⚠️ 经营现金流恶化（31.2亿→18.7亿）是新的风险信号，可能意味着应收账款增加或库存积压
4. 毛利率19.8%虽低于2022年，但高于2021年，规模效应仍在发挥作用
5. 现金流问题需要高度关注，如果2024年不改善，将推翻我"持续改善"的判断

**更新后的结论**：盈利能力基本稳定，"持续改善"的判断需附加条件——2024年现金流必须回暖。如果现金流继续恶化，则判断转为"承压"。

**更新后的置信度**：80%
（现金流不确定性降低了置信度）""",
        },
    }
    
    def __init__(self):
        # 跳过父类的 API key 检查
        self._config = None  # type: ignore
    
    def call(self, model_id: str, system_prompt: str, user_prompt: str,
             temperature=None, max_tokens=None) -> LLMResponse:
        """模拟调用"""
        import time
        import random
        
        # 根据 prompt 内容判断是哪种操作
        content = ""
        if "投资委员会主席" in system_prompt and "continue|stop|need_info" in user_prompt:
            # Judge 节点
            content = '{"action": "stop", "reason": "已达成共识（mock）", "info_needed": ""}'
        elif "投资委员会主席" in system_prompt or "最终结论" in user_prompt[:50]:
            # Aggregate 节点
            content = self._aggregate_response()
        elif "评审" in user_prompt or "critique" in user_prompt.lower():
            content = self.MOCK_RESPONSES.get(model_id, {}).get("critique", "（模拟评审）")
        elif "修订" in user_prompt or "revise" in user_prompt.lower():
            content = self.MOCK_RESPONSES.get(model_id, {}).get("revise", "（模拟修订）")
        else:
            content = self.MOCK_RESPONSES.get(model_id, {}).get("generate", "（模拟生成）")
        
        # 模拟延迟
        time.sleep(random.uniform(0.3, 0.8))
        
        return LLMResponse(
            content=content,
            model_id=model_id,
            usage={"input_tokens": 800, "output_tokens": 600},
            latency_ms=random.uniform(500, 1200),
        )
    
    def _aggregate_response(self) -> str:
        """模拟聚合结论"""
        return """**最终结论**：该公司盈利能力在2023年出现边际拐点，从"持续改善"转为"基本稳定、暗藏压力"。

**结论依据**：
1. 营收端：三年持续增长，2023年412.7亿元（+42.6%），增长质量仍属健康
2. 利润端：净利润35.2亿元（+23.1%），绝对值增长但增速大幅回落。考虑2022年原材料成本异常下降的高基数效应，真实增速应在35-40%左右
3. ⚠️ 关键风险信号：经营现金流净额从31.2亿降至18.7亿（-40%），盈利质量实际在恶化，而非改善
4. 毛利率19.8%，高于2021年但低于2022年，规模效应被价格战部分吞噬
5. 研发费用持续高投入（42.6亿），长期竞争力支撑仍在

**关键分歧点**：
- kimi_k2.6 更关注现金流恶化和毛利率回落，判断偏谨慎（"承压"）
- deepseek_v4 更关注基数效应和长期竞争力，判断偏稳健（"基本稳定"）

**风险提示**：
1. 若2024年经营现金流未回暖至25亿以上，需下调盈利预期
2. 若毛利率跌破18%，行业价格战可能已进入恶性阶段
3. 应收账款周转天数若继续恶化（当前95天），存在坏账风险

**建议后续跟踪点**：
- 2024年Q1/Q2经营现金流趋势
- 行业产能利用率变化
- 主要客户订单续约情况
- 碳酸锂等原材料价格波动"""
    
    def call_parallel(self, calls: List[Dict[str, Any]]) -> Dict[str, LLMResponse]:
        """模拟并行调用（返回 key 为 call_id，即 role_id）"""
        results = {}
        for call in calls:
            mid = call["model_id"]
            call_id = call.get("call_id", mid)
            results[call_id] = self.call(
                model_id=mid,
                system_prompt=call.get("system_prompt", ""),
                user_prompt=call["user_prompt"],
            )
        return results


def run_mock_debate():
    """运行 Mock 辩论演示"""
    
    # 用 Mock 客户端替换真实客户端
    import debater.nodes as nodes_module
    import debater.graph as graph_module
    
    # 保存原始类
    _orig_client_class = nodes_module.LLMClient
    
    # 替换为 Mock
    nodes_module.LLMClient = MockLLMClient  # type: ignore
    
    try:
        from debater.graph import run_debate
        
        result = run_debate(
            question="根据财务数据判断该公司盈利能力是否持续改善",
            context="""公司：某新能源电池企业
营业收入（亿元）：2021:156.8 / 2022:289.4(+84.6%) / 2023:412.7(+42.6%)
归母净利润（亿元）：2021:12.3 / 2022:28.6(+132.5%) / 2023:35.2(+23.1%)
毛利率：2021:18.5% / 2022:22.3% / 2023:19.8%
研发费用（亿元）：2021:15.2 / 2022:28.4 / 2023:42.6
""",
            scenario_type="financial_interpretation",
            max_rounds=2,
        )
        
        print("\n" + "=" * 70)
        print("📋 最终统一结论")
        print("=" * 70)
        print(result["final_answer"])
        
        print("\n" + "=" * 70)
        print("📊 各模型最终置信度")
        print("=" * 70)
        for mid, conf in result["confidences"].items():
            print(f"   {mid}: {conf*100:.0f}%")
        
    finally:
        # 恢复原始客户端
        nodes_module.LLMClient = _orig_client_class


if __name__ == "__main__":
    run_mock_debate()
