"""股票分析场景示例

展示如何使用多模型辩论系统进行：
- 财务数据解读
- 事件影响分析
- 风险评估
"""

import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from debater.graph import run_debate


def example_financial_interpretation():
    """示例：财务数据解读"""
    
    question = "根据以下财务数据，判断该公司盈利能力是否持续改善？"
    
    context = """
公司：某新能源电池企业（2021-2023年财务数据）

营业收入（亿元）：
- 2021: 156.8
- 2022: 289.4 (+84.6%)
- 2023: 412.7 (+42.6%)

归母净利润（亿元）：
- 2021: 12.3
- 2022: 28.6 (+132.5%)
- 2023: 35.2 (+23.1%)

毛利率：
- 2021: 18.5%
- 2022: 22.3%
- 2023: 19.8%

研发费用（亿元）：
- 2021: 15.2
- 2022: 28.4
- 2023: 42.6

行业背景：2022年受益于原材料价格下降和产能释放，行业整体盈利改善；
2023年行业竞争加剧，价格战导致毛利率承压，但头部企业份额提升。
"""
    
    result = run_debate(
        question=question,
        context=context,
        scenario_type="financial_interpretation",
        max_rounds=2,
    )
    
    print("\n" + "=" * 60)
    print("📊 财务解读结论")
    print("=" * 60)
    print(result["final_answer"])
    return result


def example_risk_assessment():
    """示例：风险评估"""
    
    question = "评估投资该房地产公司的主要风险"
    
    context = """
公司概况：某中型房地产开发商，主要布局二三线城市

关键数据：
- 资产负债率：78%（行业平均65%）
- 短期借款：156亿元，现金及等价物：89亿元
- 2023年合同销售额同比下降32%
- 存货周转天数：890天（2021年为420天）
- 已售未结金额：45亿元
- 三条红线：踩中两条（剔除预收款后的资产负债率、净负债率）

宏观环境：
- 当地出台限购放松政策
- 但居民购房意愿持续低迷
- 银行对房企开发贷仍偏谨慎
"""
    
    result = run_debate(
        question=question,
        context=context,
        scenario_type="risk_assessment",
        max_rounds=2,
    )
    
    print("\n" + "=" * 60)
    print("⚠️ 风险评估结论")
    print("=" * 60)
    print(result["final_answer"])
    return result


def example_event_analysis():
    """示例：事件分析"""
    
    question = "分析该监管政策对互联网教育行业的影响"
    
    context = """
事件：教育部等五部门发布《关于规范校外线上培训的实施意见》

核心条款：
1. 校外线上培训机构需取得办学许可证
2. 不得占用国家法定节假日、休息日组织学科类培训
3. 不得传播不良学习方法，不得渲染教育焦虑
4. 每节课不超过30分钟，课程间隔不少于10分钟
5. 不得提供和传播"拍照搜题"等惰化学生思维的功能
6. 预收费资金纳入银行监管，不得一次性收取超3个月费用

受影响的上市公司：
- A公司：K12在线辅导龙头，营收中70%为学科类培训
- B公司：成人职业教育为主，受影响较小
- C公司：教育信息化服务商，主要面向学校采购
"""
    
    result = run_debate(
        question=question,
        context=context,
        scenario_type="event_analysis",
        max_rounds=2,
    )
    
    print("\n" + "=" * 60)
    print("📰 事件分析结论")
    print("=" * 60)
    print(result["final_answer"])
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "example",
        choices=["financial", "risk", "event", "all"],
        help="选择要运行的示例",
    )
    args = parser.parse_args()
    
    if args.example == "financial":
        example_financial_interpretation()
    elif args.example == "risk":
        example_risk_assessment()
    elif args.example == "event":
        example_event_analysis()
    elif args.example == "all":
        example_financial_interpretation()
        example_risk_assessment()
        example_event_analysis()
