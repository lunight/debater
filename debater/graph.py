"""LangGraph 图定义

将各节点组装成完整的多模型辩论工作流：

    initialize
        ↓
    generate  (并行生成初始回答)
        ↓
    critique  (并行互相评审)
        ↓
    revise    (并行修订回答)
        ↓
    judge     (判断是否收敛)
        ↓
      ┌───────┴───────┐
   未收敛            已收敛/最大轮次
      ↓                  ↓
    critique         aggregate
      ↑                  ↓
      └──── revise ←─── judge
                         ↓
                        END
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from .state import DebateState
from .nodes import DebateNodes


def create_debate_graph() -> StateGraph:
    """创建并返回辩论工作流图"""
    
    nodes = DebateNodes()
    
    # 定义图
    workflow = StateGraph(DebateState)
    
    # 添加节点
    workflow.add_node("initialize", nodes.initialize)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("critique", nodes.critique)
    workflow.add_node("revise", nodes.revise)
    workflow.add_node("judge", nodes.judge)
    workflow.add_node("aggregate", nodes.aggregate)
    
    # 定义边
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_edge("critique", "revise")
    workflow.add_edge("revise", "judge")
    
    # 条件边：判断下一步
    def route_after_judge(state: DebateState) -> Literal["critique", "aggregate"]:
        """根据 judge 结果决定下一步"""
        if state.get("consensus_reached", False):
            return "aggregate"
        # 未达成共识，继续下一轮
        return "critique"
    
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "critique": "critique",
            "aggregate": "aggregate",
        },
    )
    
    # 聚合后结束
    workflow.add_edge("aggregate", END)
    
    return workflow.compile()


def run_debate(
    question: str,
    context: str = "",
    scenario_type: str = "question_analysis",
    max_rounds: int = 3,
) -> DebateState:
    """运行一次完整的多模型辩论
    
    Args:
        question: 用户问题
        context: 附加上下文（如财务数据）
        scenario_type: 场景类型
        max_rounds: 最大辩论轮次
    
    Returns:
        最终状态，包含 final_answer 等结果
    """
    graph = create_debate_graph()
    
    initial_state: DebateState = {
        "question": question,
        "context": context,
        "scenario_type": scenario_type,
        "round": 0,
        "max_rounds": max_rounds,
        "answers": {},
        "previous_answers": {},
        "critiques": {},
        "confidences": {},
        "consensus_reached": False,
        "final_answer": None,
        "reasoning_process": "",
        "divergence_points": [],
        "role_models": {},
    }
    
    print("=" * 60)
    print(f"🚀 启动多模型辩论")
    print(f"📋 场景: {scenario_type}")
    print(f"📝 问题: {question[:80]}...")
    print(f"🔄 最大轮次: {max_rounds}")
    print("=" * 60)
    
    final_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("✅ 辩论结束")
    print("=" * 60)
    
    return final_state
