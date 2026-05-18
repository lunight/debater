"""LangGraph 状态定义"""

from typing import Dict, List, TypedDict, Optional, Any


class DebateState(TypedDict):
    """辩论状态
    
    整个多模型辩论过程的状态快照，LangGraph 会在每个节点间传递此状态。
    """
    # === 输入 ===
    question: str
    """用户提出的问题"""
    
    context: str
    """附加上下文（如财务数据、新闻内容等）"""
    
    scenario_type: str
    """场景类型: question_analysis / financial_interpretation / event_analysis / risk_assessment"""
    
    # === 过程状态 ===
    round: int
    """当前辩论轮次（从 1 开始）"""
    
    max_rounds: int
    """最大辩论轮次"""
    
    answers: Dict[str, str]
    """当前轮次各模型的回答 {model_id: answer}"""
    
    previous_answers: Dict[str, str]
    """历史回答记录 {model_id: latest_answer}"""
    
    role_models: Dict[str, str]
    """角色-模型映射 {role_id: model_id}，如 {'bull': 'kimi', 'bear': 'deepseek'}"""
    
    critiques: Dict[str, Dict[str, str]]
    """评审意见 {model_id: {target_model_id: critique_text}}"""
    
    confidences: Dict[str, float]
    """各模型对当前答案的置信度 {model_id: 0.0~1.0}"""
    
    consensus_reached: bool
    """是否已达成共识"""
    
    # === 输出 ===
    final_answer: Optional[str]
    """最终统一结论"""
    
    reasoning_process: Optional[str]
    """推理过程记录（用于追溯）"""
    
    divergence_points: Optional[List[str]]
    """分歧点记录（即使达成共识，也保留分歧点）"""
