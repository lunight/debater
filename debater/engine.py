"""公共引擎工具函数

抽取 nodes.py 与 session.py 共用的纯工具函数，消除代码重复。
所有函数均为纯函数（无状态/无 side effect），不依赖特定类实例。
"""

import re
from typing import Tuple, Optional


# 数据缺失关键词（用于置信度强制降权）
_DATA_GAP_KEYWORDS = [
    "工具失败", "数据缺失", "无法获取", "无法验证", "未经工具验证",
    "缺少关键数据", "缺少验证", "无法确认", "信息不足",
    "工具调用失败", "搜索失败", "未找到相关结果",
]


def extract_confidence(text: str) -> float:
    """从回答中提取置信度 (0.0~1.0)
    
    如果检测到工具失败导致的数据缺失关键词，强制降低置信度，
    防止模型基于不完整信息给出高置信度结论。
    """
    text_lower = text.lower()
    has_data_gap = any(kw in text_lower for kw in _DATA_GAP_KEYWORDS)

    patterns = [
        r'置信度[^\d]*(\d+(?:\.\d+)?)\s*%',
        r'(?:confidence|Confidence)[^\d]*(\d+(?:\.\d+)?)\s*%',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            confidence = min(max(val / 100.0, 0.0), 1.0)
            if has_data_gap:
                confidence = min(confidence, 0.5)
            return confidence

    # 默认置信度：有数据缺失时更低
    return 0.3 if has_data_gap else 0.7


def extract_thinking(text: str) -> Tuple[str, str]:
    """从回答中提取 thinking 过程和正式输出
    
    同时移除 <tool_call> 和 <tool_result> 标签，避免它们出现在最终输出中。
    
    Returns:
        (thinking_text, formal_text)
    """
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        formal = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
    else:
        thinking = ""
        formal = text

    # 移除 tool_call 和 tool_result 标签
    formal = re.sub(
        r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
        '', formal, flags=re.DOTALL
    ).strip()
    formal = re.sub(
        r'<tool_result\s+name="[^"]+">\s*.*?\s*</tool_result>',
        '', formal, flags=re.DOTALL
    ).strip()

    return thinking, formal


def extract_summary(text: str) -> str:
    """从回答中提取 <summary> 总结块"""
    match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


import json


def parse_judge_output(text: str) -> dict:
    """从 Judge 模型的输出中提取 action 和 reason
    
    支持多种输出格式：
    1. ```json 代码块
    2. 裸 JSON 对象
    3. 自然语言中包含关键词（continue/stop/need_info）
    4. 默认 fallback
    
    Returns:
        {"action": "continue|stop|need_info", "reason": str, "info_needed": str}
    """
    text = text.strip()
    
    # 1. 尝试提取 ```json 代码块
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 2. 尝试提取裸 JSON（第一个 { 到最后一个 }）
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 3. 关键词提取 fallback（即使 JSON 解析失败也能提取意图）
    text_lower = text.lower()
    if '"stop"' in text_lower or "'stop'" in text_lower or 'action: stop' in text_lower:
        return {"action": "stop", "reason": text[:300], "info_needed": ""}
    if '"need_info"' in text_lower or "'need_info'" in text_lower or 'action: need_info' in text_lower or '需要补充信息' in text_lower:
        return {"action": "need_info", "reason": text[:300], "info_needed": ""}
    # 默认 continue，保留解析失败提示便于调试
    return {"action": "continue", "reason": f"JSON 解析失败，回退到关键词提取: {text[:200]}", "info_needed": ""}


def clean_tool_tags(text: str) -> str:
    """移除文本中的所有 tool_call / tool_result XML 标签
    
    用于从模型输出中清理掉工具调用标记，保留纯分析文本。
    """
    if not text:
        return ""
    cleaned = re.sub(
        r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
        '', text, flags=re.DOTALL
    )
    cleaned = re.sub(
        r'<tool_result\s+name="[^"]+">\s*.*?\s*</tool_result>',
        '', cleaned, flags=re.DOTALL
    )
    return cleaned.strip()
