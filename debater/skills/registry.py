"""Skill 注册表

管理已加载的 skills，支持按场景类型匹配。
"""

from typing import Dict, List, Optional
from .loader import Skill, SkillLoader


# 场景 → skill 显式映射（确保每个场景都能匹配到合适的 skill）
SCENARIO_SKILL_MAP: Dict[str, str] = {
    "question_analysis": "brainstorming",
    "financial_interpretation": "3-statement-model",
    "event_analysis": "competitive-analysis",
    "risk_assessment": "systematic-debugging",
}


class SkillRegistry:
    """Skill 注册表"""
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._loader = SkillLoader()
        self._loaded = False
    
    def load(self):
        """加载所有 skill（延迟加载）"""
        if self._loaded:
            return
        self._skills = self._loader.load_all()
        self._loaded = True
        if self._skills:
            print(f"[SKILL] 已加载 {len(self._skills)} 个 skill: {list(self._skills.keys())}")
    
    def get(self, name: str) -> Optional[Skill]:
        """按名称获取 skill"""
        self.load()
        return self._skills.get(name)
    
    def get_by_scenario(self, scenario_type: str) -> Optional[Skill]:
        """按场景类型匹配 skill
        
        匹配规则（按优先级）：
        1. 显式映射（SCENARIO_SKILL_MAP）
        2. 精确匹配 scenario_type 作为 skill name
        3. skill name 包含 scenario_type 关键词
        4. skill description 包含 scenario_type 关键词
        """
        self.load()
        
        # 1. 显式映射（最高优先级，确保场景能匹配到正确的 skill）
        mapped_name = SCENARIO_SKILL_MAP.get(scenario_type)
        if mapped_name and mapped_name in self._skills:
            return self._skills[mapped_name]
        
        # 2. 精确匹配
        if scenario_type in self._skills:
            return self._skills[scenario_type]
        
        # 3. skill name 包含 scenario_type
        for name, skill in self._skills.items():
            if scenario_type.replace("_", "-") in name or scenario_type in name:
                return skill
        
        # 4. description 包含 scenario_type 关键词（兜底）
        keywords = scenario_type.split("_")
        for skill in self._skills.values():
            desc_lower = skill.description.lower()
            if any(kw in desc_lower for kw in keywords):
                return skill
        
        return None
    
    def list_all(self) -> List[Skill]:
        """列出所有 skill"""
        self.load()
        return list(self._skills.values())
    
    def get_context_for_scenario(self, scenario_type: str) -> str:
        """获取某场景的 skill context（用于 prompt 注入）"""
        skill = self.get_by_scenario(scenario_type)
        if not skill:
            return ""
        
        return f"""
【专业 Skill 指导】
{skill.description}

{skill.body[:3000]}  
""".strip()
