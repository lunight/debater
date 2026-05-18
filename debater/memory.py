"""Memory 管理系统

三层架构：
- Working Memory：每5轮增量更新，活跃讨论区（不含事实，事实单独维护）
- Facts Registry：独立维护，追加式，支持标记失效，不参与压缩
- Stage Memory：每10轮落地快照 + 压缩 Working Memory
- Obsidian：辩论结束一次性生成
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .llm_client import LLMClient
from .config import get_config


class MemoryManager:
    """Memory 管理器"""
    
    STAGE_INTERVAL = 10  # 每10轮落地一次
    UPDATE_INTERVAL = 5  # 每5轮更新一次
    
    def __init__(self, debate_id: str, topic: str):
        self.debate_id = debate_id
        self.topic = topic
        self.root = Path("/Users/lunight/dev/debater/memories")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "stage_memory").mkdir(exist_ok=True)
        (self.root / "obsidian").mkdir(exist_ok=True)
        
        self.working_file = self.root / "working_memory.md"
        self.facts_file = self.root / "facts.json"
        self.client = LLMClient()
        self.config = get_config()
        
        # 初始化 working memory
        if not self.working_file.exists():
            self._init_working_memory()
        
        # 初始化 facts registry
        if not self.facts_file.exists():
            self._save_facts({"facts": [], "version": 1})
    
    # ==================== Facts Registry ====================
    
    def _load_facts(self) -> Dict[str, Any]:
        """加载事实注册表"""
        try:
            return json.loads(self.facts_file.read_text(encoding="utf-8"))
        except Exception:
            return {"facts": [], "version": 1}
    
    def _save_facts(self, data: Dict[str, Any]):
        """保存事实注册表"""
        self.facts_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def add_fact(self, text: str, source_round: int, confidence: float = 1.0) -> bool:
        """添加一条事实（自动去重）
        
        Args:
            text: 事实文本
            source_round: 来源轮次
            confidence: 事实可信度（0.0~1.0）
        
        Returns:
            True if added, False if duplicate or empty
        """
        text = text.strip()
        if not text or len(text) < 2:
            return False
        
        data = self._load_facts()
        facts = data.get("facts", [])
        
        # 简单去重：完全相同 或 新事实被已有事实包含（已有更完整，无需重复添加）
        for f in facts:
            if not f.get("stale", False):
                existing = f.get("text", "")
                if text == existing or text in existing:
                    return False
        
        facts.append({
            "text": text,
            "source_round": source_round,
            "added_at": datetime.now().isoformat(),
            "stale": False,
            "stale_reason": "",
            "confidence": confidence,
        })
        
        self._save_facts(data)
        return True
    
    def mark_fact_stale(self, text_or_index, reason: str = ""):
        """标记事实为失效
        
        Args:
            text_or_index: 事实文本（部分匹配）或索引
            reason: 失效原因
        """
        data = self._load_facts()
        facts = data.get("facts", [])
        
        if isinstance(text_or_index, int) and 0 <= text_or_index < len(facts):
            facts[text_or_index]["stale"] = True
            facts[text_or_index]["stale_reason"] = reason
            facts[text_or_index]["stale_at"] = datetime.now().isoformat()
        elif isinstance(text_or_index, str):
            for f in facts:
                if text_or_index in f.get("text", ""):
                    f["stale"] = True
                    f["stale_reason"] = reason
                    f["stale_at"] = datetime.now().isoformat()
                    break
        
        self._save_facts(data)
    
    def get_facts_context(self, max_items: int = 30) -> str:
        """获取格式化的事实文本，用于 prompt 注入
        
        Returns:
            Markdown 格式的事实列表，失效事实默认不显示
        """
        data = self._load_facts()
        facts = [f for f in data.get("facts", []) if not f.get("stale", False)]
        
        if not facts:
            return ""
        
        lines = ["## 已验证核心事实", ""]
        for i, f in enumerate(facts[-max_items:], 1):
            text = f.get("text", "")
            src = f.get("source_round", "?")
            conf = f.get("confidence", 1.0)
            conf_str = f" (置信度{conf*100:.0f}%)" if conf < 1.0 else ""
            lines.append(f"{i}. {text}{conf_str} [R{src}]")
        
        lines.append("")
        return "\n".join(lines)
    
    def get_all_facts(self, include_stale: bool = False) -> List[Dict[str, Any]]:
        """获取所有事实（可选包含已失效的）"""
        data = self._load_facts()
        facts = data.get("facts", [])
        if not include_stale:
            facts = [f for f in facts if not f.get("stale", False)]
        return facts
    
    # ==================== Working Memory ====================
    
    def _init_working_memory(self):
        """初始化工作记忆"""
        content = f"""---
debate_id: {self.debate_id}
topic: {self.topic}
created: {datetime.now().isoformat()}
rounds_covered: 0
---

# Working Memory - {self.topic}

> ⚠️ 本文档在辩论过程中持续更新。每{self.UPDATE_INTERVAL}轮更新一次。
> 注意：核心事实已单独维护在 facts registry 中，此处只保留观点、分歧和推理框架。

## 活跃观点
| 角色 | 当前立场 | 置信度 | 变化轨迹 |
|------|----------|--------|----------|
| （待补充） | （待补充） | - | - |

## 关键分歧（未解决）
（待补充）

## 待跟踪
（待补充）

## 上次更新：第0轮
"""
        self.working_file.write_text(content, encoding="utf-8")
    
    def update_working_memory(self, rounds_data: List[Dict[str, Any]]):
        """增量更新工作记忆，同时提取新事实到 Facts Registry"""
        current = self.working_file.read_text(encoding="utf-8")
        
        # 构建最近5轮的文本
        rounds_text = ""
        for rd in rounds_data:
            rounds_text += f"\n### 第{rd['round']}轮 · {rd['type']}\n"
            for key in sorted(rd.keys()):
                if key.endswith("_output"):
                    safe_id = key[:-7]
                    conf_key = f"{safe_id}_conf"
                    thinking_key = f"{safe_id}_thinking"
                    conf = rd.get(conf_key, 0)
                    output = rd.get(key, '')[:800]
                    rounds_text += f"**{safe_id}** (置信度{conf*100:.0f}%)\n"
                    rounds_text += f"{output}...\n\n"
                    if rd.get(thinking_key):
                        rounds_text += f"{safe_id}思考: {rd[thinking_key][:300]}...\n\n"
        
        aggregator = self.config.get_aggregator_model()
        
        # === 第一步：让模型更新 Working Memory（不含事实） ===
        user_prompt = f"""你正在维护一份辩论工作记忆文档。

【当前工作记忆】
{current}

【最近{len(rounds_data)}轮新讨论】
{rounds_text}

更新要求：
1. 核心事实已单独维护，你不需要在工作记忆中重复事实细节
2. 保留：观点演进、分歧变化、推理框架修正
3. 新增：本轮出现的新立场、新分歧、态度转变
4. 删除：已被推翻的观点、已解决的分歧
5. 用 [Rxx] 标注信息来源轮次
6. 保留 frontmatter（--- 之间的内容），只更新 rounds_covered
7. 输出必须是标准 Markdown 格式

请输出更新后的完整工作记忆 Markdown。
"""
        
        try:
            resp = self.client.call(
                model_id=aggregator.id,
                system_prompt="你是一位资深纪要整理专家，擅长压缩和更新多轮讨论记录。",
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=4000,
            )
            
            new_content = resp.content
            latest_round = max(rd['round'] for rd in rounds_data)
            new_content = new_content.replace(
                f"rounds_covered: {latest_round - len(rounds_data)}",
                f"rounds_covered: {latest_round}"
            )
            
            self.working_file.write_text(new_content, encoding="utf-8")
            print(f"[MEMORY] Working memory 已更新至第{latest_round}轮")
            
        except Exception as e:
            print(f"[MEMORY] 更新 working memory 失败: {e}")
        
        # === 第二步：让模型提取新事实到 Facts Registry ===
        self._extract_facts_from_rounds(rounds_data, aggregator)
    
    def _extract_facts_from_rounds(self, rounds_data: List[Dict[str, Any]], aggregator):
        """从最近轮次中提取新事实，追加到 Facts Registry"""
        # 收集所有角色输出
        all_outputs = []
        for rd in rounds_data:
            for key in sorted(rd.keys()):
                if key.endswith("_output"):
                    text = rd.get(key, '')[:1500]
                    if text:
                        all_outputs.append(f"【第{rd['round']}轮 · {key[:-7]}】\n{text}\n")
        
        if not all_outputs:
            return
        
        combined_text = "\n".join(all_outputs)
        
        # 获取已有事实（避免重复提取）
        existing_facts = self.get_all_facts(include_stale=False)
        existing_texts = [f["text"] for f in existing_facts]
        existing_summary = "\n".join(f"- {t}" for t in existing_texts) if existing_texts else "（暂无）"
        
        user_prompt = f"""请从以下辩论记录中提取**新的事实/数据点**。

【已有事实】（不要重复提取）
{existing_summary}

【最近讨论记录】
{combined_text}

提取要求：
1. 只提取**可被验证的客观事实或具体数据**，不要提取观点、推测、推理
2. 事实必须包含具体数字、时间、比例、金额等量化信息（如果有）
3. 每个事实一行，格式：事实描述 [R轮次]
4. 如果该事实在"已有事实"列表中已存在（或高度相似），不要重复提取
5. 如果没有新事实，输出"无"

请直接输出事实列表，不要加任何解释。
"""
        
        try:
            resp = self.client.call(
                model_id=aggregator.id,
                system_prompt="你是一位严格的事实核查员，只提取客观可验证的数据和事实。",
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = resp.content.strip()
            if content == "无" or not content:
                return
            
            # 解析事实列表
            added_count = 0
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("-") is False:
                    continue
                # 去掉开头的 "- "
                if line.startswith("-"):
                    line = line[1:].strip()
                
                # 提取 [Rxx] 标注
                source_round = 0
                import re
                round_match = re.search(r'\[R(\d+)\]', line)
                if round_match:
                    source_round = int(round_match.group(1))
                    line = re.sub(r'\s*\[R\d+\]\s*', ' ', line).strip()
                
                if self.add_fact(line, source_round=source_round):
                    added_count += 1
            
            if added_count > 0:
                print(f"[MEMORY] 从最近讨论中提取了 {added_count} 条新事实")
                
        except Exception as e:
            print(f"[MEMORY] 提取事实失败: {e}")
    
    def should_update(self, current_round: int) -> bool:
        if current_round == 0:
            return False
        if current_round == 2:
            return True
        return current_round % self.UPDATE_INTERVAL == 0
    
    def should_save_stage(self, current_round: int) -> bool:
        if current_round == 0:
            return False
        return current_round % self.STAGE_INTERVAL == 0
    
    def save_stage_memory(self, round_num: int):
        """落地 stage memory：保存完整快照 + 压缩 Working Memory"""
        # 1. 保存完整快照
        current = self.working_file.read_text(encoding="utf-8")
        stage_file = self.root / "stage_memory" / f"stage_{round_num}.md"
        stage_file.write_text(current, encoding="utf-8")
        print(f"[MEMORY] Stage memory 已落地: {stage_file}")
        
        # 2. 压缩 working memory（保留骨架，不碰 facts）
        aggregator = self.config.get_aggregator_model()
        
        # 获取当前事实数量，用于 prompt 说明
        facts_count = len(self.get_all_facts(include_stale=False))
        
        user_prompt = f"""请把以下工作记忆压缩成"骨架版"，用于后续轮次快速回顾。

重要：核心事实已单独维护（当前共 {facts_count} 条），你**不需要在骨架中重复事实细节**。
骨架只需保留：观点演进、分歧框架、推理脉络、待跟踪事项。

要求：
- 保留：核心结论、关键分歧框架、待跟踪事项、立场变化轨迹
- 删除：具体论据、计算过程、已被解决的子问题、事实数据（已单独维护）
- 保留 frontmatter
- 输出不超过500字（不含 frontmatter）

【完整工作记忆】
{current}
"""
        
        try:
            resp = self.client.call(
                model_id=aggregator.id,
                system_prompt="你是一位资深纪要整理专家。输出必须是标准Markdown格式。",
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2000,
            )
            
            skeleton = resp.content
            # 确保 frontmatter 正确
            if "---" not in skeleton:
                skeleton = f"""---
debate_id: {self.debate_id}
topic: {self.topic}
archived: {datetime.now().isoformat()}
rounds_covered: {round_num}
---

# Working Memory - {self.topic}

> 📦 前{round_num}轮已归档至 [[stage_memory/stage_{round_num}.md]]
> 核心事实已单独维护，详见 facts registry。

## 当前骨架
{skeleton}
"""
            
            self.working_file.write_text(skeleton, encoding="utf-8")
            print(f"[MEMORY] Working memory 已压缩为骨架版（不含事实）")
            
        except Exception as e:
            print(f"[MEMORY] 压缩 working memory 失败: {e}")
    
    def generate_obsidian(self, session) -> str:
        """生成 Obsidian 文档"""
        stage_files = sorted((self.root / "stage_memory").glob("stage_*.md"))
        stages_content = ""
        for f in stage_files:
            stages_content += f"\n## {f.stem}\n\n"
            stages_content += f.read_text(encoding="utf-8")[:1500]
            stages_content += "\n\n---\n\n"
        
        final_wm = self.working_file.read_text(encoding="utf-8") if self.working_file.exists() else ""
        facts = self.get_all_facts(include_stale=False)
        facts_text = "\n".join(f"- {f['text']} [R{f['source_round']}]" for f in facts) if facts else "（无）"
        tags = self._extract_tags(session)
        
        aggregator = self.config.get_aggregator_model()
        user_prompt = f"""你是一位资深投研总监。请基于以下辩论记录，生成一份投资备忘录。

【核心事实】
{facts_text}

【阶段记忆】
{stages_content}

【最终工作记忆（骨架）】
{final_wm}

要求：
1. 用 Markdown 格式
2. 包含 Obsidian frontmatter（tags, date, topic）
3. 提取关键实体，使用 [[双向链接]] 语法
4. 结构：问题定义 → 核心结论 → 关键实体 → 核心论据 → 关键分歧 → 风险提示 → 后续跟踪
5. 在文末添加 "辩论档案" 章节，链接到所有 stage memory
"""
        
        try:
            resp = self.client.call(
                model_id=aggregator.id,
                system_prompt="你是一位资深投资总监，擅长撰写机构级投资备忘录。",
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=4000,
            )
            
            conf_lines = []
            for role_id in session.role_models.keys():
                conf = session.confidences.get(role_id, 0) * 100
                conf_lines.append(f"final_{role_id}_conf: {conf:.0f}%")
            
            frontmatter = f"""---
topic: {self.topic}
date: {datetime.now().strftime('%Y-%m-%d')}
debate_id: {self.debate_id}
rounds: {session.current_round}
tags: {json.dumps(tags)}
{chr(10).join(conf_lines)}
---

"""
            
            doc_content = frontmatter + resp.content
            
            # 添加辩论档案
            doc_content += "\n\n## 辩论档案\n\n"
            for f in stage_files:
                doc_content += f"- [[{f.name}]]\n"
            doc_content += f"- [[working_memory.md]]\n"
            doc_content += f"- [[facts.json]]\n"
            
            # 写入文件
            filename = f"debate_{self.debate_id}_{datetime.now().strftime('%Y%m%d')}.md"
            output_file = self.root / "obsidian" / filename
            output_file.write_text(doc_content, encoding="utf-8")
            
            print(f"[MEMORY] Obsidian 文档已生成: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"[MEMORY] 生成 Obsidian 失败: {e}")
            return ""
    
    def generate_mid_obsidian(self, session, current_round: int) -> str:
        """生成中途 Obsidian 总结文档"""
        wm = self.read_working_memory()
        facts = self.get_all_facts(include_stale=False)
        
        lines = [f"# 辩论进行中总结 - {self.topic}", ""]
        lines.append(f"> **状态**: 进行中 | **当前轮次**: 第 {current_round} 轮")
        lines.append(f"> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # 各模型当前立场
        lines.append("## 当前立场")
        lines.append("")
        lines.append("| 角色 | 当前结论 | 置信度 |")
        lines.append("|------|----------|--------|")
        for role_id in session.role_models.keys():
            name = role_id
            answer = session.answers.get(role_id, "（暂无）")[:100]
            conf = session.confidences.get(role_id, 0) * 100
            lines.append(f"| {name} | {answer}... | {conf:.0f}% |")
        lines.append("")
        
        # 核心事实
        if facts:
            lines.append("## 核心事实")
            lines.append("")
            for f in facts:
                text = f.get("text", "")
                src = f.get("source_round", "?")
                lines.append(f"- {text} [R{src}]")
            lines.append("")
        
        # 评审历史
        if session.critiques_history:
            lines.append("## 评审历史")
            lines.append("")
            for round_num in sorted(session.critiques_history.keys()):
                if round_num >= current_round:
                    continue
                lines.append(f"### 第 {round_num} 轮")
                round_data = session.critiques_history[round_num]
                for critiquer_id, targets in round_data.items():
                    c_name = critiquer_id.replace("_", " ")
                    for target_id, text in targets.items():
                        t_name = target_id.replace("_", " ")
                        text_short = text[:120] + "..." if len(text) > 120 else text
                        lines.append(f"- **{c_name}** → {t_name}: {text_short}")
                lines.append("")
        
        # Working Memory 内容
        if wm:
            lines.append("## Working Memory（当前骨架）")
            lines.append("")
            lines.append("```")
            lines.append(wm[:2000])
            if len(wm) > 2000:
                lines.append("...（已截断）")
            lines.append("```")
            lines.append("")
        
        # 待跟踪事项
        lines.append("## 待跟踪事项")
        lines.append("")
        lines.append("- [ ] 继续深入讨论关键分歧")
        lines.append("- [ ] 关注置信度变化趋势")
        lines.append("- [ ] 补充背景资料或数据")
        lines.append("")
        
        # Frontmatter
        frontmatter = f"""---
topic: {self.topic}
date: {datetime.now().strftime('%Y-%m-%d')}
debate_id: {self.debate_id}
current_round: {current_round}
status: in_progress
tags: ["辩论进行中", "临时总结"]
---

"""
        
        doc_content = frontmatter + "\n".join(lines)
        
        filename = f"debate_{self.debate_id}_mid_R{current_round}_{datetime.now().strftime('%Y%m%d')}.md"
        output_file = self.root / "obsidian" / filename
        output_file.write_text(doc_content, encoding="utf-8")
        
        print(f"[MEMORY] 中途总结已生成: {output_file}")
        return str(output_file)
    
    def _extract_tags(self, session) -> List[str]:
        tags = []
        
        scenario_tags = {
            "question_analysis": "问题分析",
            "financial_interpretation": "财务分析",
            "event_analysis": "事件分析",
            "risk_assessment": "风险评估",
        }
        if session.scenario_type in scenario_tags:
            tags.append(scenario_tags[session.scenario_type])
        
        question = session.question.lower()
        industry_keywords = {
            "新能源": ["新能源", "电池", "锂电", "光伏", "储能"],
            "房地产": ["房地产", "地产", "房价", "楼市"],
            "互联网": ["互联网", "电商", "平台", "app"],
            "医药": ["医药", "药品", "医疗", "生物"],
            "金融": ["银行", "保险", "证券", "金融"],
        }
        
        for industry, keywords in industry_keywords.items():
            if any(kw in question for kw in keywords):
                tags.append(industry)
        
        return tags
    
    def get_context_for_prompt(self) -> str:
        """获取用于 prompt 注入的 context 字符串
        
        合并返回：
        1. Facts Registry（已验证核心事实，不压缩）
        2. Working Memory（骨架版，可能被压缩过）
        
        早期轮次返回空字符串。
        """
        parts = []
        
        # 1. 事实部分（始终完整，不参与压缩）
        facts_ctx = self.get_facts_context()
        if facts_ctx:
            parts.append(facts_ctx)
        
        # 2. Working Memory 骨架
        if self.working_file.exists():
            content = self.working_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts_split = content.split("---", 2)
                if len(parts_split) >= 3:
                    content = parts_split[2].strip()
            
            # 如果内容还是初始模板，跳过
            if "（待补充）" not in content or "上次更新：第0轮" not in content:
                if content.strip():
                    parts.append("## 辩论演进摘要\n\n" + content)
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)
    
    def read_working_memory(self) -> str:
        if self.working_file.exists():
            return self.working_file.read_text(encoding="utf-8")
        return ""
