"""手动控制的辩论会话

支持逐步推进、用户随时介入评判的可视化辩论流程。
不依赖 LangGraph 的自动流转，每个阶段结束后暂停等待用户决策。
"""

import json
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .config import get_config, reload_config
from . import prompts
from .memory import MemoryManager
from .skills import SkillRegistry
from .tools import ToolRegistry
from .logger import llm_logger, session_logger, log_prompt, log_response, log_event


class Stage:
    """辩论阶段"""
    INIT = "init"
    GENERATING = "generating"
    PAUSE_AFTER_GENERATE = "pause_after_generate"
    CRITIQUING = "critiquing"
    PAUSE_AFTER_CRITIQUE = "pause_after_critique"
    REVISING = "revising"
    PAUSE_AFTER_REVISE = "pause_after_revise"
    AGGREGATING = "aggregating"
    DONE = "done"


@dataclass
class DebateSession:
    """辩论会话"""
    
    # 输入
    question: str
    scenario_type: str
    context: str
    
    # 状态
    stage: str = Stage.INIT
    current_round: int = 0
    answers: Dict[str, str] = field(default_factory=dict)           # 正式输出
    thinkings: Dict[str, str] = field(default_factory=dict)         # 思考过程
    summaries: Dict[str, str] = field(default_factory=dict)         # 每轮末尾总结
    critiques_history: Dict[int, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    # critiques 保留向后兼容（最新一轮的快照）
    critiques: Dict[str, Dict[str, str]] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    previous_answers: Dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    known_facts: List[str] = field(default_factory=list)            # 已知事实（用户补充）
    
    # Memory 系统
    memory_manager: Optional[MemoryManager] = None
    
    # 角色-模型映射：固定两个角色（bull / bear），各分配一个模型
    role_models: Dict[str, str] = field(default_factory=dict)
    
    # 运行时
    _client: Optional[LLMClient] = None
    _config: Optional[Any] = None
    _models: List[str] = field(default_factory=list)
    
    @staticmethod
    def _dedupe_model_ids(model_ids: List[str], context: str = "") -> List[str]:
        """去重 model ID 列表，保持顺序。"""
        seen = set()
        unique = []
        dupes = []
        for m in model_ids:
            if m not in seen:
                unique.append(m)
                seen.add(m)
            else:
                dupes.append(m)
        if dupes:
            llm_logger.warning(
                f"{context} 检测到重复模型 ID: {dupes}，已自动去重为: {unique}"
            )
        return unique
    
    def _role_of_model(self, model_id: str) -> str:
        """根据 model_id 反查其角色（bull/bear）"""
        for role, mid in self.role_models.items():
            if mid == model_id:
                return role
        return ""
    
    def _sync_models_from_roles(self):
        """根据 role_models 同步 _models 列表（去重）"""
        self._models = list(dict.fromkeys(self.role_models.values()))
    
    # 类常量
    MAX_TOOL_ROUNDS = 10  # 每角色最多工具链 followup 轮数
    
    def __post_init__(self):
        self._client = LLMClient()
        self._config = get_config()
        available = self._dedupe_model_ids(
            list(self._config.get_proposer_models().keys()),
            "[DebateSession.__init__]"
        )
        
        # 默认角色分配：第一个模型 = bull，第二个 = bear
        self.role_models = {
            "bull": available[0] if len(available) > 0 else "",
            "bear": available[1] if len(available) > 1 else available[0] if len(available) > 0 else "",
        }
        self._sync_models_from_roles()
        
        self.answers = {rid: "" for rid in self.role_models.keys()}
        self.thinkings = {rid: "" for rid in self.role_models.keys()}
        self.summaries = {rid: "" for rid in self.role_models.keys()}
        self.confidences = {rid: 0.0 for rid in self.role_models.keys()}
        self.previous_answers = {rid: "" for rid in self.role_models.keys()}
        self.critiques_history = {}
        self.critiques = {rid: {} for rid in self.role_models.keys()}
        self.known_facts = []
        
        # 初始化 Memory、Skill、Tool 系统（避免未调用 reload_config 时崩溃）
        self._skill_registry = SkillRegistry()
        self._skill_registry.load()
        self._tool_registry = ToolRegistry(base_dir=os.getcwd())
        self._tool_registry.register_default_tools()
    
    def set_role_model(self, role_id: str, model_id: str):
        """为指定角色设置模型
        
        切换模型后，新轮次将使用新模型。历史数据保留。
        
        Args:
            role_id: "bull" 或 "bear"
            model_id: 模型 ID（必须在配置中存在）
        """
        if role_id not in ("bull", "bear"):
            llm_logger.warning(f"[set_role_model] 无效角色 '{role_id}'，忽略")
            return
        
        old_model = self.role_models.get(role_id)
        self.role_models[role_id] = model_id
        self._sync_models_from_roles()
        
        # 如果该角色从未初始化过，创建空状态
        if role_id not in self.answers:
            self.answers[role_id] = ""
            self.thinkings[role_id] = ""
            self.summaries[role_id] = ""
            self.confidences[role_id] = 0.0
            self.previous_answers[role_id] = ""
            self.critiques[role_id] = {}
        
        llm_logger.info(f"[set_role_model] {role_id} -> {model_id} (原: {old_model})")
    
    def reload_config(self, config_path: Optional[str] = None):
        """重新加载配置文件并刷新客户端
        
        用于 UI 上动态切换配置文件。
        """
        # 重新加载全局配置
        reload_config(config_path)
        # 刷新 LLMClient
        self._client.reload()
        # 刷新本地配置引用
        self._config = get_config()
        # 重新获取模型列表（保留用户已选择的）
        available = set(self._config.models.keys())
        self._models = [m for m in self._models if m in available]
        # 如果有新模型加入配置，可以自动添加（但这里保持用户选择不变）
        
        # 初始化 MemoryManager
        from datetime import datetime
        debate_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory_manager = MemoryManager(
            debate_id=debate_id,
            topic=self.question[:30],
        )
        
        # 初始化 SkillRegistry 和 ToolRegistry
        self._skill_registry = SkillRegistry()
        self._skill_registry.load()
        
        self._tool_registry = ToolRegistry(base_dir=os.getcwd())
        self._tool_registry.register_default_tools()
    
    def _get_skill_context(self) -> str:
        """获取当前场景对应的 skill context"""
        if self._skill_registry is None:
            return ""
        return self._skill_registry.get_context_for_scenario(self.scenario_type)
    
    def _get_tool_context(self) -> str:
        """获取 tool context（用于 prompt 注入）"""
        if self._tool_registry is None:
            return ""
        return self._tool_registry.get_prompt_context()
    
    def _execute_tool_calls(self, text: str, model_id: str = "unknown") -> Dict[str, Any]:
        """检查并执行文本中的 tool calls
        
        Args:
            text: 包含可能的 <tool_call> 标签的文本
            model_id: 调用该工具的模型 ID（用于日志关联）
        
        Returns:
            dict with keys:
            - text: tool results message for model injection
            - errors: list of error messages
            - has_error: whether any tool failed
            - fallback_notices: list of fallback notices (e.g. Tavily->DDG)
            - executions: list of detailed execution info for UI display
        """
        from .logger import tool_logger
        
        overall_start = time.time()
        text_preview = text[:200].replace("\n", " ") if text else "(empty)"
        tool_logger.info(
            f"[_execute_tool_calls] model={model_id} 开始 | "
            f"text_preview={text_preview!r} | text_len={len(text)}"
        )
        
        executor = self._tool_registry.create_executor()
        has_tc = executor.has_tool_calls(text)
        tool_logger.info(f"[_execute_tool_calls] model={model_id} has_tool_calls={has_tc}")
        
        if not has_tc:
            tool_logger.info(f"[_execute_tool_calls] model={model_id} 无 tool call，直接返回空结果")
            return {"text": "", "errors": [], "has_error": False, "fallback_notices": [], "executions": []}
        
        try:
            tool_logger.info(f"[_execute_tool_calls] model={model_id} 调用 extract_and_execute...")
            results = executor.extract_and_execute(text)
            tool_logger.info(
                f"[_execute_tool_calls] model={model_id} extract_and_execute 返回 {len(results)} 个结果"
            )
        except Exception as e:
            duration = time.time() - overall_start
            err_msg = f"extract_and_execute 异常: {type(e).__name__}: {e}"
            tool_logger.error(
                f"[_execute_tool_calls] model={model_id} CRASH | 耗时={duration:.2f}s | {err_msg}"
            )
            traceback.print_exc()
            return {
                "text": f"<tool_result name=\"internal_error\">\n【工具执行器崩溃】{err_msg}\n</tool_result>",
                "errors": [err_msg],
                "has_error": True,
                "fallback_notices": [],
                "executions": [{
                    "tool_name": "internal_error",
                    "params": {},
                    "status": "error",
                    "result_preview": "",
                    "error_detail": err_msg,
                    "sources": [],
                    "fallback_notice": None,
                    "duration": round(duration, 2),
                }],
            }
        
        if not results:
            duration = time.time() - overall_start
            tool_logger.info(f"[_execute_tool_calls] model={model_id} 完成 | 耗时={duration:.2f}s | 无结果")
            # 即使 results 为空，也返回非空 text，确保模型能进入 Round 2
            # results 为空意味着正则未匹配到 <tool_call>（尽管 has_tool_calls 返回了 True，可能是误检测）
            return {
                "text": '<tool_result name="internal_error">\n【工具解析异常】系统检测到工具调用标记但未能提取有效参数，请基于已有知识继续分析。\n</tool_result>',
                "errors": ["检测到工具调用标记但未能提取有效参数"],
                "has_error": True,
                "fallback_notices": [],
                "executions": [],
            }
        
        # 收集结构化信息（用于 UI 展示）
        errors = []
        fallback_notices = []
        executions = []
        for idx, r in enumerate(results):
            result = r["result"]
            status = "error" if result.error else ("success" if result.content else "empty")
            dur = r.get("duration", 0)
            
            # 提取结果预览（前 300 字符）
            content = result.content or ""
            preview = content[:300]
            if content and len(content) > 300:
                preview += "..."
            
            # 检测回退通知
            fb_notice = None
            if content and "[Tavily 不可用" in content:
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("[Tavily 不可用"):
                        fb_notice = line.strip("[]")
                        fallback_notices.append(fb_notice)
                        break
            
            if result.error:
                errors.append(f"[{r['name']}] {result.error}")
            
            executions.append({
                "tool_name": r["name"],
                "params": r["params"],
                "status": status,
                "result_preview": preview,
                "error_detail": result.error or "",
                "sources": result.sources or [],
                "fallback_notice": fb_notice,
                "duration": dur,
            })
            
            tool_logger.info(
                f"[_execute_tool_calls] model={model_id} 结果#{idx+1} | "
                f"tool={r['name']} | params={r['params']} | status={status} | "
                f"duration={dur:.2f}s | content_len={len(content)} | "
                f"error={result.error[:150] if result.error else 'None'}"
            )
        
        try:
            tool_text = executor.build_tool_result_message(results)
            tool_logger.info(
                f"[_execute_tool_calls] model={model_id} build_tool_result_message 成功 | "
                f"text_len={len(tool_text)}"
            )
        except Exception as e:
            err_msg = f"build_tool_result_message 异常: {type(e).__name__}: {e}"
            tool_logger.error(f"[_execute_tool_calls] model={model_id} {err_msg}")
            traceback.print_exc()
            tool_text = f"<tool_result name=\"internal_error\">\n{err_msg}\n</tool_result>"
            errors.append(err_msg)
        
        overall_duration = time.time() - overall_start
        tool_logger.info(
            f"[_execute_tool_calls] model={model_id} 完成 | 总耗时={overall_duration:.2f}s | "
            f"has_error={len(errors)>0} | errors={len(errors)} | "
            f"fallback_notices={len(fallback_notices)} | executions={len(executions)}"
        )
        
        return {
            "text": tool_text,
            "errors": errors,
            "has_error": len(errors) > 0,
            "fallback_notices": fallback_notices,
            "executions": executions,
        }
    
    def _build_tool_followup_prompt(
        self,
        previous_analysis: str,
        tool_info: Dict[str, Any],
        tool_round: int,
    ) -> str:
        """构建工具调用后的 followup prompt
        
        Args:
            previous_analysis: 上一轮清理后的分析文本（已移除 tool_call 标签）
            tool_info: _execute_tool_calls 返回的结构
            tool_round: 当前是第几轮 followup（从2开始）
        
        Returns:
            followup prompt 文本
        """
        has_error = tool_info["has_error"]
        errors = tool_info["errors"]
        fallback_notices = list(tool_info["fallback_notices"])
        
        # 检测模型是否只输出工具、没有自己的分析
        is_pure_tool_only = previous_analysis == "（你之前只调用了工具，尚未给出分析）"
        
        # 判断是否是最后一轮（强制禁止工具）
        is_last_round = tool_round >= self.MAX_TOOL_ROUNDS + 1
        
        # 构建工具策略提示
        if is_pure_tool_only:
            # 上一轮只输出工具，本轮强制禁止工具——必须先给出分析
            tool_strategy = (
                "【⚠️ 本轮强制禁止工具】\n"
                "你上一轮只输出了工具调用标签，没有给出自己的分析。\n"
                "**本轮绝对禁止调用任何工具**，直接基于已有信息（包括之前已获取的工具结果）"
                "输出完整的分析。你的专业判断才是核心，工具只是辅助。"
            )
        elif is_last_round:
            tool_strategy = (
                "【⚠️ 这是最后一轮分析】\n"
                "你已使用了全部可用工具调用机会。\n"
                "**绝对不要调用任何工具**，直接基于已有信息输出完整的最终分析。"
            )
        else:
            tool_strategy = (
                f"【第 {tool_round} 轮分析】\n"
                f"基于以上工具结果完善你的分析。\n"
                f"如果你认为还需要补充关键数据，可以继续调用工具（但不要重复搜索已查过的内容）。"
                f"如果信息已经足够，直接输出完整的分析。"
            )
        
        pure_tool_warning = ""
        if is_pure_tool_only:
            pure_tool_warning = (
                "\n⚠️ 【再次强调】本轮禁止调用任何工具。"
                "如果你再次输出 <tool_call> 标签，系统会直接忽略并判定为无效输出。"
                "请立即输出你的完整分析。"
            )
        
        # 判断工具执行状态：全部成功 / 部分失败 / 全部失败
        executions = tool_info.get("executions", [])
        total_count = len(executions)
        error_count = len(errors)
        
        if error_count > 0 and error_count < total_count:
            # 部分失败：既有成功结果，也有失败
            success_tools = [e["tool_name"] for e in executions if e.get("status") == "success"]
            failed_tools = [e["tool_name"] for e in executions if e.get("status") == "error"]
            error_hint = (
                f"\n⚠️ 警告：部分工具执行失败（{error_count}/{total_count}）。"
                f"成功：{', '.join(success_tools)}；"
                f"失败：{', '.join(failed_tools)}。"
                f"请基于已成功返回的数据继续分析，对失败的数据点标注'（搜索超时/失败，未经实时验证）'。"
            )
            prompt = (
                f"【你之前的分析】\n"
                f"{previous_analysis}\n\n"
                f"【工具返回的结果】\n"
                f"{tool_info['text']}\n\n"
                f"【原始问题】\n{self.question}\n\n"
                f"【上下文】\n{self.context or '（无）'}\n\n"
                f"{tool_strategy}{error_hint}{pure_tool_warning}"
            )
        elif error_count > 0 and error_count == total_count:
            # 全部失败
            prompt = (
                f"【你之前的分析】\n"
                f"{previous_analysis}\n\n"
                f"【🚨 工具调用全部失败——关键数据缺失】\n\n"
                f"你之前尝试调用工具验证关键数据，但全部执行失败：\n"
                + "\n".join(f"  - {e}" for e in errors) +
                f"\n\n"
                f"【原始问题】\n{self.question}\n\n"
                f"【上下文】\n{self.context or '（无）'}\n\n"
                f"{tool_strategy}\n"
                f"1. 你必须明确列出：哪些关键数据因工具失败而无法获取\n"
                f"2. 对于无法验证的数据点，**不得给出确定性结论**\n"
                f"3. **置信度必须低于 50%**\n"
                f"4. 在 <summary> 中明确标注'因工具失败导致的数据缺失盲区'\n"
                f"5. 不要虚构数据，不要假设工具应该返回的结果\n"
                f"6. 基于已有知识做分析，但每处涉及缺失数据的地方必须标注'（未经工具验证）'"
                f"{pure_tool_warning}"
            )
        else:
            # 全部成功（或没有工具调用）
            error_hint = ""
            if fallback_notices:
                error_hint = (
                    f"\n⚠️ 警告：部分工具经历了回退（{'；'.join(fallback_notices)}）。"
                    f"回退数据来源可能不如原始工具可靠，"
                    f"请在分析中明确标注哪些数据来自回退来源。"
                )
            prompt = (
                f"【你之前的分析】\n"
                f"{previous_analysis}\n\n"
                f"【工具返回的结果】\n"
                f"{tool_info['text']}\n\n"
                f"【原始问题】\n{self.question}\n\n"
                f"【上下文】\n{self.context or '（无）'}\n\n"
                f"{tool_strategy}{error_hint}{pure_tool_warning}"
            )
        
        return prompt
    
    def _build_simple_followup_prompt(self, previous_analysis: str) -> str:
        """构建 high risk 重试时的简化 followup prompt"""
        return (
            f"【你之前的分析】\n"
            f"{previous_analysis}\n\n"
            f"【工具状态】\n"
            f"搜索工具已执行，但结果因内容安全策略无法展示。"
            f"请基于你之前分析中已识别的信息缺口，"
            f"结合你的专业知识，直接输出完整的最终分析。\n\n"
            f"【原始问题】\n{self.question}\n\n"
            f"【上下文】\n{self.context or '（无）'}\n\n"
            f"不要调用任何工具，直接输出完整的最终分析。"
        )
    
    def _run_tool_chain_loop(
        self,
        role_id: str,
        model_id: str,
        model_name: str,
        system_prompt: str,
        first_round_text: str,
        event_queue: queue.Queue,
        initial_tool_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """运行工具链循环，支持最多 MAX_TOOL_ROUNDS 轮 followup
        
        Args:
            role_id: 角色 ID
            model_id: 模型 ID
            model_name: 模型显示名
            system_prompt: system prompt
            first_round_text: 第一轮完整输出
            event_queue: 事件队列（用于发送 token/tool_executing/error 事件）
            initial_tool_info: 如果提供，跳过第一次的 tool call 检测和执行，直接使用该结果
        
        Returns:
            最终 full_text，或 None（如果超过最大轮次仍输出 tool_call）
        """
        # 从第一轮输出中移除 tool_call 标签，保留分析内容
        clean_first_round = re.sub(
            r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
            '', first_round_text, flags=re.DOTALL
        ).strip()
        if not clean_first_round:
            clean_first_round = "（你之前只调用了工具，尚未给出分析）"
        
        current_text = first_round_text
        previous_analysis = clean_first_round
        tool_round = 1  # 已完成的生成轮次
        
        while True:
            executor = self._tool_registry.create_executor()
            has_tc = executor.has_tool_calls(current_text)
            
            if not has_tc:
                return current_text
            
            # 使用外部传入的 tool_info（如果提供），否则执行工具
            if initial_tool_info is not None and tool_round == 1:
                tool_info = initial_tool_info
                initial_tool_info = None  # 仅使用一次
            else:
                # 发送初始 tool_executing 事件（仅第一轮，且没有外部 tool_info 时）
                if tool_round == 1:
                    try:
                        detected = list(executor.TOOL_CALL_PATTERN.finditer(current_text))
                        tool_names = [m.group(1) for m in detected]
                        event_queue.put({
                            "type": "tool_executing",
                            "model_id": model_id,
                            "role_id": role_id,
                            "message": f"🔧 检测到 {len(detected)} 个工具调用，正在并行执行...",
                            "has_error": False,
                            "errors": [],
                            "fallback_notices": [],
                            "executions": [],
                        })
                    except AttributeError:
                        # 如果 executor 没有 TOOL_CALL_PATTERN，跳过详细检测
                        event_queue.put({
                            "type": "tool_executing",
                            "model_id": model_id,
                            "role_id": role_id,
                            "message": "🔧 检测到工具调用，正在执行...",
                            "has_error": False,
                            "errors": [],
                            "fallback_notices": [],
                            "executions": [],
                        })
                
                # 执行工具
                tool_info = self._execute_tool_calls(current_text, model_id=model_id)
            
            if not tool_info["text"]:
                return current_text
            
            tool_round += 1
            
            # 构建 followup prompt
            followup_prompt = self._build_tool_followup_prompt(
                previous_analysis, tool_info, tool_round
            )
            
            # 发送 tool_executing 事件
            has_error = tool_info["has_error"]
            errors = tool_info["errors"]
            fallback_notices = list(tool_info["fallback_notices"])
            
            if has_error and not fallback_notices:
                status_msg = f"⚠️ 工具全部失败 ({len(errors)} 个错误)"
            elif fallback_notices:
                status_msg = f"🔄 工具回退: {'; '.join(fallback_notices)}"
            else:
                status_msg = "✅ 工具执行完成"
            
            event_queue.put({
                "type": "tool_executing",
                "model_id": model_id,
                "role_id": role_id,
                "message": status_msg,
                "tool_results": tool_info["text"],
                "has_error": has_error,
                "errors": errors,
                "fallback_notices": fallback_notices,
                "executions": tool_info.get("executions", []),
            })
            
            # 记录 prompt
            log_prompt(
                session_logger,
                f"TOOL_CHAIN R{tool_round} | role={role_id} | model={model_id}",
                system_prompt,
                followup_prompt,
            )
            
            # 流式调用下一轮
            next_text = ""
            try:
                for token, accumulated in self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=followup_prompt,
                ):
                    next_text = accumulated
                    event_queue.put({
                        "type": "token",
                        "model_id": model_id,
                        "role_id": role_id,
                        "model_name": model_name,
                        "token": token,
                        "accumulated": accumulated,
                    })
            except RuntimeError as e:
                if "high risk" in str(e).lower():
                    llm_logger.warning(
                        f"[_run_tool_chain_loop] model={model_id} 第{tool_round}轮因内容安全被拒绝，"
                        f"使用简化 prompt 重试"
                    )
                    simple_followup = self._build_simple_followup_prompt(previous_analysis)
                    log_prompt(
                        session_logger,
                        f"TOOL_CHAIN R{tool_round} SIMPLE | role={role_id} | model={model_id}",
                        system_prompt,
                        simple_followup,
                    )
                    for token, accumulated in self._client.call_stream(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=simple_followup,
                    ):
                        next_text = accumulated
                        event_queue.put({
                            "type": "token",
                            "model_id": model_id,
                            "role_id": role_id,
                            "model_name": model_name,
                            "token": token,
                            "accumulated": accumulated,
                        })
                else:
                    raise
            
            # 空输出防护
            if not next_text or not next_text.strip():
                llm_logger.warning(
                    f"[_run_tool_chain_loop] model={model_id} 第{tool_round}轮输出为空，"
                    f"回退到上一轮输出 | prev_len={len(current_text)}"
                )
                next_text = current_text
            
            # 记录响应
            log_response(
                session_logger,
                f"TOOL_CHAIN R{tool_round} | role={role_id} | model={model_id}",
                next_text,
            )
            
            # 检查是否还有 tool_call
            has_tc_next = executor.has_tool_calls(next_text)
            llm_logger.info(
                f"[_run_tool_chain_loop] model={model_id} tool_round={tool_round} 完成 | "
                f"has_tool_calls={has_tc_next} | output_len={len(next_text)}"
            )
            
            if not has_tc_next:
                return next_text
            
            # 如果上一轮只输出工具，本轮强制禁止——但模型仍然输出工具，直接返回（不执行，不循环）
            if previous_analysis == "（你之前只调用了工具，尚未给出分析）":
                llm_logger.warning(
                    f"[_run_tool_chain_loop] model={model_id} 强制禁止工具但模型仍输出tool_call，"
                    f"直接返回当前文本不再循环 | output_len={len(next_text)}"
                )
                return next_text
            
            if tool_round >= self.MAX_TOOL_ROUNDS:
                # 超过最大轮次，报错
                error_msg = (
                    f"模型连续 {self.MAX_TOOL_ROUNDS} 轮输出了 tool call，"
                    f"无法生成最终分析。"
                )
                llm_logger.error(
                    f"[_run_tool_chain_loop] model={model_id} {error_msg}"
                )
                event_queue.put({
                    "type": "error",
                    "model_id": model_id,
                    "role_id": role_id,
                    "error": error_msg,
                })
                return None
            
            # 继续下一轮：更新 current_text 和 previous_analysis
            current_text = next_text
            previous_analysis = re.sub(
                r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                '', current_text, flags=re.DOTALL
            ).strip()
            if not previous_analysis:
                previous_analysis = "（你之前只调用了工具，尚未给出分析）"
    
    # ==================== 工具方法 ====================
    
    def _extract_thinking(self, text: str) -> tuple:
        """从回答中提取 thinking 过程和正式输出
        
        同时移除 <tool_call> 和 <tool_result> 标签，避免它们出现在最终输出中。
        """
        match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            # 移除 thinking 标签后的正式输出
            formal = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
        else:
            thinking = ""
            formal = text
        
        # 移除 tool_call 和 tool_result 标签（避免出现在最终输出中）
        formal = re.sub(r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>', '', formal, flags=re.DOTALL).strip()
        formal = re.sub(r'<tool_result\s+name="[^"]+">\s*.*?\s*</tool_result>', '', formal, flags=re.DOTALL).strip()
        
        return thinking, formal
    
    def _extract_confidence(self, text: str) -> float:
        """从回答中提取置信度
        
        如果检测到工具失败导致的数据缺失关键词，强制降低置信度，
        防止模型基于不完整信息给出高置信度结论。
        """
        # 检测工具失败导致的数据缺失——强制降权
        data_gaps = [
            "工具失败", "数据缺失", "无法获取", "无法验证", "未经工具验证",
            "缺少关键数据", "缺少验证", "无法确认", "信息不足",
            "工具调用失败", "搜索失败", "未找到相关结果",
        ]
        text_lower = text.lower()
        has_data_gap = any(kw in text_lower for kw in data_gaps)
        
        patterns = [
            r'置信度[^\d]*(\d+(?:\.\d+)?)\s*%',
            r'(?:confidence|Confidence)[^\d]*(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                confidence = min(max(val / 100.0, 0.0), 1.0)
                # 如果检测到数据缺失，强制不超过 50%
                if has_data_gap:
                    confidence = min(confidence, 0.5)
                return confidence
        
        # 默认置信度：有数据缺失时更低
        return 0.3 if has_data_gap else 0.7
    
    def _extract_summary(self, text: str) -> str:
        """从回答中提取 <summary> 总结块"""
        match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _maybe_summarize(self) -> str:
        """获取 memory context 用于 prompt 注入
        
        从 MemoryManager 读取骨架摘要 + 最近2轮完整记录
        """
        if self.memory_manager is None:
            return ""
        return self.memory_manager.get_context_for_prompt()
    
    def _get_model_display_name(self, model_id: str) -> str:
        """获取模型显示名"""
        cfg = self._config.models.get(model_id)
        return cfg.name if cfg else model_id
    
    def _build_critique_history_text(self, target_role_id: str, current_round: int) -> str:
        """构建某角色的历史评审文本（用于 revise prompt 注入）
        
        收集 current_round 之前的所有轮次中，其他角色对 target_role_id 的评审。
        每条评审截取前 200 字，避免过长。
        """
        lines = []
        for round_num in sorted(self.critiques_history.keys()):
            if round_num >= current_round:
                continue
            round_data = self.critiques_history[round_num]
            round_lines = []
            for critiquer_id, targets in round_data.items():
                if target_role_id in targets:
                    critiquer_mid = self.role_models.get(critiquer_id, critiquer_id)
                    critiquer_name = self._get_model_display_name(critiquer_mid)
                    text = targets[target_role_id]
                    # 截取前 200 字，避免单条过长
                    if len(text) > 200:
                        text = text[:200] + "..."
                    round_lines.append(f"- {critiquer_name}: {text}")
            if round_lines:
                lines.append(f"第{round_num}轮:\n" + "\n".join(round_lines))
        
        return "\n\n".join(lines) if lines else ""
    
    def _build_all_critique_history_text(self, current_round: int) -> str:
        """构建全局历史评审摘要（用于 critique prompt 注入）
        
        收集 current_round 之前的所有轮次中，所有角色间的评审互动。
        每轮只保留核心结论（前 150 字）。
        """
        lines = []
        for round_num in sorted(self.critiques_history.keys()):
            if round_num >= current_round:
                continue
            round_data = self.critiques_history[round_num]
            round_lines = []
            for critiquer_id, targets in round_data.items():
                critiquer_mid = self.role_models.get(critiquer_id, critiquer_id)
                critiquer_name = self._get_model_display_name(critiquer_mid)
                for target_id, text in targets.items():
                    target_mid = self.role_models.get(target_id, target_id)
                    target_name = self._get_model_display_name(target_mid)
                    if len(text) > 150:
                        text = text[:150] + "..."
                    round_lines.append(f"- {critiquer_name} → {target_name}: {text}")
            if round_lines:
                lines.append(f"第{round_num}轮:\n" + "\n".join(round_lines))
        
        return "\n\n".join(lines) if lines else ""
    
    def _build_calls(self, prompt_builder) -> List[Dict[str, Any]]:
        """为所有角色构建并行调用配置"""
        calls = []
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        for role_id, model_id in self.role_models.items():
            model_cfg = proposers.get(model_id)
            model_name = model_cfg.name if model_cfg else model_id
            user_prompt = prompt_builder(role_id, model_id, model_name)
            calls.append({
                "call_id": role_id,
                "model_id": model_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            })
        return calls
    
    # ==================== 阶段推进 ====================
    
    def generate(self) -> Dict[str, Any]:
        """第1轮：并行生成初始回答（同步）"""
        self.stage = Stage.GENERATING
        self.current_round = 1
        
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        def build_prompt(role_id, model_id, model_name):
            return prompts.build_generate_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                round_num=1,
                debater_role=role_id,
            )
        
        calls = self._build_calls(build_prompt)
        responses = self._client.call_parallel(calls)
        
        for role_id, resp in responses.items():
            thinking, formal = self._extract_thinking(resp.content)
            self.thinkings[role_id] = thinking
            self.answers[role_id] = formal
            self.confidences[role_id] = self._extract_confidence(resp.content)
            self.previous_answers[role_id] = resp.content
        
        # === Tool call 处理（支持最多 MAX_TOOL_ROUNDS 轮）===
        for role_id, model_id in list(self.role_models.items()):
            tool_info = self._execute_tool_calls(self.answers[role_id], model_id=model_id)
            if not tool_info["text"]:
                llm_logger.info(f"[generate] model={model_id} 无 tool call，跳过")
                continue
            
            has_error = tool_info["has_error"]
            errors = tool_info["errors"]
            fallback_notices = tool_info["fallback_notices"]
            llm_logger.info(
                f"[generate] model={model_id} tool_round=1 执行完成 | "
                f"has_error={has_error} | errors={len(errors)} | "
                f"fallback_notices={len(fallback_notices)}"
            )
            
            first_round_text = self.previous_answers.get(role_id, "")
            clean_first_round = re.sub(
                r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                '', first_round_text, flags=re.DOTALL
            ).strip()
            if not clean_first_round:
                clean_first_round = "（你之前只调用了工具，尚未给出分析）"
            
            current_text = first_round_text
            previous_analysis = clean_first_round
            tool_round = 1
            full_text = first_round_text
            
            while True:
                if tool_round > 1:
                    tool_info = self._execute_tool_calls(current_text, model_id=model_id)
                    if not tool_info["text"]:
                        full_text = current_text
                        break
                
                tool_round += 1
                
                followup_prompt = self._build_tool_followup_prompt(
                    previous_analysis, tool_info, tool_round
                )
                
                llm_logger.info(f"[generate] model={model_id} 开始 tool_round={tool_round} | prompt_len={len(followup_prompt)}")
                resp = self._client.call(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=followup_prompt,
                )
                next_text = resp.content
                
                has_tc_next = self._tool_registry.create_executor().has_tool_calls(next_text)
                llm_logger.info(f"[generate] model={model_id} tool_round={tool_round} 完成 | has_tool_calls={has_tc_next}")
                
                if not has_tc_next:
                    full_text = next_text
                    break
                
                # 紧急断路器：如果上一轮已经只输出工具（previous_analysis 是占位符），
                # 这一轮仍然输出工具 → 停止循环，保留当前输出
                if previous_analysis == "（你之前只调用了工具，尚未给出分析）":
                    llm_logger.error(
                        f"[generate] model={model_id} 第{tool_round}轮在纯工具输出后又输出工具，"
                        f"紧急停止，保留当前输出"
                    )
                    full_text = next_text
                    break
                
                if tool_round >= self.MAX_TOOL_ROUNDS:
                    llm_logger.error(
                        f"[generate] model={model_id} 连续{self.MAX_TOOL_ROUNDS}轮输出 tool call，"
                        f"跳过该模型，保留第一轮结果"
                    )
                    full_text = first_round_text
                    break
                
                current_text = next_text
                previous_analysis = re.sub(
                    r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                    '', current_text, flags=re.DOTALL
                ).strip()
                if not previous_analysis:
                    previous_analysis = "（你之前只调用了工具，尚未给出分析）"
            
            thinking, formal = self._extract_thinking(full_text)
            self.thinkings[role_id] = thinking
            self.answers[role_id] = formal
            self.confidences[role_id] = self._extract_confidence(full_text)
            self.summaries[role_id] = self._extract_summary(full_text)
            self.previous_answers[role_id] = full_text
        
        self.stage = Stage.PAUSE_AFTER_GENERATE
        return self._build_ui_state()
    
    def brainstorm_stream_parallel(self) -> Generator[Dict[str, Any], None, None]:
        """并行流式头脑风暴：分析问题、识别缺口、向用户提问"""
        
        self.stage = Stage.INIT  # 用 INIT 表示 brainstorm 阶段
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        event_queue = queue.Queue()
        
        def stream_single(role_id, model_id):
            try:
                model_name = proposers[model_id].name
                skill_context = self._get_skill_context()
                tool_context = self._get_tool_context()
                
                base_prompt = prompts.build_brainstorm_prompt(
                    question=self.question,
                    context=self.context,
                    debater_role=role_id,
                )
                # 注入 skill 和 tool context
                extra = ""
                if skill_context:
                    extra += f"\n\n{skill_context}"
                if tool_context:
                    extra += f"\n\n{tool_context}"
                user_prompt = base_prompt + extra if extra else base_prompt
                
                # ========== 第一轮 ==========
                full_text = ""
                first_round_text = ""
                for token, accumulated in self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ):
                    full_text = accumulated
                    first_round_text = accumulated
                    event_queue.put({
                        "type": "token",
                        "model_id": model_id,
                        "role_id": role_id,
                        "model_name": model_name,
                        "token": token,
                        "accumulated": accumulated,
                    })
                
                # ========== 工具调用（与 generate_stream_parallel 保持一致）==========
                executor = self._tool_registry.create_executor()
                has_tc_1 = executor.has_tool_calls(full_text)
                llm_logger.info(
                    f"[brainstorm_stream_parallel] model={model_id} tool_round=1 完成 | "
                    f"has_tool_calls={has_tc_1} | output_len={len(full_text)}"
                )
                
                tool_info = None
                fallback_search = False
                
                if has_tc_1:
                    detected = list(executor.TOOL_CALL_PATTERN.finditer(full_text))
                    tool_names = [m.group(1) for m in detected]
                    llm_logger.info(
                        f"[brainstorm_stream_parallel] model={model_id} 检测到 {len(detected)} 个工具调用 | "
                        f"tools={tool_names}"
                    )
                    event_queue.put({
                        "type": "tool_executing",
                        "model_id": model_id,
                        "role_id": role_id,
                        "message": f"🔧 检测到 {len(detected)} 个工具调用，正在并行执行...",
                        "has_error": False,
                        "errors": [],
                        "fallback_notices": [],
                        "executions": [],
                    })
                    tool_info = self._execute_tool_calls(full_text, model_id=model_id)
                elif full_text:
                    # 辅助诊断：模型是否用自然语言表达了工具意图但未输出 XML
                    intent_keywords = ["搜索", "查询", "查一下", "获取数据", "用工具", "调用工具", "事实核查"]
                    if any(kw in full_text for kw in intent_keywords):
                        llm_logger.warning(
                            f"[brainstorm_stream_parallel] model={model_id} 检测到工具意图关键词但无 XML 标签，"
                            f"触发系统回退搜索 | output_preview={full_text[:200]!r}"
                        )
                        fallback_search = True
                        import html
                        fallback_query = self.question[:100]
                        safe_query = html.escape(fallback_query, quote=False)
                        fallback_text = f'<tool_call name="web_search"><query>{safe_query}</query></tool_call>'
                        tool_info = self._execute_tool_calls(fallback_text, model_id=model_id)
                        event_queue.put({
                            "type": "tool_executing",
                            "model_id": model_id,
                            "role_id": role_id,
                            "message": "🔄 模型未输出工具标签，系统自动执行回退搜索",
                            "has_error": False,
                            "errors": [],
                            "fallback_notices": ["模型未按格式输出工具标签，系统自动搜索补充数据"],
                            "executions": [],
                        })
                
                # ========== 工具链循环（支持最多 MAX_TOOL_ROUNDS 轮）==========
                if tool_info and tool_info["text"]:
                    has_error = tool_info["has_error"]
                    errors = tool_info["errors"]
                    fallback_notices = list(tool_info["fallback_notices"])
                    if fallback_search:
                        fallback_notices.append("【系统回退】模型未按格式输出工具标签，系统自动搜索补充数据")
                    
                    llm_logger.info(
                        f"[brainstorm_stream_parallel] model={model_id} tool_round=1 执行完成 | "
                        f"has_error={has_error} | errors={len(errors)}"
                    )
                    
                    # 修正 tool_info 中的 fallback_notices
                    tool_info = {**tool_info, "fallback_notices": fallback_notices}
                    
                    # 使用工具链循环处理后续轮次
                    full_text = self._run_tool_chain_loop(
                        role_id=role_id,
                        model_id=model_id,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        first_round_text=first_round_text,
                        event_queue=event_queue,
                        initial_tool_info=tool_info,
                    )
                    if full_text is None:
                        return  # 失败，直接退出
                
                thinking, formal = self._extract_thinking(full_text)
                self.thinkings[role_id] = thinking
                self.answers[role_id] = formal  # 用 answers 存 brainstorm 结果
                self.summaries[role_id] = self._extract_summary(full_text)
                
                event_queue.put({
                    "type": "model_done",
                    "role_id": role_id,
                    "model_id": model_id,
                    "full_text": full_text,
                })
                
            except Exception as e:
                tb = traceback.format_exc()
                llm_logger.error(
                    f"[brainstorm_stream_parallel] model={model_id} CRASH\n"
                    f"Error: {e}\n{tb}"
                )
                event_queue.put({
                    "type": "error",
                    "model_id": model_id,
                    "role_id": role_id,
                    "error": f"{e}",
                })
        
        threads = []
        for role_id, model_id in self.role_models.items():
            event_queue.put({
                "type": "start",
                "model_id": model_id,
                "role_id": role_id,
                "model_name": proposers[model_id].name,
            })
            t = threading.Thread(target=stream_single, args=(role_id, model_id), daemon=True)
            t.start()
            threads.append(t)
        thread_infos = list(self.role_models.items())
        
        done_count = 0
        errors = []
        timed_out = set()
        while done_count < len(thread_infos):
            try:
                event = event_queue.get(timeout=300)
            except queue.Empty:
                for idx, (rid, mid) in enumerate(thread_infos):
                    t = threads[idx]
                    if rid in timed_out:
                        continue
                    if t.is_alive():
                        timed_out.add(rid)
                        err_msg = "线程超时无响应（可能API连接卡住或工具执行耗时过长）"
                        llm_logger.error(f"[brainstorm_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                    else:
                        # 线程在 prompt 构建阶段就崩溃，从未进入 try 块
                        timed_out.add(rid)
                        err_msg = "线程异常退出（prompt构建或API初始化阶段崩溃，详见日志）"
                        llm_logger.error(f"[brainstorm_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                if done_count >= len(thread_infos):
                    break
                continue
            
            yield event
            rid = event.get("role_id", "")
            if event["type"] == "model_done" and rid not in timed_out:
                done_count += 1
            elif event["type"] == "error" and rid not in timed_out:
                done_count += 1
                errors.append(f"{rid}: {event['error']}")
        
        # 超时后 drain 剩余事件，避免 tool_executing/model_done 被吞
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                yield event
            except queue.Empty:
                break
        
        if errors:
            yield {"type": "partial_error", "errors": errors}
    
    def _update_memory(self, round_type: str):
        """更新 Memory 系统
        
        每5轮更新 working memory，每10轮落地 stage memory
        """
        if self.memory_manager is None:
            return
        
        # 收集最近5轮的数据
        history = getattr(self, '_round_history', [])
        
        # 构建当前轮次数据（动态支持任意模型）
        round_data = {
            "round": self.current_round,
            "type": round_type,
        }
        for role_id, model_id in self.role_models.items():
            safe_id = model_id.replace(".", "_")
            round_data[f"{safe_id}_thinking"] = self.thinkings.get(role_id, "")
            round_data[f"{safe_id}_output"] = self.answers.get(role_id, "")
            round_data[f"{safe_id}_conf"] = self.confidences.get(role_id, 0)
        
        # 保存到 _round_history（用于 memory 更新）
        if not hasattr(self, '_round_history'):
            self._round_history = []
        self._round_history.append(round_data)
        
        # 检查是否需要更新 working memory
        if self.memory_manager.should_update(self.current_round):
            # 取最近5轮
            recent = self._round_history[-self.memory_manager.UPDATE_INTERVAL:]
            self.memory_manager.update_working_memory(recent)
        
        # 检查是否需要落地 stage memory
        if self.memory_manager.should_save_stage(self.current_round):
            self.memory_manager.save_stage_memory(self.current_round)
    
    def finish_debate(self) -> str:
        """结束辩论，生成 Obsidian 文档
        
        Returns:
            生成的 Obsidian 文件路径
        """
        if self.memory_manager:
            return self.memory_manager.generate_obsidian(self)
        return ""
    
    def generate_stream_parallel(self) -> Generator[Dict[str, Any], None, None]:
        """并行流式生成"""
        
        self.stage = Stage.GENERATING
        if self.current_round == 0:
            self.current_round = 1
        else:
            self.current_round += 1
        
        session_logger.info(
            f"========== ROUND {self.current_round} GENERATE START =========="
        )
        session_logger.info(f"question={self.question!r}")
        session_logger.info(f"scenario={self.scenario_type}")
        session_logger.info(f"role_models={dict(self.role_models)}")
        session_logger.info(f"context_len={len(self.context)}")
        
        memory_context = self._maybe_summarize()
        
        proposers = self._config.get_proposer_models()
        base_system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        # 把工具说明注入 system_prompt，模型对 system 的遵循度通常更高
        system_prompt = base_system_prompt
        if tool_context.strip():
            system_prompt += f"\n\n{tool_context}"
        
        def build_prompt(role_id, model_id, model_name):
            # tool_context 已在 system_prompt 中，user prompt 中不再重复工具列表
            # 但保留 tool_usage_strategy（❌/✅ 正误示例），这是 DeepSeek 遵循 XML 格式的关键
            # 注意：直接用 role_id，不要通过 _role_of_model(model_id) 反查（同一模型双角色时会返回错误角色）
            return prompts.build_generate_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                round_num=self.current_round,
                memory_context=memory_context,
                skill_context=skill_context,
                tool_context="",  # 已在 system_prompt 中，避免重复
                debater_role=role_id,
                include_tool_strategy=True,
            )
        
        event_queue = queue.Queue()
        
        def stream_single(role_id, model_id):
            try:
                model_name = proposers[model_id].name
                user_prompt = build_prompt(role_id, model_id, model_name)
                
                # 打印本轮该角色的完整 prompt
                log_prompt(
                    session_logger,
                    f"ROUND {self.current_round} | role={role_id} | model={model_id}",
                    system_prompt,
                    user_prompt,
                )
                
                full_text = ""
                for token, accumulated in self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ):
                    full_text = accumulated
                    event_queue.put({
                        "type": "token",
                        "model_id": model_id,
                        "role_id": role_id,
                        "model_name": model_name,
                        "token": token,
                        "accumulated": accumulated,
                    })
                
                # 保存第一轮完整输出（用于第二轮 prompt 中保留上下文）
                first_round_text = full_text
                
                # 打印第一轮响应
                log_response(
                    session_logger,
                    f"ROUND {self.current_round} R1 | role={role_id} | model={model_id}",
                    full_text,
                )
                
                # 检查并执行 tool calls（如果有）
                executor = self._tool_registry.create_executor()
                has_tc_1 = executor.has_tool_calls(full_text)
                log_event(
                    session_logger,
                    "TOOL_CHECK_R1",
                    role_id=role_id,
                    model_id=model_id,
                    has_tool_calls=has_tc_1,
                    output_len=len(full_text),
                )
                llm_logger.info(
                    f"[generate_stream_parallel] model={model_id} tool_round=1 完成 | "
                    f"has_tool_calls={has_tc_1} | output_len={len(full_text)}"
                )
                
                tool_info = None
                fallback_search = False
                
                if has_tc_1:
                    # 正常工具调用流程
                    detected = list(executor.TOOL_CALL_PATTERN.finditer(full_text))
                    tool_names = [m.group(1) for m in detected]
                    llm_logger.info(
                        f"[generate_stream_parallel] model={model_id} 检测到 {len(detected)} 个工具调用 | "
                        f"tools={tool_names}"
                    )
                    event_queue.put({
                        "type": "tool_executing",
                        "model_id": model_id,
                        "role_id": role_id,
                        "message": f"🔧 检测到 {len(detected)} 个工具调用，正在并行执行...",
                        "has_error": False,
                        "errors": [],
                        "fallback_notices": [],
                        "executions": [],
                    })
                    tool_info = self._execute_tool_calls(full_text, model_id=model_id)
                elif full_text:
                    # 辅助诊断：模型是否用自然语言表达了工具意图但未输出 XML
                    intent_keywords = ["搜索", "查询", "查一下", "获取数据", "用工具", "调用工具", "事实核查"]
                    if any(kw in full_text for kw in intent_keywords):
                        llm_logger.warning(
                            f"[generate_stream_parallel] model={model_id} 检测到工具意图关键词但无 XML 标签，"
                            f"触发系统回退搜索 | output_preview={full_text[:200]!r}"
                        )
                        fallback_search = True
                        # 构造回退搜索：用问题的前 100 字作为查询
                        fallback_query = self.question[:100]
                        import html
                        safe_query = html.escape(fallback_query, quote=False)
                        fallback_text = f'<tool_call name="web_search"><query>{safe_query}</query></tool_call>'
                        tool_info = self._execute_tool_calls(fallback_text, model_id=model_id)
                        event_queue.put({
                            "type": "tool_executing",
                            "model_id": model_id,
                            "role_id": role_id,
                            "message": "🔄 模型未输出工具标签，系统自动执行回退搜索",
                            "has_error": False,
                            "errors": [],
                            "fallback_notices": ["模型未按格式输出工具标签，系统自动搜索补充数据"],
                            "executions": [],
                        })
                
                if tool_info and tool_info["text"]:
                    has_error = tool_info["has_error"]
                    errors = tool_info["errors"]
                    fallback_notices = list(tool_info["fallback_notices"])
                    if fallback_search:
                        fallback_notices.append("【系统回退】模型未按格式输出工具标签，系统自动搜索补充数据")
                    
                    log_event(
                        session_logger,
                        "TOOL_EXEC_DONE",
                        role_id=role_id,
                        model_id=model_id,
                        fallback_search=fallback_search,
                        has_error=has_error,
                        errors=len(errors),
                        fallback_notices=len(fallback_notices),
                        tool_results_len=len(tool_info["text"]),
                    )
                    llm_logger.info(
                        f"[generate_stream_parallel] model={model_id} tool_round=1 执行完成 | "
                        f"has_error={has_error} | errors={len(errors)} | "
                        f"fallback_notices={len(fallback_notices)}"
                    )
                    
                    # 修正 tool_info 中的 fallback_notices
                    tool_info = {**tool_info, "fallback_notices": fallback_notices}
                    
                    # 使用工具链循环处理后续轮次（支持最多 MAX_TOOL_ROUNDS 轮）
                    full_text = self._run_tool_chain_loop(
                        role_id=role_id,
                        model_id=model_id,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        first_round_text=first_round_text,
                        event_queue=event_queue,
                        initial_tool_info=tool_info,
                    )
                    if full_text is None:
                        return  # 失败，直接退出
                
                thinking, formal = self._extract_thinking(full_text)
                self.thinkings[role_id] = thinking
                self.answers[role_id] = formal
                self.confidences[role_id] = self._extract_confidence(full_text)
                self.summaries[role_id] = self._extract_summary(full_text)
                self.previous_answers[role_id] = full_text
                
                log_event(
                    session_logger,
                    "MODEL_DONE",
                    role_id=role_id,
                    model_id=model_id,
                    final_text_len=len(full_text),
                    thinking_len=len(thinking),
                    formal_len=len(formal),
                    confidence=self.confidences[role_id],
                )
                
                event_queue.put({
                    "type": "model_done",
                    "role_id": role_id,
                    "model_id": model_id,
                    "full_text": full_text,
                })
                
            except Exception as e:
                tb = traceback.format_exc()
                llm_logger.error(
                    f"[generate_stream_parallel] model={model_id} CRASH\n"
                    f"Error: {e}\n{tb}"
                )
                session_logger.error(
                    f"[ERROR] role={role_id} model={model_id} CRASH | {e}\n{tb}"
                )
                event_queue.put({
                    "type": "error",
                    "model_id": model_id,
                    "role_id": role_id,
                    "error": f"{e}",
                })
        
        threads = []
        for role_id, model_id in self.role_models.items():
            event_queue.put({
                "type": "start",
                "model_id": model_id,
                "role_id": role_id,
                "model_name": proposers[model_id].name,
            })
            t = threading.Thread(target=stream_single, args=(role_id, model_id), daemon=True)
            t.start()
            threads.append(t)
        thread_infos = list(self.role_models.items())
        
        done_count = 0
        errors = []
        timed_out = set()
        while done_count < len(thread_infos):
            try:
                event = event_queue.get(timeout=300)
            except queue.Empty:
                for idx, (rid, mid) in enumerate(thread_infos):
                    t = threads[idx]
                    if rid in timed_out:
                        continue
                    if t.is_alive():
                        timed_out.add(rid)
                        err_msg = "线程超时无响应（可能API连接卡住或工具执行耗时过长）"
                        llm_logger.error(f"[generate_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                    else:
                        timed_out.add(rid)
                        err_msg = "线程异常退出（prompt构建或API初始化阶段崩溃，详见日志）"
                        llm_logger.error(f"[generate_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                if done_count >= len(thread_infos):
                    break
                continue
            
            yield event
            rid = event.get("role_id", "")
            if event["type"] == "model_done" and rid not in timed_out:
                done_count += 1
            elif event["type"] == "error" and rid not in timed_out:
                done_count += 1
                errors.append(f"{rid}: {event['error']}")
        
        # 超时后 drain 剩余事件，避免 tool_executing/model_done 被吞
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                yield event
            except queue.Empty:
                break
        
        if errors:
            yield {"type": "partial_error", "errors": errors}
        
        self.stage = Stage.PAUSE_AFTER_GENERATE
        
        session_logger.info(
            f"========== ROUND {self.current_round} GENERATE END =========="
        )
        session_logger.info(f"final_status: stage={self.stage} errors={errors}")
        for rid, mid in self.role_models.items():
            session_logger.info(
                f"  role={rid} model={mid} "
                f"answer_len={len(self.answers.get(rid, ''))} "
                f"confidence={self.confidences.get(rid, 0)}"
            )
        
        # 更新 Memory
        self._update_memory("generate")
    
    def generate_stream(self) -> Generator[Dict[str, Any], None, None]:
        """第1轮：流式生成初始回答
        
        Yields:
            事件字典，包含 type 字段：
            - "start": 开始调用某个模型
            - "token": 收到一个 token
            - "model_done": 某个模型完成
            - "done": 所有模型完成
        """
        self.stage = Stage.GENERATING
        self.current_round = 1
        
        def build_prompt(role_id, model_id, model_name):
            return prompts.build_generate_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                round_num=1,
                debater_role=role_id,
            )
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        for role_id, model_id in self.role_models.items():
            model_name = proposers[model_id].name
            user_prompt = build_prompt(role_id, model_id, model_name)
            
            yield {"type": "start", "model_id": model_id, "role_id": role_id, "model_name": model_name}
            
            full_text = ""
            try:
                stream_gen = self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                
                for token, accumulated in stream_gen:
                    full_text = accumulated
                    yield {"type": "token", "model_id": model_id, "token": token, "accumulated": accumulated}
                
                # === 工具链循环（支持最多 MAX_TOOL_ROUNDS 轮）===
                first_round_text = full_text
                tool_info = self._execute_tool_calls(full_text, model_id=model_id)
                if tool_info["text"]:
                    clean_first_round = re.sub(
                        r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                        '', first_round_text, flags=re.DOTALL
                    ).strip()
                    if not clean_first_round:
                        clean_first_round = "（你之前只调用了工具，尚未给出分析）"
                    
                    current_text = first_round_text
                    previous_analysis = clean_first_round
                    tool_round = 1
                    
                    while True:
                        if tool_round > 1:
                            tool_info = self._execute_tool_calls(current_text, model_id=model_id)
                            if not tool_info["text"]:
                                full_text = current_text
                                break
                        
                        tool_round += 1
                        
                        followup_prompt = self._build_tool_followup_prompt(
                            previous_analysis, tool_info, tool_round
                        )
                        
                        llm_logger.info(
                            f"[generate_stream] model={model_id} 开始 tool_round={tool_round} | "
                            f"prompt_len={len(followup_prompt)}"
                        )
                        
                        next_text = ""
                        try:
                            for token, accumulated in self._client.call_stream(
                                model_id=model_id,
                                system_prompt=system_prompt,
                                user_prompt=followup_prompt,
                            ):
                                next_text = accumulated
                                yield {"type": "token", "model_id": model_id, "token": token, "accumulated": accumulated}
                        except RuntimeError as e:
                            if "high risk" in str(e).lower():
                                llm_logger.warning(
                                    f"[generate_stream] model={model_id} 第{tool_round}轮因内容安全被拒绝，"
                                    f"使用简化 prompt 重试"
                                )
                                simple_followup = self._build_simple_followup_prompt(previous_analysis)
                                for token, accumulated in self._client.call_stream(
                                    model_id=model_id,
                                    system_prompt=system_prompt,
                                    user_prompt=simple_followup,
                                ):
                                    next_text = accumulated
                                    yield {"type": "token", "model_id": model_id, "token": token, "accumulated": accumulated}
                            else:
                                raise
                        
                        # 空输出防护
                        if not next_text or not next_text.strip():
                            llm_logger.warning(
                                f"[generate_stream] model={model_id} 第{tool_round}轮输出为空，"
                                f"回退到上一轮输出"
                            )
                            next_text = current_text
                        
                        has_tc_next = self._tool_registry.create_executor().has_tool_calls(next_text)
                        llm_logger.info(
                            f"[generate_stream] model={model_id} tool_round={tool_round} 完成 | "
                            f"has_tool_calls={has_tc_next} | output_len={len(next_text)}"
                        )
                        
                        if not has_tc_next:
                            full_text = next_text
                            break
                        
                        # 紧急断路器：如果上一轮已经只输出工具，这一轮仍然输出工具 → 停止循环
                        if previous_analysis == "（你之前只调用了工具，尚未给出分析）":
                            llm_logger.error(
                                f"[generate_stream] model={model_id} 第{tool_round}轮在纯工具输出后又输出工具，"
                                f"紧急停止，保留当前输出"
                            )
                            full_text = next_text
                            break
                        
                        if tool_round >= self.MAX_TOOL_ROUNDS:
                            err_msg = "模型连续多轮输出了 tool call，无法生成最终分析。"
                            llm_logger.error(f"[generate_stream] model={model_id} {err_msg}")
                            yield {"type": "error", "model_id": model_id, "error": err_msg}
                            break
                        
                        current_text = next_text
                        previous_analysis = re.sub(
                            r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                            '', current_text, flags=re.DOTALL
                        ).strip()
                        if not previous_analysis:
                            previous_analysis = "（你之前只调用了工具，尚未给出分析）"
                
                # 流式结束后解析
                thinking, formal = self._extract_thinking(full_text)
                self.thinkings[role_id] = thinking
                self.answers[role_id] = formal
                self.confidences[role_id] = self._extract_confidence(full_text)
                self.summaries[role_id] = self._extract_summary(full_text)
                self.previous_answers[role_id] = full_text
                
                yield {"type": "model_done", "role_id": role_id, "model_id": model_id, "full_text": full_text}
                
            except Exception as e:
                tb = traceback.format_exc()
                llm_logger.error(
                    f"[generate_stream] model={model_id} CRASH\n"
                    f"Error: {e}\n{tb}"
                )
                yield {"type": "error", "model_id": model_id, "error": f"{e}"}
                # 不再 raise，让其他模型继续
        
        self.stage = Stage.PAUSE_AFTER_GENERATE
    
    def critique_stream_parallel(self) -> Generator[Dict[str, Any], None, None]:
        """并行流式评审"""
        
        self.stage = Stage.CRITIQUING
        
        memory_context = self._maybe_summarize()
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        event_queue = queue.Queue()
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        critique_history = self._build_all_critique_history_text(self.current_round)
        
        def stream_single(role_id, model_id):
            try:
                other_answers = {
                    other_role: self.answers[other_role]
                    for other_role, other_mid in self.role_models.items()
                    if other_role != role_id
                }
                model_name = proposers[model_id].name
                
                user_prompt = prompts.build_critique_prompt(
                    scenario_type=self.scenario_type,
                    question=self.question,
                    context=self.context,
                    my_answer=self.answers[role_id],
                    other_answers=other_answers,
                    memory_context=memory_context,
                    critique_history=critique_history,
                    skill_context=skill_context,
                    tool_context=tool_context,
                    critiquer_role=role_id,
                )
                
                full_text = ""
                for token, accumulated in self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ):
                    full_text = accumulated
                    event_queue.put({
                        "type": "token",
                        "model_id": model_id,
                        "role_id": role_id,
                        "model_name": model_name,
                        "token": token,
                        "accumulated": accumulated,
                    })
                
                # 按轮次存储历史评审（以 role_id 为 key，支持同一模型多角色）
                if self.current_round not in self.critiques_history:
                    self.critiques_history[self.current_round] = {}
                self.critiques_history[self.current_round][role_id] = {}
                for other_role, other_mid in self.role_models.items():
                    if other_role == role_id:
                        continue
                    self.critiques_history[self.current_round][role_id][other_role] = full_text
                    # 快照（向后兼容）
                    self.critiques[role_id] = {other_role: full_text}
                
                event_queue.put({
                    "type": "model_done",
                    "role_id": role_id,
                    "model_id": model_id,
                    "full_text": full_text,
                })
                
            except Exception as e:
                tb = traceback.format_exc()
                llm_logger.error(
                    f"[critique_stream_parallel] model={model_id} CRASH\n"
                    f"Error: {e}\n{tb}"
                )
                event_queue.put({
                    "type": "error",
                    "model_id": model_id,
                    "role_id": role_id,
                    "error": f"{e}",
                })
        
        threads = []
        for role_id, model_id in self.role_models.items():
            event_queue.put({
                "type": "start",
                "model_id": model_id,
                "role_id": role_id,
                "model_name": proposers[model_id].name,
            })
            t = threading.Thread(target=stream_single, args=(role_id, model_id), daemon=True)
            t.start()
            threads.append(t)
        thread_infos = list(self.role_models.items())
        
        done_count = 0
        errors = []
        timed_out = set()
        while done_count < len(thread_infos):
            try:
                event = event_queue.get(timeout=300)
            except queue.Empty:
                for idx, (rid, mid) in enumerate(thread_infos):
                    t = threads[idx]
                    if rid in timed_out:
                        continue
                    if t.is_alive():
                        timed_out.add(rid)
                        err_msg = "线程超时无响应（可能API连接卡住或工具执行耗时过长）"
                        llm_logger.error(f"[critique_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                    else:
                        timed_out.add(rid)
                        err_msg = "线程异常退出（prompt构建或API初始化阶段崩溃，详见日志）"
                        llm_logger.error(f"[critique_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                if done_count >= len(thread_infos):
                    break
                continue
            
            yield event
            rid = event.get("role_id", "")
            if event["type"] == "model_done" and rid not in timed_out:
                done_count += 1
            elif event["type"] == "error" and rid not in timed_out:
                done_count += 1
                errors.append(f"{rid}: {event['error']}")
        
        # 超时后 drain 剩余事件，避免 tool_executing/model_done 被吞
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                yield event
            except queue.Empty:
                break
        
        if errors:
            yield {"type": "partial_error", "errors": errors}
        
        self.stage = Stage.PAUSE_AFTER_CRITIQUE
        
        # 更新 Memory
        self._update_memory("critique")
    
    def critique_stream_single(
        self, critic_role: str, target_role: str
    ) -> Generator[Dict[str, Any], None, None]:
        """串行流式评审：单个角色对单个目标的评审
        
        与 critique_stream_parallel 的区别：
        - 只运行一个 critic → target 的 critique，不启动线程
        - 用于严格串行交替辩论流程
        """
        self.stage = Stage.CRITIQUING
        
        memory_context = self._maybe_summarize()
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        critique_history = self._build_all_critique_history_text(self.current_round)
        
        model_id = self.role_models.get(critic_role)
        if not model_id:
            yield {"type": "error", "model_id": critic_role, "role_id": critic_role, "error": f"未知角色: {critic_role}"}
            return
        
        model_cfg = proposers.get(model_id)
        if not model_cfg:
            yield {"type": "error", "model_id": model_id, "role_id": critic_role, "error": f"未知模型: {model_id}"}
            return
        
        try:
            other_answers = {target_role: self.answers[target_role]}
            model_name = model_cfg.name
            
            user_prompt = prompts.build_critique_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                my_answer=self.answers[critic_role],
                other_answers=other_answers,
                memory_context=memory_context,
                critique_history=critique_history,
                skill_context=skill_context,
                tool_context=tool_context,
                critiquer_role=critic_role,
            )
            
            full_text = ""
            yield {"type": "start", "model_id": model_id, "role_id": critic_role, "model_name": model_name}
            
            for token, accumulated in self._client.call_stream(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                full_text = accumulated
                yield {
                    "type": "token",
                    "model_id": model_id,
                    "role_id": critic_role,
                    "model_name": model_name,
                    "token": token,
                    "accumulated": accumulated,
                }
            
            # 按轮次存储历史评审
            if self.current_round not in self.critiques_history:
                self.critiques_history[self.current_round] = {}
            if critic_role not in self.critiques_history[self.current_round]:
                self.critiques_history[self.current_round][critic_role] = {}
            self.critiques_history[self.current_round][critic_role][target_role] = full_text
            # 快照
            if critic_role not in self.critiques:
                self.critiques[critic_role] = {}
            self.critiques[critic_role][target_role] = full_text
            
            yield {"type": "model_done", "role_id": critic_role, "model_id": model_id, "full_text": full_text}
            
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(
                f"[critique_stream_single] model={model_id} CRASH\n"
                f"Error: {e}\n{tb}"
            )
            yield {"type": "error", "model_id": model_id, "role_id": critic_role, "error": f"{e}"}
        
        self.stage = Stage.PAUSE_AFTER_CRITIQUE
        self._update_memory("critique")
    
    def revise_stream_parallel(self, user_critique: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """并行流式修订"""
        
        self.stage = Stage.REVISING
        
        memory_context = self._maybe_summarize()
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        event_queue = queue.Queue()
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        
        def stream_single(role_id, model_id):
            try:
                critiques_on_me = {}
                for other_role, other_mid in self.role_models.items():
                    if other_role == role_id:
                        continue
                    if other_role in self.critiques and role_id in self.critiques[other_role]:
                        critiques_on_me[other_role] = self.critiques[other_role][role_id]
                    else:
                        critiques_on_me[other_role] = "（未收到明确评审意见）"
                
                if user_critique:
                    critiques_on_me["用户"] = f"【用户评判】\n{user_critique}"
                
                # 注入该模型的历史评审
                critique_history = self._build_critique_history_text(role_id, self.current_round)
                
                model_name = proposers[model_id].name
                user_prompt = prompts.build_revise_prompt(
                    scenario_type=self.scenario_type,
                    question=self.question,
                    context=self.context,
                    my_previous_answer=self.answers[role_id],
                    critiques_on_me=critiques_on_me,
                    memory_context=memory_context,
                    critique_history=critique_history,
                    skill_context=skill_context,
                    tool_context=tool_context,
                    my_role=role_id,
                )
                
                full_text = ""
                for token, accumulated in self._client.call_stream(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ):
                    full_text = accumulated
                    event_queue.put({
                        "type": "token",
                        "model_id": model_id,
                        "role_id": role_id,
                        "model_name": model_name,
                        "token": token,
                        "accumulated": accumulated,
                    })
                
                thinking, formal = self._extract_thinking(full_text)
                self.thinkings[role_id] = thinking
                self.answers[role_id] = formal
                self.confidences[role_id] = self._extract_confidence(full_text)
                self.summaries[role_id] = self._extract_summary(full_text)
                self.previous_answers[role_id] = full_text
                
                event_queue.put({
                    "type": "model_done",
                    "role_id": role_id,
                    "model_id": model_id,
                    "full_text": full_text,
                })
                
            except Exception as e:
                tb = traceback.format_exc()
                llm_logger.error(
                    f"[revise_stream_parallel] model={model_id} CRASH\n"
                    f"Error: {e}\n{tb}"
                )
                event_queue.put({
                    "type": "error",
                    "model_id": model_id,
                    "role_id": role_id,
                    "error": f"{e}",
                })
        
        threads = []
        for role_id, model_id in self.role_models.items():
            event_queue.put({
                "type": "start",
                "model_id": model_id,
                "role_id": role_id,
                "model_name": proposers[model_id].name,
            })
            t = threading.Thread(target=stream_single, args=(role_id, model_id), daemon=True)
            t.start()
            threads.append(t)
        thread_infos = list(self.role_models.items())
        
        done_count = 0
        errors = []
        timed_out = set()
        while done_count < len(thread_infos):
            try:
                event = event_queue.get(timeout=300)
            except queue.Empty:
                for idx, (rid, mid) in enumerate(thread_infos):
                    t = threads[idx]
                    if rid in timed_out:
                        continue
                    if t.is_alive():
                        timed_out.add(rid)
                        err_msg = "线程超时无响应（可能API连接卡住或工具执行耗时过长）"
                        llm_logger.error(f"[revise_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                    else:
                        timed_out.add(rid)
                        err_msg = "线程异常退出（prompt构建或API初始化阶段崩溃，详见日志）"
                        llm_logger.error(f"[revise_stream_parallel] {mid}: {err_msg}")
                        errors.append(f"{mid}: {err_msg}")
                        yield {
                            "type": "error",
                            "model_id": mid,
                            "role_id": rid,
                            "error": err_msg,
                        }
                        done_count += 1
                if done_count >= len(thread_infos):
                    break
                continue
            
            yield event
            rid = event.get("role_id", "")
            if event["type"] == "model_done" and rid not in timed_out:
                done_count += 1
            elif event["type"] == "error" and rid not in timed_out:
                done_count += 1
                errors.append(f"{rid}: {event['error']}")
        
        # 超时后 drain 剩余事件，避免 tool_executing/model_done 被吞
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                yield event
            except queue.Empty:
                break
        
        if errors:
            yield {"type": "partial_error", "errors": errors}
        
        self.stage = Stage.PAUSE_AFTER_REVISE
        
        # 更新 Memory
        self._update_memory("revise")
    
    def revise_stream_single(
        self, role_id: str, user_critique: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """串行流式修订：单个角色根据收到的评审进行修订
        
        与 revise_stream_parallel 的区别：
        - 只运行一个角色的 revise，不启动线程
        - 用于严格串行交替辩论流程
        """
        self.stage = Stage.REVISING
        
        memory_context = self._maybe_summarize()
        
        proposers = self._config.get_proposer_models()
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        
        model_id = self.role_models.get(role_id)
        if not model_id:
            yield {"type": "error", "model_id": role_id, "role_id": role_id, "error": f"未知角色: {role_id}"}
            return
        
        model_cfg = proposers.get(model_id)
        if not model_cfg:
            yield {"type": "error", "model_id": model_id, "role_id": role_id, "error": f"未知模型: {model_id}"}
            return
        
        try:
            critiques_on_me = {}
            for other_role, other_mid in self.role_models.items():
                if other_role == role_id:
                    continue
                if other_role in self.critiques and role_id in self.critiques[other_role]:
                    critiques_on_me[other_role] = self.critiques[other_role][role_id]
                else:
                    critiques_on_me[other_role] = "（未收到明确评审意见）"
            
            if user_critique:
                critiques_on_me["用户"] = f"【用户评判】\n{user_critique}"
            
            critique_history = self._build_critique_history_text(role_id, self.current_round)
            
            model_name = model_cfg.name
            user_prompt = prompts.build_revise_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                my_previous_answer=self.answers[role_id],
                critiques_on_me=critiques_on_me,
                memory_context=memory_context,
                critique_history=critique_history,
                skill_context=skill_context,
                tool_context=tool_context,
                my_role=role_id,
            )
            
            full_text = ""
            yield {"type": "start", "model_id": model_id, "role_id": role_id, "model_name": model_name}
            
            for token, accumulated in self._client.call_stream(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                full_text = accumulated
                yield {
                    "type": "token",
                    "model_id": model_id,
                    "role_id": role_id,
                    "model_name": model_name,
                    "token": token,
                    "accumulated": accumulated,
                }
            
            thinking, formal = self._extract_thinking(full_text)
            self.thinkings[role_id] = thinking
            self.answers[role_id] = formal
            self.confidences[role_id] = self._extract_confidence(full_text)
            self.summaries[role_id] = self._extract_summary(full_text)
            self.previous_answers[role_id] = full_text
            
            yield {"type": "model_done", "role_id": role_id, "model_id": model_id, "full_text": full_text}
            
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(
                f"[revise_stream_single] model={model_id} CRASH\n"
                f"Error: {e}\n{tb}"
            )
            yield {"type": "error", "model_id": model_id, "role_id": role_id, "error": f"{e}"}
        
        self.stage = Stage.PAUSE_AFTER_REVISE
        self._update_memory("revise")
    
    def critique(self) -> Dict[str, Any]:
        """评审阶段：串行单方向评审（bear → bull）"""
        self.stage = Stage.CRITIQUING
        
        memory_context = self._maybe_summarize()
        critique_history = self._build_all_critique_history_text(self.current_round)
        
        # 串行：bear 评审 bull
        critic_role = "bear"
        target_role = "bull"
        model_id = self.role_models.get(critic_role)
        
        if model_id:
            other_answers = {target_role: self.answers[target_role]}
            user_prompt = prompts.build_critique_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                my_answer=self.answers[critic_role],
                other_answers=other_answers,
                memory_context=memory_context,
                critique_history=critique_history,
                critiquer_role=critic_role,
            )
            
            system_prompt = prompts.SCENARIO_ROLES.get(
                self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
            )
            
            resp = self._client.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            if self.current_round not in self.critiques_history:
                self.critiques_history[self.current_round] = {}
            if critic_role not in self.critiques_history[self.current_round]:
                self.critiques_history[self.current_round][critic_role] = {}
            self.critiques_history[self.current_round][critic_role][target_role] = resp.content
            if critic_role not in self.critiques:
                self.critiques[critic_role] = {}
            self.critiques[critic_role][target_role] = resp.content
        
        self.stage = Stage.PAUSE_AFTER_CRITIQUE
        return self._build_ui_state()
    
    def revise(self, user_critique: Optional[str] = None) -> Dict[str, Any]:
        """修订阶段：串行单个角色修订（bull）"""
        self.stage = Stage.REVISING
        
        memory_context = self._maybe_summarize()
        
        role_id = "bull"
        model_id = self.role_models.get(role_id)
        
        if model_id:
            critiques_on_me = {}
            for other_role, other_mid in self.role_models.items():
                if other_role == role_id:
                    continue
                if other_role in self.critiques and role_id in self.critiques[other_role]:
                    critiques_on_me[other_role] = self.critiques[other_role][role_id]
                else:
                    critiques_on_me[other_role] = "（未收到明确评审意见）"
            
            if user_critique:
                critiques_on_me["用户"] = f"【用户评判】\n{user_critique}"
            
            critique_history = self._build_critique_history_text(role_id, self.current_round)
            
            user_prompt = prompts.build_revise_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=self.context,
                my_previous_answer=self.answers[role_id],
                critiques_on_me=critiques_on_me,
                memory_context=memory_context,
                critique_history=critique_history,
                my_role=role_id,
            )
            
            system_prompt = prompts.SCENARIO_ROLES.get(
                self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
            )
            
            resp = self._client.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            thinking, formal = self._extract_thinking(resp.content)
            self.thinkings[role_id] = thinking
            self.answers[role_id] = formal
            self.confidences[role_id] = self._extract_confidence(resp.content)
            self.previous_answers[role_id] = resp.content
        
        self.stage = Stage.PAUSE_AFTER_REVISE
        return self._build_ui_state()
    
    def aggregate(self) -> Dict[str, Any]:
        """聚合最终结论"""
        self.stage = Stage.AGGREGATING
        
        aggregator_cfg = self._config.get_aggregator_model()
        user_prompt = prompts.build_aggregate_prompt(
            question=self.question,
            context=self.context,
            all_answers=self.answers,
            all_confidences=self.confidences,
            all_rounds=self.current_round,
        )
        
        resp = self._client.call(
            model_id=aggregator_cfg.id,
            system_prompt="你是一位资深投资委员会主席，擅长综合多方观点得出平衡、专业的结论。",
            user_prompt=user_prompt,
            temperature=0.2,
        )
        
        self.final_answer = resp.content
        self.stage = Stage.DONE
        return self._build_ui_state()
    
    # ==================== 流程控制 ====================
    
    def step_stream(self) -> Generator[Dict[str, Any], None, None]:
        """统一的单步流式执行入口。
        
        UI 层只需调用此方法，DebateSession 根据当前 stage 自动决定执行什么：
        - INIT                  -> generate_stream_parallel()
        - PAUSE_AFTER_GENERATE  -> critique_stream_single("bear", "bull")
        - PAUSE_AFTER_CRITIQUE  -> revise_stream_single("bull")
        - PAUSE_AFTER_REVISE    -> current_round += 1, critique_stream_single("bear", "bull")
        
        每个 yield 的事件会注入 ``step_type`` 字段（"generate"/"critique"/"revise"），
        供 UI 层（如 run_stream）自动推断当前步骤类型，无需 UI 自行维护状态机。
        """
        if self.stage == Stage.INIT:
            step_type = "generate"
            stream = self.generate_stream_parallel()
        elif self.stage == Stage.PAUSE_AFTER_GENERATE:
            step_type = "critique"
            stream = self.critique_stream_single("bear", "bull")
        elif self.stage == Stage.PAUSE_AFTER_CRITIQUE:
            step_type = "revise"
            stream = self.revise_stream_single("bull")
        elif self.stage == Stage.PAUSE_AFTER_REVISE:
            step_type = "critique"
            self.current_round += 1
            stream = self.critique_stream_single("bear", "bull")
        else:
            raise ValueError(f"无法从 stage={self.stage} 执行下一步")
        
        for event in stream:
            event["step_type"] = step_type
            yield event
    
    def step(self) -> Dict[str, Any]:
        """同步版 step：消费 step_stream() 所有事件后返回 ui_state"""
        for _ in self.step_stream():
            pass
        return self._build_ui_state()
    
    def continue_next(self) -> Dict[str, Any]:
        """用户点击'继续下一轮'时的处理
        
        保持现有行为（一次调用完成 critique + revise 两步），
        内部复用 step() 实现。
        """
        if self.stage == Stage.PAUSE_AFTER_GENERATE:
            # 从初始生成 → 评审
            return self.critique()
        
        elif self.stage == Stage.PAUSE_AFTER_CRITIQUE:
            # 从评审 → 修订
            return self.revise()
        
        elif self.stage == Stage.PAUSE_AFTER_REVISE:
            # 从修订 → 进入下一轮：评审 → 修订
            self.current_round += 1
            return self.critique()
        
        else:
            return self._build_ui_state()
    
    def submit_user_critique(self, critique_text: str) -> Dict[str, Any]:
        """用户提交评判意见后，模型根据用户意见修订"""
        return self.revise(user_critique=critique_text)
    
    def finish(self) -> Dict[str, Any]:
        """用户点击'结束辩论'，直接聚合"""
        return self.aggregate()
    
    def auto_debate_stream(self, max_cycles: int = 3) -> Generator[Dict[str, Any], None, None]:
        """自动辩论：严格串行交替 critique → revise → judge，直到统一或达到最大轮数
        
        串行交替流程（按用户要求）：
        1. bear 分析 bull 的问题（critique）
        2. bull 根据 bear 的意见修订后再输出（revise）
        3. auto_judge 判断是否统一
        4. 如果未统一且未达 max_cycles，进入下一轮（bear 继续分析 bull 的新问题）
        
        与并行版本的区别：
        - critique 和 revise 严格串行，先 bear critique bull，再 bull revise
        - 不并行运行两个角色，保证一方输出后另一方才能基于最新输出进行反应
        
        Args:
            max_cycles: 最大辩论轮数（每轮 = critique + revise + judge）
        
        Yields:
            所有 critique、revise、judge 事件
        """
        session_logger.info(
            f"========== AUTO DEBATE START (串行交替) | max_cycles={max_cycles} =========="
        )
        
        for cycle in range(1, max_cycles + 1):
            self.current_round = cycle
            session_logger.info(f"========== AUTO DEBATE CYCLE {cycle} ==========")
            
            # === Critique 阶段：bear → bull ===
            self.stage = Stage.CRITIQUING
            llm_logger.info(f"[auto_debate_stream] 开始第 {cycle} 轮 critique (bear → bull)")
            for event in self.critique_stream_single("bear", "bull"):
                yield event
            
            # === Revise 阶段：bull ===
            self.stage = Stage.REVISING
            llm_logger.info(f"[auto_debate_stream] 开始第 {cycle} 轮 revise (bull)")
            for event in self.revise_stream_single("bull"):
                yield event
            
            # === Judge 阶段 ===
            llm_logger.info(f"[auto_debate_stream] 第 {cycle} 轮结束，调用 judge")
            judge_result = self.auto_judge(max_rounds=max_cycles)
            
            yield {
                "type": "judge",
                "action": judge_result["action"],
                "reason": judge_result["reason"],
                "info_needed": judge_result["info_needed"],
                "round": cycle,
            }
            
            session_logger.info(
                f"[auto_debate_stream] judge 结果: action={judge_result['action']} | "
                f"reason={judge_result['reason'][:100]!r}"
            )
            
            if judge_result["action"] == "stop":
                self.stage = Stage.DONE
                session_logger.info("========== AUTO DEBATE STOP (统一) ==========")
                break
            elif judge_result["action"] == "need_info":
                self.stage = Stage.PAUSE_AFTER_REVISE
                session_logger.info("========== AUTO DEBATE PAUSE (需补充信息) ==========")
                break
        else:
            # 达到 max_cycles 仍未统一
            self.stage = Stage.PAUSE_AFTER_REVISE
            session_logger.info(
                f"========== AUTO DEBATE END (达到 max_cycles={max_cycles}) =========="
            )
    
    def auto_judge(self, max_rounds: int = 5) -> Dict[str, Any]:
        """自动裁判：判断辩论是否应该继续、结束，还是需要补充信息
        
        Returns:
            dict with keys:
            - action: "continue" | "stop" | "need_info"
            - reason: 判断依据
            - info_needed: 如果需要补充信息，具体说明需要什么
        """
        aggregator_cfg = self._config.get_aggregator_model()
        
        # 收集最近一轮的评审（用于 judge 判断）
        latest_critiques = {}
        if self.current_round in self.critiques_history:
            latest_critiques = self.critiques_history[self.current_round]
        elif self.critiques:
            latest_critiques = self.critiques
        
        user_prompt = prompts.build_auto_judge_prompt(
            question=self.question,
            context=self.context,
            all_answers=self.answers,
            all_confidences=self.confidences,
            all_critiques=latest_critiques,
            round_num=self.current_round,
            max_rounds=max_rounds,
        )
        
        try:
            resp = self._client.call(
                model_id=aggregator_cfg.id,
                system_prompt="你是一位资深投资委员会主席，擅长客观评估辩论质量并做出流程决策。",
                user_prompt=user_prompt,
                temperature=0.2,
            )
            
            # 解析 JSON 输出
            content = resp.content.strip()
            # 尝试提取 JSON 块（模型可能用 ```json 包裹）
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果解析失败，尝试找第一个 { 到最后一个 }
                brace_match = re.search(r'\{.*\}', content, re.DOTALL)
                if brace_match:
                    try:
                        result = json.loads(brace_match.group(0))
                    except json.JSONDecodeError:
                        result = {"action": "continue", "reason": f"JSON 解析失败: {content[:200]}", "info_needed": ""}
                else:
                    result = {"action": "continue", "reason": f"无法解析 judge 输出: {content[:200]}", "info_needed": ""}
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(
                f"[auto_judge] 裁判调用失败\n"
                f"Error: {e}\n{tb}"
            )
            result = {"action": "continue", "reason": f"裁判调用失败: {e}", "info_needed": ""}
        
        # 确保字段存在
        action = result.get("action", "continue")
        if action not in ("continue", "stop", "need_info"):
            action = "continue"
        
        return {
            "action": action,
            "reason": result.get("reason", ""),
            "info_needed": result.get("info_needed", ""),
        }
    
    # ==================== 单角色重试 ====================
    
    def retry_generate_role_stream(self, role_id: str, extra_context: str = "", round_type: str = "generate") -> Generator[Dict[str, Any], None, None]:
        """重新生成指定角色的输出（流式，支持补充新信息）
        
        Args:
            role_id: 角色 ID
            extra_context: 用户补充的上下文信息
            round_type: 轮次类型，"generate" / "brainstorm" / "revise"，决定使用哪个 prompt 模板
        """
        model_id = self.role_models.get(role_id)
        if not model_id:
            yield {"type": "error", "model_id": role_id, "role_id": role_id, "error": f"未知角色: {role_id}"}
            return
        
        proposers = self._config.get_proposer_models()
        model_cfg = proposers.get(model_id)
        if not model_cfg:
            yield {"type": "error", "model_id": model_id, "role_id": role_id, "error": f"未知模型: {model_id}"}
            return
        
        session_logger.info(
            f"========== RETRY GENERATE | role={role_id} | model={model_id} | round={self.current_round} | type={round_type} =========="
        )
        
        system_prompt = prompts.SCENARIO_ROLES.get(
            self.scenario_type, prompts.SCENARIO_ROLES["question_analysis"]
        )
        
        skill_context = self._get_skill_context()
        tool_context = self._get_tool_context()
        # 将 tool 说明合并到 system_prompt（同 generate_stream_parallel）
        if tool_context:
            system_prompt = system_prompt + "\n\n" + tool_context
        
        # 根据 round_type 选择正确的 prompt 模板
        context_with_extra = self.context + (f"\n\n【用户补充信息】\n{extra_context}" if extra_context.strip() else "")
        if round_type == "brainstorm":
            user_prompt = prompts.build_brainstorm_prompt(
                question=self.question,
                context=context_with_extra,
                debater_role=role_id,
            )
        else:
            # generate / revise 均使用 build_generate_prompt
            user_prompt = prompts.build_generate_prompt(
                scenario_type=self.scenario_type,
                question=self.question,
                context=context_with_extra,
                round_num=self.current_round,
                debater_role=role_id,
                skill_context=skill_context,
                tool_context="",
                include_tool_strategy=True,
            )
        
        log_prompt(
            session_logger,
            f"RETRY | role={role_id} | model={model_id}",
            system_prompt,
            user_prompt,
        )
        
        yield {"type": "start", "model_id": model_id, "role_id": role_id, "model_name": model_cfg.name}
        
        full_text = ""
        try:
            # ========== 第一轮 ==========
            for token, accumulated in self._client.call_stream(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                full_text = accumulated
                yield {"type": "token", "model_id": model_id, "role_id": role_id, "token": token, "accumulated": accumulated}
            
            first_round_text = full_text
            
            log_response(
                session_logger,
                f"RETRY R1 | role={role_id} | model={model_id}",
                full_text,
            )
            
            # ========== 工具调用（同 generate_stream_parallel 逻辑）==========
            executor = self._tool_registry.create_executor()
            has_tc_1 = executor.has_tool_calls(full_text)
            log_event(
                session_logger,
                "TOOL_CHECK_R1",
                role_id=role_id,
                model_id=model_id,
                has_tool_calls=has_tc_1,
                output_len=len(full_text),
            )
            llm_logger.info(
                f"[retry_generate_role_stream] model={model_id} tool_round=1 完成 | "
                f"has_tool_calls={has_tc_1} | output_len={len(full_text)}"
            )
            
            tool_info = None
            fallback_search = False
            
            if has_tc_1:
                detected = list(executor.TOOL_CALL_PATTERN.finditer(full_text))
                tool_names = [m.group(1) for m in detected]
                llm_logger.info(
                    f"[retry_generate_role_stream] model={model_id} 检测到 {len(detected)} 个工具调用 | "
                    f"tools={tool_names}"
                )
                yield {
                    "type": "tool_executing",
                    "model_id": model_id,
                    "role_id": role_id,
                    "message": f"🔧 检测到 {len(detected)} 个工具调用，正在并行执行...",
                    "has_error": False,
                    "errors": [],
                    "fallback_notices": [],
                    "executions": [],
                }
                tool_info = self._execute_tool_calls(full_text, model_id=model_id)
            elif full_text:
                intent_keywords = ["搜索", "查询", "查一下", "获取数据", "用工具", "调用工具", "事实核查"]
                if any(kw in full_text for kw in intent_keywords):
                    llm_logger.warning(
                        f"[retry_generate_role_stream] model={model_id} 检测到工具意图关键词但无 XML 标签，"
                        f"触发系统回退搜索 | output_preview={full_text[:200]!r}"
                    )
                    fallback_search = True
                    fallback_query = self.question[:100]
                    import html
                    safe_query = html.escape(fallback_query, quote=False)
                    fallback_text = f'<tool_call name="web_search"><query>{safe_query}</query></tool_call>'
                    tool_info = self._execute_tool_calls(fallback_text, model_id=model_id)
                    yield {
                        "type": "tool_executing",
                        "model_id": model_id,
                        "role_id": role_id,
                        "message": "🔄 模型未输出工具标签，系统自动执行回退搜索",
                        "has_error": False,
                        "errors": [],
                        "fallback_notices": ["模型未按格式输出工具标签，系统自动搜索补充数据"],
                        "executions": [],
                    }
            
            # ========== 工具链循环（支持最多 MAX_TOOL_ROUNDS 轮）==========
            if tool_info and tool_info["text"]:
                has_error = tool_info["has_error"]
                errors = tool_info["errors"]
                fallback_notices = list(tool_info["fallback_notices"])
                if fallback_search:
                    fallback_notices.append("【系统回退】模型未按格式输出工具标签，系统自动搜索补充数据")
                
                llm_logger.info(
                    f"[retry_generate_role_stream] model={model_id} tool_round=1 执行完成 | "
                    f"has_error={has_error} | errors={len(errors)} | "
                    f"fallback_notices={len(fallback_notices)}"
                )
                
                # 修正 tool_info
                tool_info = {**tool_info, "fallback_notices": fallback_notices}
                
                # 构建 clean_first_round
                clean_first_round = re.sub(
                    r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                    '', first_round_text, flags=re.DOTALL
                ).strip()
                if not clean_first_round:
                    clean_first_round = "（你之前只调用了工具，尚未给出分析）"
                
                current_text = first_round_text
                previous_analysis = clean_first_round
                tool_round = 1
                
                while True:
                    if tool_round == 1:
                        pass  # 使用已有的 tool_info
                    else:
                        tool_info = self._execute_tool_calls(current_text, model_id=model_id)
                        if not tool_info["text"]:
                            full_text = current_text
                            break
                    
                    tool_round += 1
                    
                    # 发送 tool_executing 事件
                    has_error = tool_info["has_error"]
                    errors = tool_info["errors"]
                    fallback_notices = list(tool_info["fallback_notices"])
                    
                    if has_error and not fallback_notices:
                        status_msg = f"⚠️ 工具全部失败 ({len(errors)} 个错误)"
                    elif fallback_notices:
                        status_msg = f"🔄 工具回退: {'; '.join(fallback_notices)}"
                    else:
                        status_msg = "✅ 工具执行完成"
                    
                    yield {
                        "type": "tool_executing",
                        "model_id": model_id,
                        "role_id": role_id,
                        "message": status_msg,
                        "tool_results": tool_info["text"],
                        "has_error": has_error,
                        "errors": errors,
                        "fallback_notices": fallback_notices,
                        "executions": tool_info.get("executions", []),
                    }
                    
                    # 构建 followup prompt
                    followup_prompt = self._build_tool_followup_prompt(
                        previous_analysis, tool_info, tool_round
                    )
                    
                    log_prompt(
                        session_logger,
                        f"RETRY R{tool_round} | role={role_id} | model={model_id}",
                        system_prompt,
                        followup_prompt,
                    )
                    
                    next_text = ""
                    try:
                        for token, accumulated in self._client.call_stream(
                            model_id=model_id,
                            system_prompt=system_prompt,
                            user_prompt=followup_prompt,
                        ):
                            next_text = accumulated
                            yield {"type": "token", "model_id": model_id, "role_id": role_id, "token": token, "accumulated": accumulated}
                    except RuntimeError as e:
                        if "high risk" in str(e).lower():
                            llm_logger.warning(
                                f"[retry_generate_role_stream] model={model_id} 第{tool_round}轮因内容安全被拒绝，"
                                f"使用简化 prompt 重试"
                            )
                            simple_followup = self._build_simple_followup_prompt(previous_analysis)
                            log_prompt(
                                session_logger,
                                f"RETRY R{tool_round} SIMPLE | role={role_id} | model={model_id}",
                                system_prompt,
                                simple_followup,
                            )
                            for token, accumulated in self._client.call_stream(
                                model_id=model_id,
                                system_prompt=system_prompt,
                                user_prompt=simple_followup,
                            ):
                                next_text = accumulated
                                yield {"type": "token", "model_id": model_id, "role_id": role_id, "token": token, "accumulated": accumulated}
                        else:
                            raise
                    
                    # 空输出防护
                    if not next_text or not next_text.strip():
                        llm_logger.warning(
                            f"[retry_generate_role_stream] model={model_id} 第{tool_round}轮输出为空，"
                            f"回退到上一轮输出 | prev_len={len(current_text)}"
                        )
                        next_text = current_text
                    
                    log_response(
                        session_logger,
                        f"RETRY R{tool_round} | role={role_id} | model={model_id}",
                        next_text,
                    )
                    llm_logger.info(
                        f"[retry_generate_role_stream] model={model_id} tool_round={tool_round} 完成 | "
                        f"output_len={len(next_text)}"
                    )
                    
                    # 检查是否还有 tool_call
                    has_tc_next = self._tool_registry.create_executor().has_tool_calls(next_text)
                    if not has_tc_next:
                        full_text = next_text
                        break
                    
                    # 紧急断路器：如果上一轮已经只输出工具，这一轮仍然输出工具 → 停止循环
                    if previous_analysis == "（你之前只调用了工具，尚未给出分析）":
                        llm_logger.error(
                            f"[retry_generate_role_stream] model={model_id} 第{tool_round}轮"
                            f"在纯工具输出后又输出工具，紧急停止，保留当前输出"
                        )
                        full_text = next_text
                        break
                    
                    if tool_round >= self.MAX_TOOL_ROUNDS:
                        error_msg = (
                            f"模型连续 {self.MAX_TOOL_ROUNDS} 轮输出了 tool call，"
                            f"无法生成最终分析。"
                        )
                        llm_logger.error(f"[retry_generate_role_stream] model={model_id} {error_msg}")
                        yield {"type": "error", "model_id": model_id, "role_id": role_id, "error": error_msg}
                        return
                    
                    current_text = next_text
                    previous_analysis = re.sub(
                        r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
                        '', current_text, flags=re.DOTALL
                    ).strip()
                    if not previous_analysis:
                        previous_analysis = "（你之前只调用了工具，尚未给出分析）"
            
            # 保存结果
            thinking, formal = self._extract_thinking(full_text)
            self.thinkings[role_id] = thinking
            self.answers[role_id] = formal
            self.confidences[role_id] = self._extract_confidence(full_text)
            self.summaries[role_id] = self._extract_summary(full_text)
            self.previous_answers[role_id] = full_text
            
            log_event(
                session_logger,
                "RETRY_DONE",
                role_id=role_id,
                model_id=model_id,
                final_text_len=len(full_text),
                thinking_len=len(thinking),
                formal_len=len(formal),
                confidence=self.confidences[role_id],
            )
            session_logger.info(
                f"========== RETRY GENERATE END | role={role_id} | model={model_id} =========="
            )
            
            yield {"type": "model_done", "model_id": model_id, "role_id": role_id, "full_text": full_text}
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[retry_generate_role_stream] {model_id} CRASH\nError: {e}\n{tb}")
            session_logger.error(
                f"[ERROR] RETRY role={role_id} model={model_id} CRASH | {e}\n{tb}"
            )
            yield {"type": "error", "model_id": model_id, "role_id": role_id, "error": f"{e}"}
    
    # ==================== UI 状态构建 ====================
    
    def _build_ui_state(self) -> Dict[str, Any]:
        """构建供 Gradio 使用的 UI 状态字典"""
        
        # 构建阶段描述
        stage_desc = {
            Stage.INIT: "准备就绪",
            Stage.GENERATING: "正在生成初始回答...",
            Stage.PAUSE_AFTER_GENERATE: f"第 {self.current_round} 轮 - 初始生成完成，等待你的决策",
            Stage.CRITIQUING: f"第 {self.current_round} 轮 - 正在互相评审...",
            Stage.PAUSE_AFTER_CRITIQUE: f"第 {self.current_round} 轮 - 评审完成，等待你的决策",
            Stage.REVISING: f"第 {self.current_round} 轮 - 正在修订回答...",
            Stage.PAUSE_AFTER_REVISE: f"第 {self.current_round} 轮 - 修订完成，等待你的决策",
            Stage.AGGREGATING: "正在聚合最终结论...",
            Stage.DONE: "辩论结束",
        }.get(self.stage, self.stage)
        
        # 判断各按钮是否可用
        can_continue = self.stage in [
            Stage.PAUSE_AFTER_GENERATE,
            Stage.PAUSE_AFTER_CRITIQUE,
            Stage.PAUSE_AFTER_REVISE,
        ]
        can_critique = self.stage in [
            Stage.PAUSE_AFTER_GENERATE,
            Stage.PAUSE_AFTER_CRITIQUE,
            Stage.PAUSE_AFTER_REVISE,
        ]
        can_finish = self.stage in [
            Stage.PAUSE_AFTER_GENERATE,
            Stage.PAUSE_AFTER_CRITIQUE,
            Stage.PAUSE_AFTER_REVISE,
        ]
        is_done = self.stage == Stage.DONE
        
        # 动态构建各模型状态
        ui_state = {
            "stage": self.stage,
            "stage_info": stage_desc,
            "final_answer": self.final_answer if is_done else "",
            "can_continue": can_continue,
            "can_critique": can_critique,
            "can_finish": can_finish,
            "is_done": is_done,
            "round_info": f"第 {self.current_round} 轮",
            "models": [],
        }
        for role_id, model_id in self.role_models.items():
            safe_id = model_id.replace(".", "_")
            model_cfg = self._config.models.get(model_id)
            ui_state[f"{safe_id}_thinking"] = self.thinkings.get(role_id, "")
            ui_state[f"{safe_id}_output"] = self.answers.get(role_id, "")
            ui_state[f"{safe_id}_conf"] = f"{self.confidences.get(role_id, 0) * 100:.0f}%"
            ui_state["models"].append({
                "id": model_id,
                "name": model_cfg.name if model_cfg else model_id,
                "critique_style": model_cfg.critique_style if model_cfg else "standard",
            })
        return ui_state
    
    # ==================== Save / Load ====================
    
    def save(self, save_dir: Optional[str] = None) -> str:
        """保存辩论状态到 JSON 文件
        
        Returns:
            保存的文件路径
        """
        save_dir = Path(save_dir or "/Users/lunight/dev/debater/saves")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        debate_id = self.memory_manager.debate_id if self.memory_manager else "unknown"
        save_path = save_dir / f"debate_{debate_id}.json"
        
        state = {
            "version": 1,
            "question": self.question,
            "scenario_type": self.scenario_type,
            "context": self.context,
            "current_round": self.current_round,
            "stage": self.stage,
            "answers": self.answers,
            "thinkings": self.thinkings,
            "summaries": self.summaries,
            "confidences": {k: v for k, v in self.confidences.items()},
            "previous_answers": dict(self.previous_answers),
            "critiques": {k: dict(v) for k, v in self.critiques.items()},
            "critiques_history": {
                str(r): {m: dict(t) for m, t in round_data.items()}
                for r, round_data in self.critiques_history.items()
            },
            "role_models": dict(self.role_models),
            "final_answer": self.final_answer,
            "known_facts": self.known_facts,
        }
        
        # 保存 MemoryManager 的文件内容
        if self.memory_manager:
            state["memory"] = {
                "debate_id": self.memory_manager.debate_id,
                "topic": self.memory_manager.topic,
                "working_memory": self.memory_manager.read_working_memory(),
            }
            stage_files = sorted(self.memory_manager.root.glob("stage_memory/stage_*.md"))
            if stage_files:
                state["memory"]["stage_memories"] = {
                    f.stem: f.read_text(encoding="utf-8")
                    for f in stage_files
                }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVE] 辩论状态已保存: {save_path}")
        return str(save_path)
    
    @classmethod
    def load(cls, save_path: str) -> "DebateSession":
        """从 JSON 文件加载辩论状态
        
        Args:
            save_path: JSON 文件路径
        
        Returns:
            恢复后的 DebateSession 实例
        """
        with open(save_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # 使用 __new__ 绕过 __post_init__，手动恢复所有字段
        session = object.__new__(cls)
        
        # 恢复基础字段
        session.question = state["question"]
        session.scenario_type = state["scenario_type"]
        session.context = state["context"]
        session.current_round = state["current_round"]
        session.stage = state["stage"]
        session.answers = state.get("answers", {})
        session.thinkings = state.get("thinkings", {})
        session.summaries = state.get("summaries", {})
        session.confidences = state.get("confidences", {})
        # 向后兼容：旧格式存的是 list，取最后一个元素
        raw = state.get("previous_answers", {})
        pa = {}
        for k, v in raw.items():
            if isinstance(v, list):
                pa[k] = v[-1] if v else ""
            else:
                pa[k] = v
        session.previous_answers = pa
        session.critiques = state.get("critiques", {})
        session.known_facts = state.get("known_facts", [])
        
        # 恢复 critiques_history（round key 从 str 转回 int）
        critiques_history_raw = state.get("critiques_history", {})
        session.critiques_history = {
            int(r): round_data
            for r, round_data in critiques_history_raw.items()
        }
        
        session.final_answer = state.get("final_answer", "")
        
        # 初始化运行时对象
        session._client = LLMClient()
        session._config = get_config()
        session.memory_manager = None
        
        # 恢复角色-模型映射（向后兼容：旧格式没有 role_models）
        raw_roles = state.get("role_models")
        if raw_roles:
            session.role_models = dict(raw_roles)
        else:
            # 从 _models 推断：第一个 = bull，第二个 = bear
            available = list(session._config.get_proposer_models().keys())
            session.role_models = {
                "bull": available[0] if len(available) > 0 else "",
                "bear": available[1] if len(available) > 1 else available[0] if len(available) > 0 else "",
            }
        session._sync_models_from_roles()
        
        # 恢复 MemoryManager
        mem_state = state.get("memory")
        if mem_state:
            from datetime import datetime
            debate_id = mem_state.get("debate_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            topic = mem_state.get("topic", session.question[:30])
            session.memory_manager = MemoryManager(debate_id=debate_id, topic=topic)
            
            # 恢复 working_memory.md
            working_content = mem_state.get("working_memory", "")
            if working_content:
                session.memory_manager.working_file.write_text(working_content, encoding="utf-8")
            
            # 恢复 stage_memory 文件
            stage_memories = mem_state.get("stage_memories", {})
            for stage_name, content in stage_memories.items():
                stage_file = session.memory_manager.root / "stage_memory" / f"{stage_name}.md"
                stage_file.write_text(content, encoding="utf-8")
        
        print(f"[LOAD] 辩论状态已加载: {save_path} (第 {session.current_round} 轮, {session.stage})")
        return session
