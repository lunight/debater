"""LangGraph 节点实现

包含多模型辩论的所有节点逻辑：
- initialize: 初始化
- generate: 并行生成初始回答（结果以 role_id 为 key）
- critique: 串行评审（bear → bull）
- revise: 串行修订（bull only）
- judge: LLM-based 判断是否收敛
- aggregate: 最终聚合

与设计文档保持一致：
1. 状态以 role_id（"bull"/"bear"）为 key，而非 model_id
2. critique/revise 严格串行交替，非并行
3. generate 只做一次，后续 critique→revise→judge 循环
"""

import json
import re
import traceback
from typing import Dict, List, Any

import os

from .state import DebateState
from .config import get_config
from .llm_client import LLMClient
from . import prompts
from .tools import ToolRegistry
from .logger import llm_logger


# ==================== 工具函数 ====================

def _extract_confidence(text: str) -> float:
    """从回答中提取置信度 (0.0~1.0)"""
    patterns = [
        r'置信度[^\d]*(\d+(?:\.\d+)?)\s*%',
        r'(?:confidence|Confidence)[^\d]*(\d+(?:\.\d+)?)\s*%',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            return min(max(val / 100.0, 0.0), 1.0)
    return 0.7


def _build_reasoning_log(state: DebateState) -> str:
    """构建推理过程日志"""
    lines = []
    lines.append(f"问题: {state['question']}")
    lines.append(f"场景: {state['scenario_type']}")
    lines.append(f"总轮次: {state['round']} / {state['max_rounds']}")
    lines.append("")

    role_models = state.get("role_models", {})
    for role_id in state.get("answers", {}):
        model_id = role_models.get(role_id, role_id)
        lines.append(f"--- {role_id} ({model_id}) ---")
        prev = state.get("previous_answers", {}).get(role_id, "")
        if isinstance(prev, str):
            lines.append(f"  回答片段: {prev[:200]}...")
        conf = state.get("confidences", {}).get(role_id, 0)
        lines.append(f"  最终置信度: {conf*100:.0f}%")
        lines.append("")

    return "\n".join(lines)


# ==================== 节点类 ====================

class DebateNodes:
    """辩论节点集合"""

    def __init__(self):
        self.client = LLMClient()
        self.config = get_config()
        self._tool_registry = ToolRegistry(base_dir=os.getcwd())
        self._tool_registry.register_default_tools()

    def _execute_tool_calls(self, text: str, model_id: str = "unknown") -> dict:
        """检查并执行文本中的 tool calls"""
        executor = self._tool_registry.create_executor()
        has_tc = executor.has_tool_calls(text)
        llm_logger.info(f"[nodes._execute_tool_calls] model={model_id} has_tool_calls={has_tc}")
        if not has_tc:
            return {"text": "", "errors": [], "has_error": False, "fallback_notices": [], "executions": []}

        llm_logger.info(f"[nodes._execute_tool_calls] model={model_id} 调用 extract_and_execute...")
        results = executor.extract_and_execute(text)
        llm_logger.info(
            f"[nodes._execute_tool_calls] model={model_id} extract_and_execute 返回 {len(results)} 个结果"
        )
        if not results:
            return {
                "text": '<tool_result name="internal_error">\n【工具解析异常】系统检测到工具调用标记但未能提取有效参数，请基于已有知识继续分析。\n</tool_result>',
                "errors": ["检测到工具调用标记但未能提取有效参数"],
                "has_error": True,
                "fallback_notices": [],
                "executions": [],
            }

        errors = []
        fallback_notices = []
        executions = []
        for idx, r in enumerate(results):
            result = r["result"]
            status = "error" if result.error else ("success" if result.content else "empty")
            preview = (result.content or "")[:300]
            if result.content and len(result.content) > 300:
                preview += "..."
            fb_notice = None
            if result.content and "[Tavily 不可用" in result.content:
                for line in result.content.split("\n"):
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
                "duration": r.get("duration", 0),
            })
            llm_logger.info(
                f"[nodes._execute_tool_calls] model={model_id} 结果#{idx+1} | "
                f"tool={r['name']} | status={status} | duration={r.get('duration', 0):.2f}s | "
                f"error={result.error[:150] if result.error else 'None'}"
            )

        tool_text = executor.build_tool_result_message(results)
        llm_logger.info(
            f"[nodes._execute_tool_calls] model={model_id} 完成 | "
            f"has_error={len(errors)>0} | errors={len(errors)} | "
            f"fallback_notices={len(fallback_notices)}"
        )
        return {
            "text": tool_text,
            "errors": errors,
            "has_error": len(errors) > 0,
            "fallback_notices": fallback_notices,
            "executions": executions,
        }

    def _tool_chain_loop(self, first_round_text: str, model_id: str, system_prompt: str, question: str, context: str) -> str:
        """运行工具链循环（简化版，最多3轮防护）"""
        clean_first = re.sub(
            r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>',
            '', first_round_text, flags=re.DOTALL
        ).strip()
        if not clean_first:
            clean_first = "（你之前只调用了工具，尚未给出分析）"

        current_text = first_round_text
        previous_analysis = clean_first
        tool_round = 1
        full_text = first_round_text

        while True:
            if tool_round > 1:
                tool_info = self._execute_tool_calls(current_text, model_id=model_id)
                if not tool_info["text"]:
                    full_text = current_text
                    break
            else:
                tool_info = self._execute_tool_calls(first_round_text, model_id=model_id)
                if not tool_info["text"]:
                    full_text = first_round_text
                    break

            tool_round += 1
            has_error = tool_info["has_error"]
            errors = tool_info["errors"]
            fallback_notices = tool_info["fallback_notices"]

            if has_error and not fallback_notices:
                followup_prompt = (
                    f"【你之前的初步分析】\n"
                    f"{previous_analysis}\n\n"
                    f"【🚨 工具调用全部失败——关键数据缺失】\n\n"
                    f"你之前尝试调用工具验证关键数据，但全部执行失败：\n"
                    + "\n".join(f"  - {e}" for e in errors) +
                    f"\n\n"
                    f"【原始问题】\n{question}\n\n"
                    f"【上下文】\n{context or '（无）'}\n\n"
                    f"【⚠️ 强制要求——工具失败时结论不可靠】\n"
                    f"1. 你必须明确列出：哪些关键数据因工具失败而无法获取\n"
                    f"2. 对于无法验证的数据点，**不得给出确定性结论**\n"
                    f"3. **置信度必须低于 50%**\n"
                    f"4. 在 <summary> 中明确标注'因工具失败导致的数据缺失盲区'\n"
                    f"5. 不要虚构数据，不要假设工具应该返回的结果\n"
                    f"6. 基于已有知识做分析，但每处涉及缺失数据的地方必须标注'（未经工具验证）'"
                )
            else:
                error_hint = ""
                if fallback_notices:
                    error_hint = (
                        f"\n⚠️ 警告：部分工具经历了回退（{'；'.join(fallback_notices)}）。"
                        f"回退数据来源可能不如原始工具可靠，"
                        f"请在分析中明确标注哪些数据来自回退来源。"
                    )
                followup_prompt = (
                    f"【你之前的初步分析】\n"
                    f"{previous_analysis}\n\n"
                    f"【工具返回的结果】\n"
                    f"{tool_info['text']}\n\n"
                    f"【原始问题】\n{question}\n\n"
                    f"【上下文】\n{context or '（无）'}\n\n"
                    f"请基于以上工具结果，完善并更新你之前的初步分析。"
                    f"不要调用任何工具，直接输出完整的最终分析。"
                    f"{error_hint}"
                )

            resp = self.client.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=followup_prompt,
            )
            next_text = resp.content

            executor = self._tool_registry.create_executor()
            has_tc_next = executor.has_tool_calls(next_text)
            if not has_tc_next:
                full_text = next_text
                break

            if previous_analysis == "（你之前只调用了工具，尚未给出分析）":
                llm_logger.error(
                    f"[nodes._tool_chain_loop] model={model_id} 强制禁止工具但模型仍输出tool_call，"
                    f"保留当前输出"
                )
                full_text = next_text
                break

            if tool_round >= 3:
                llm_logger.error(
                    f"[nodes._tool_chain_loop] model={model_id} 连续3轮输出 tool call，保留第一轮结果"
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

        return full_text

    # ---------- 初始化节点 ----------

    def initialize(self, state: DebateState) -> DebateState:
        """初始化辩论状态"""
        if not state.get("round"):
            state["round"] = 0
        if not state.get("max_rounds"):
            state["max_rounds"] = getattr(self.config.debate, 'max_rounds', 5)
        if not state.get("answers"):
            state["answers"] = {}
        if not state.get("previous_answers"):
            state["previous_answers"] = {}
        if not state.get("critiques"):
            state["critiques"] = {}
        if not state.get("confidences"):
            state["confidences"] = {}
        if "consensus_reached" not in state:
            state["consensus_reached"] = False
        if not state.get("final_answer"):
            state["final_answer"] = None
        if not state.get("reasoning_process"):
            state["reasoning_process"] = ""
        if not state.get("divergence_points"):
            state["divergence_points"] = []
        if not state.get("role_models"):
            proposers = self.config.get_proposer_models()
            available = list(proposers.keys())
            state["role_models"] = {
                "bull": available[0] if len(available) > 0 else "",
                "bear": available[1] if len(available) > 1 else (available[0] if len(available) > 0 else ""),
            }

        return state

    # ---------- 生成节点（并行，结果以 role_id 为 key） ----------

    def generate(self, state: DebateState) -> DebateState:
        """第一轮：所有角色并行生成初始回答（支持 tool calls）"""
        state["round"] = 1
        print(f"\n🔄 第 1 轮：角色并行生成初始回答...")

        try:
            proposers = self.config.get_proposer_models()
            system_prompt = prompts.SCENARIO_ROLES.get(
                state["scenario_type"], prompts.SCENARIO_ROLES["question_analysis"]
            )
            role_models = state.get("role_models", {})

            # 构建并行调用（call_id = role_id）
            calls = []
            for role_id, model_id in role_models.items():
                user_prompt = prompts.build_generate_prompt(
                    scenario_type=state["scenario_type"],
                    question=state["question"],
                    context=state.get("context", ""),
                    round_num=1,
                    debater_role=role_id,
                )
                calls.append({
                    "call_id": role_id,
                    "model_id": model_id,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })

            # 并行调用（第一轮）
            responses = self.client.call_parallel(calls)

            # 解析结果 + tool call 双轮（结果以 role_id 为 key）
            for role_id, resp in responses.items():
                try:
                    model_id = role_models.get(role_id, role_id)
                    full_text = resp.content

                    # === Tool call 处理 ===
                    full_text = self._tool_chain_loop(
                        first_round_text=full_text,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        question=state["question"],
                        context=state.get("context", ""),
                    )

                    state["answers"][role_id] = full_text
                    state["confidences"][role_id] = _extract_confidence(full_text)
                    state["previous_answers"][role_id] = full_text
                    print(f"   ✅ {role_id} ({model_id}): 置信度 {state['confidences'][role_id]*100:.0f}% ({resp.latency_ms:.0f}ms)")
                except Exception as e:
                    tb = traceback.format_exc()
                    llm_logger.error(f"[nodes.generate] role={role_id} FAILED\nError: {e}\n{tb}")
                    print(f"   ❌ {role_id}: 生成失败: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[nodes.generate] 整体调用失败\nError: {e}\n{tb}")
            print(f"   ❌ 整体生成失败: {e}")

        return state

    # ---------- 评审节点（串行：bear → bull） ----------

    def critique(self, state: DebateState) -> DebateState:
        """评审阶段：串行单方向评审（bear 评审 bull）"""
        print(f"\n🔄 第 {state['round']} 轮：bear 评审 bull...")

        role_models = state.get("role_models", {})
        critic_role = "bear"
        target_role = "bull"
        model_id = role_models.get(critic_role)

        if not model_id:
            llm_logger.warning("[nodes.critique] bear 未分配模型，跳过评审")
            return state

        try:
            system_prompt = prompts.SCENARIO_ROLES.get(
                state["scenario_type"], prompts.SCENARIO_ROLES["question_analysis"]
            )

            other_answers = {target_role: state["answers"].get(target_role, "")}
            user_prompt = prompts.build_critique_prompt(
                scenario_type=state["scenario_type"],
                question=state["question"],
                context=state.get("context", ""),
                my_answer=state["answers"].get(critic_role, ""),
                other_answers=other_answers,
                critiquer_role=critic_role,
            )

            resp = self.client.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # 以 role_id 为 key 存储
            if critic_role not in state["critiques"]:
                state["critiques"][critic_role] = {}
            state["critiques"][critic_role][target_role] = resp.content
            print(f"   ✅ {critic_role} ({model_id}) 完成评审")
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[nodes.critique] {critic_role} FAILED\nError: {e}\n{tb}")
            print(f"   ❌ {critic_role}: 评审失败: {e}")

        return state

    # ---------- 修订节点（串行：bull only） ----------

    def revise(self, state: DebateState) -> DebateState:
        """修订阶段：串行单个角色修订（bull 根据 bear 的评审修订）"""
        print(f"\n🔄 第 {state['round']} 轮：bull 修订回答...")

        role_models = state.get("role_models", {})
        role_id = "bull"
        model_id = role_models.get(role_id)

        if not model_id:
            llm_logger.warning("[nodes.revise] bull 未分配模型，跳过修订")
            return state

        try:
            system_prompt = prompts.SCENARIO_ROLES.get(
                state["scenario_type"], prompts.SCENARIO_ROLES["question_analysis"]
            )

            # 收集其他角色对我的评审
            critiques_on_me = {}
            for other_role, other_mid in role_models.items():
                if other_role == role_id:
                    continue
                if other_role in state["critiques"] and role_id in state["critiques"][other_role]:
                    critiques_on_me[other_role] = state["critiques"][other_role][role_id]
                else:
                    critiques_on_me[other_role] = "（未收到明确评审意见）"

            user_prompt = prompts.build_revise_prompt(
                scenario_type=state["scenario_type"],
                question=state["question"],
                context=state.get("context", ""),
                my_previous_answer=state["answers"].get(role_id, ""),
                critiques_on_me=critiques_on_me,
                my_role=role_id,
            )

            resp = self.client.call(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            state["answers"][role_id] = resp.content
            state["confidences"][role_id] = _extract_confidence(resp.content)
            state["previous_answers"][role_id] = resp.content
            print(f"   ✅ {role_id} ({model_id}): 修订后置信度 {state['confidences'][role_id]*100:.0f}%")
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[nodes.revise] {role_id} FAILED\nError: {e}\n{tb}")
            print(f"   ❌ {role_id}: 修订失败: {e}")

        return state

    # ---------- 裁判节点（LLM-based） ----------

    def judge(self, state: DebateState) -> DebateState:
        """判断是否收敛（LLM-based judge）"""
        print(f"\n⚖️  裁判判断...")

        aggregator_cfg = self.config.get_aggregator_model()

        # 收集最近一轮的评审
        latest_critiques = state.get("critiques", {})

        user_prompt = prompts.build_auto_judge_prompt(
            question=state["question"],
            context=state.get("context", ""),
            all_answers=state["answers"],
            all_confidences=state["confidences"],
            all_critiques=latest_critiques,
            round_num=state["round"],
            max_rounds=state["max_rounds"],
        )

        try:
            resp = self.client.call(
                model_id=aggregator_cfg.id,
                system_prompt="你是一位资深投资委员会主席，擅长客观评估辩论质量并做出流程决策。",
                user_prompt=user_prompt,
                temperature=0.2,
            )

            content = resp.content.strip()
            # 尝试提取 JSON 块
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                brace_match = re.search(r'\{.*\}', content, re.DOTALL)
                if brace_match:
                    try:
                        result = json.loads(brace_match.group(0))
                    except json.JSONDecodeError:
                        result = {"action": "continue", "reason": f"JSON 解析失败: {content[:200]}", "info_needed": ""}
                else:
                    result = {"action": "continue", "reason": f"无法解析 judge 输出: {content[:200]}", "info_needed": ""}

            action = result.get("action", "continue")
            if action not in ("continue", "stop", "need_info"):
                action = "continue"

            if action == "stop":
                state["consensus_reached"] = True
                print(f"   📌 裁判判定达成共识: {result.get('reason', '')[:80]}")
            elif action == "need_info":
                state["consensus_reached"] = False
                print(f"   📌 裁判判定需要补充信息: {result.get('info_needed', '')[:80]}")
            else:
                state["consensus_reached"] = False
                state["round"] += 1
                print(f"   🔄 未达成共识，进入第 {state['round']} 轮")
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[nodes.judge] 裁判调用失败\nError: {e}\n{tb}")
            # fallback：简单启发式
            threshold = self.config.debate.consensus_threshold
            confs = state["confidences"]
            role_models = state.get("role_models", {})
            all_confident = all(confs.get(rid, 0) >= threshold for rid in role_models)
            max_rounds_reached = state["round"] >= state["max_rounds"]

            if all_confident and state["round"] >= 2:
                state["consensus_reached"] = True
                print(f"   📌 所有角色置信度 > {threshold*100:.0f}%，判定达成共识（fallback）")
            elif max_rounds_reached:
                state["consensus_reached"] = True
                print(f"   📌 达到最大轮次 ({state['max_rounds']})，结束辩论（fallback）")
            else:
                state["consensus_reached"] = False
                state["round"] += 1
                print(f"   🔄 未达成共识，进入第 {state['round']} 轮（fallback）")

        return state

    # ---------- 聚合节点 ----------

    def aggregate(self, state: DebateState) -> DebateState:
        """最终聚合输出"""
        print(f"\n📊 最终聚合...")

        try:
            aggregator_cfg = self.config.get_aggregator_model()
            if not aggregator_cfg:
                raise ValueError("未配置聚合器模型")

            user_prompt = prompts.build_aggregate_prompt(
                question=state["question"],
                context=state.get("context", ""),
                all_answers=state["answers"],
                all_confidences=state["confidences"],
                all_rounds=state["round"],
            )

            resp = self.client.call(
                model_id=aggregator_cfg.id,
                system_prompt="你是一位资深投资委员会主席，擅长综合多方观点得出平衡、专业的结论。",
                user_prompt=user_prompt,
                temperature=0.2,
            )

            state["final_answer"] = resp.content
            state["reasoning_process"] = _build_reasoning_log(state)

            # 提取分歧点（简化：用置信度差异判断）
            confs = state["confidences"]
            if len(confs) >= 2:
                vals = list(confs.values())
                if max(vals) - min(vals) > 0.2:
                    state["divergence_points"] = [
                        f"角色间置信度差异较大 ({min(vals)*100:.0f}% ~ {max(vals)*100:.0f}%)"
                    ]

            print(f"   ✅ 最终结论生成完成 ({resp.latency_ms:.0f}ms)")
        except Exception as e:
            tb = traceback.format_exc()
            llm_logger.error(f"[nodes.aggregate] 聚合失败\nError: {e}\n{tb}")
            print(f"   ❌ 聚合失败: {e}")
            # fallback：拼接所有角色回答作为最终答案
            fallback = "【聚合失败，以下为各角色原始回答】\n\n"
            for rid, ans in state.get("answers", {}).items():
                fallback += f"--- {rid} ---\n{ans}\n\n"
            state["final_answer"] = fallback
            state["reasoning_process"] = _build_reasoning_log(state)

        return state
