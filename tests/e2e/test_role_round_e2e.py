"""
端到端分角色测试：观察 bull/bear 在 generate → critique → revise → judge 各轮的实际行为

使用串行交替流程（与设计文档一致）：
- generate: 双方并行生成初始观点
- critique: 串行，bear → bull
- revise: 串行，bull only
- judge: LLM-based 判断

问题：一家人在深圳，需要多少钱才能不上班打工
"""

import json
import sys
sys.path.insert(0, "/Users/lunight/dev/debater")

from debater.session import DebateSession, Stage


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_role_state(session, label):
    """打印当前各角色状态"""
    print(f"\n--- {label} ---")
    for role_id in ["bull", "bear"]:
        model_id = session.role_models.get(role_id, "unknown")
        answer = session.answers.get(role_id, "")
        thinking = session.thinkings.get(role_id, "")
        confidence = session.confidences.get(role_id, 0.0)
        summary = session.summaries.get(role_id, "")
        print(f"  [{role_id}] model={model_id}")
        print(f"    thinking_len={len(thinking)}, answer_len={len(answer)}, confidence={confidence}")
        print(f"    summary_len={len(summary)}")
        # 检查是否包含工具标签
        has_tool_tag = "<tool" in (answer or "") or "<tool" in (thinking or "")
        print(f"    has_tool_tag_in_answer={has_tool_tag}")
        # 显示前200字符预览
        preview = (answer or "")[:200].replace("\n", " ")
        print(f"    answer_preview: {preview}...")


def collect_events(generator, label):
    """收集生成器事件并统计"""
    events = []
    tool_exec_count = 0
    error_count = 0
    token_count = 0
    model_done = []
    for event in generator:
        events.append(event)
        etype = event.get("type", "")
        if etype == "tool_executing":
            tool_exec_count += 1
        elif etype == "error":
            error_count += 1
            print(f"  ⚠️ ERROR: {event.get('error', '')[:200]}")
        elif etype == "token":
            token_count += 1
        elif etype == "model_done":
            model_done.append(event.get("role_id", "unknown"))
    print(f"  事件统计: total={len(events)}, tokens={token_count}, tool_exec={tool_exec_count}, errors={error_count}, model_done={model_done}")
    return events


def verify_serial_critique_revise(events):
    """验证 critique/revise 事件流符合串行模式"""
    errors = []
    
    # 提取所有 model_done 事件
    model_dones = [e for e in events if e.get("type") == "model_done"]
    
    # 串行模式下：generate 后只有 bear critique，然后 bull revise
    # 这里 events 是 critique/revise 阶段的事件（不含 generate）
    critique_dones = [e for e in model_dones if e.get("role_id") == "bear"]
    revise_dones = [e for e in model_dones if e.get("role_id") == "bull"]
    
    if len(critique_dones) != 1:
        errors.append(f"期望 1 个 bear critique model_done，实际 {len(critique_dones)}")
    if len(revise_dones) != 1:
        errors.append(f"期望 1 个 bull revise model_done，实际 {len(revise_dones)}")
    
    # 验证顺序：critique 在 revise 之前
    if critique_dones and revise_dones:
        c_idx = events.index(critique_dones[0])
        r_idx = events.index(revise_dones[0])
        if c_idx > r_idx:
            errors.append("critique model_done 应在 revise model_done 之前")
    
    # 验证没有 bull 的 critique
    bull_critiques = [e for e in model_dones if e.get("role_id") == "bull" and e.get("step_type") == "critique"]
    if bull_critiques:
        errors.append("bull 不应产生 critique model_done（串行模式下 bull 不 critique）")
    
    return errors


def main():
    question = "一家人在深圳，需要多少钱才能不上班打工"
    scenario = "question_analysis"
    
    print_section("初始化 DebateSession")
    session = DebateSession(
        question=question,
        scenario_type=scenario,
        context="",
    )
    print(f"role_models={session.role_models}")
    print(f"models={session._models}")
    
    # ============ Round 1: GENERATE ============
    print_section("ROUND 1: GENERATE (generate_stream_parallel)")
    events = collect_events(session.generate_stream_parallel(), "generate")
    print(f"stage={session.stage}, round={session.current_round}")
    print_role_state(session, "After Generate")
    
    # 检查 bull/bear 答案质量
    for role_id in ["bull", "bear"]:
        answer = session.answers.get(role_id, "")
        if not answer or len(answer.strip()) < 50:
            print(f"  ⚠️ [{role_id}] 答案过短或为空！长度={len(answer.strip())}")
        if "<tool" in answer or "<tool_call" in answer:
            print(f"  ⚠️ [{role_id}] 答案中包含未清理的工具标签！")
    
    # ============ Round 2: CRITIQUE (串行 bear → bull) ============
    print_section("ROUND 2: CRITIQUE (critique_stream_single: bear → bull)")
    events = collect_events(session.critique_stream_single("bear", "bull"), "critique")
    print(f"stage={session.stage}, round={session.current_round}")
    print_role_state(session, "After Critique")
    
    # 检查 critiques
    print(f"\n  critiques 结构:")
    for role_id, critiques in session.critiques.items():
        print(f"    [{role_id}] targets={list(critiques.keys())}")
        for target_id, text in critiques.items():
            print(f"      {role_id} → {target_id}: len={len(text)}")
    
    # ============ Round 3: REVISE (串行 bull only) ============
    print_section("ROUND 3: REVISE (revise_stream_single: bull)")
    events = collect_events(session.revise_stream_single("bull"), "revise")
    print(f"stage={session.stage}, round={session.current_round}")
    print_role_state(session, "After Revise")
    
    # 检查修订后的变化
    print(f"\n  修订变化对比:")
    for role_id in ["bull", "bear"]:
        prev = session.previous_answers.get(role_id, "")
        curr = session.answers.get(role_id, "")
        print(f"    [{role_id}] prev_len={len(prev)} -> curr_len={len(curr)}, delta={len(curr)-len(prev)}")
    
    # 验证串行模式
    print_section("验证串行交替模式")
    # 收集 critique + revise 阶段的所有事件
    critique_events = collect_events(session.critique_stream_single("bear", "bull"), "critique_verify")
    revise_events = collect_events(session.revise_stream_single("bull"), "revise_verify")
    all_events = critique_events + revise_events
    errors = verify_serial_critique_revise(all_events)
    if errors:
        print("  ❌ 验证失败:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  ✅ 串行交替验证通过")
    
    # ============ JUDGE ============
    print_section("JUDGE (auto_judge)")
    result = session.auto_judge()
    print(f"judge_result={json.dumps(result, ensure_ascii=False, indent=2)}")
    print(f"stage_after_judge={session.stage}")
    
    # ============ Final Summary ============
    print_section("FINAL SUMMARY")
    print(f"session.current_round={session.current_round}")
    print(f"session.stage={session.stage}")
    print_role_state(session, "Final State")
    
    # 检查 known_facts
    print(f"\n  known_facts ({len(session.known_facts)}):")
    for fact in session.known_facts:
        print(f"    - {fact[:100]}...")
    
    # 保存完整状态到文件供人工审查
    output_path = "/Users/lunight/dev/debater/tests/e2e/role_round_e2e_output.json"
    state = {
        "question": question,
        "scenario": scenario,
        "final_stage": session.stage,
        "final_round": session.current_round,
        "role_models": session.role_models,
        "answers": {k: v for k, v in session.answers.items()},
        "thinkings": {k: v for k, v in session.thinkings.items()},
        "confidences": {k: v for k, v in session.confidences.items()},
        "summaries": {k: v for k, v in session.summaries.items()},
        "critiques": {k: {str(rn): t for rn, t in v.items()} for k, v in session.critiques.items()},
        "known_facts": session.known_facts,
        "final_answer": session.final_answer,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"\n  完整状态已保存到: {output_path}")


if __name__ == "__main__":
    main()
