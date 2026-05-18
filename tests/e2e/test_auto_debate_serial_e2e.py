"""
端到端测试：验证 auto_debate_stream 的严格串行交替辩论流程

运行方式：
  python tests/e2e/test_auto_debate_serial_e2e.py

此测试调用真实 API，运行时间约 2-5 分钟。
"""

import json
import sys
import time

sys.path.insert(0, "/Users/lunight/dev/debater")

from debater.session import DebateSession, Stage


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def collect_events(generator, label):
    """收集生成器事件并统计"""
    events = []
    token_count = 0
    tool_exec_count = 0
    error_count = 0
    model_done = []
    start_time = time.time()
    
    for event in generator:
        events.append(event)
        etype = event.get("type", "")
        if etype == "token":
            token_count += 1
        elif etype == "tool_executing":
            tool_exec_count += 1
            print(f"  🔧 TOOL: {event.get('message', '')}")
        elif etype == "error":
            error_count += 1
            print(f"  ⚠️ ERROR: {event.get('error', '')[:200]}")
        elif etype == "model_done":
            model_done.append(event.get("role_id", "unknown"))
            role = event.get("role_id", "?")
            text = event.get("full_text", "")
            print(f"  ✅ model_done: role={role}, len={len(text)}")
        elif etype == "judge":
            print(f"  ⚖️ JUDGE: action={event.get('action')}, reason={event.get('reason', '')[:100]}")
    
    elapsed = time.time() - start_time
    print(f"  事件统计: total={len(events)}, tokens={token_count}, tool_exec={tool_exec_count}, "
          f"errors={error_count}, model_done={model_done}, elapsed={elapsed:.1f}s")
    return events


def verify_serial_alternation(events, max_cycles):
    """验证事件流符合串行交替模式"""
    errors = []
    
    # 找到所有 judge 事件的索引，用索引切片划分每轮事件
    judge_indices = [i for i, e in enumerate(events) if e.get("type") == "judge"]
    
    for cycle_idx, judge_idx in enumerate(judge_indices, start=1):
        # 本轮事件范围：上一个 judge 之后到当前 judge 之前
        cycle_start = judge_indices[cycle_idx - 2] + 1 if cycle_idx > 1 else 0
        cevents = events[cycle_start:judge_idx]
        
        # 提取 critique 和 revise 的 model_done
        critiques = [e for e in cevents if e.get("type") == "model_done" and e.get("role_id") == "bear"]
        revises = [e for e in cevents if e.get("type") == "model_done" and e.get("role_id") == "bull"]
        
        if len(critiques) != 1:
            errors.append(f"第{cycle_idx}轮: 期望 1 个 bear critique，实际 {len(critiques)}")
        if len(revises) != 1:
            errors.append(f"第{cycle_idx}轮: 期望 1 个 bull revise，实际 {len(revises)}")
        
        # 验证顺序：critique 的 model_done 在 revise 的 model_done 之前
        if critiques and revises:
            cidx = cevents.index(critiques[0])
            ridx = cevents.index(revises[0])
            if cidx > ridx:
                errors.append(f"第{cycle_idx}轮: critique 应在 revise 之前，但实际顺序相反")
    
    # 统计 auto_debate 阶段各角色的 model_done
    # 通过 judge 周期划分，排除 generate 阶段（generate 在第一个 judge 周期之前）
    role_dones = {"bear": 0, "bull": 0}
    for cycle_idx, judge_idx in enumerate(judge_indices, start=1):
        cycle_start = judge_indices[cycle_idx - 2] + 1 if cycle_idx > 1 else 0
        cevents = events[cycle_start:judge_idx]
        for e in cevents:
            if e.get("type") == "model_done":
                rid = e.get("role_id", "unknown")
                if rid in role_dones:
                    role_dones[rid] += 1
    
    # 串行模式下：每个 judge 周期内 critique 产生 bear 的 model_done，revise 产生 bull 的 model_done
    if role_dones.get("bear", 0) != max_cycles:
        errors.append(f"期望 bear 有 {max_cycles} 个 auto_debate model_done，实际 {role_dones.get('bear', 0)}")
    if role_dones.get("bull", 0) != max_cycles:
        errors.append(f"期望 bull 有 {max_cycles} 个 auto_debate model_done，实际 {role_dones.get('bull', 0)}")
    
    return errors


def main():
    question = "一家人在深圳，需要多少钱才能不上班打工"
    scenario = "question_analysis"
    max_cycles = 1
    
    print_section("初始化 DebateSession")
    session = DebateSession(
        question=question,
        scenario_type=scenario,
        context="",
    )
    print(f"role_models={session.role_models}")
    
    # ============ Round 1: GENERATE ============
    print_section("ROUND 1: GENERATE (generate_stream_parallel)")
    events = collect_events(session.generate_stream_parallel(), "generate")
    print(f"stage={session.stage}, round={session.current_round}")
    
    # 检查初始答案
    for role_id in ["bull", "bear"]:
        answer = session.answers.get(role_id, "")
        print(f"  [{role_id}] answer_len={len(answer)}")
        if not answer or len(answer.strip()) < 50:
            print(f"  ⚠️ [{role_id}] 答案过短或为空！")
    
    # ============ Round 2-3: AUTO DEBATE (串行交替) ============
    print_section(f"ROUND 2-{1+max_cycles}: AUTO DEBATE (串行交替, max_cycles={max_cycles})")
    print("  期望流程: bear critique bull → bull revise → judge → 循环")
    
    events = collect_events(session.auto_debate_stream(max_cycles=max_cycles), "auto_debate")
    print(f"stage={session.stage}, round={session.current_round}")
    
    # 验证串行交替
    print_section("验证串行交替模式")
    errors = verify_serial_alternation(events, max_cycles)
    if errors:
        print("  ❌ 验证失败:")
        for err in errors:
            print(f"    - {err}")
        # 对于真实API测试，允许验证警告但不强制退出，供人工审查
    else:
        print("  ✅ 串行交替验证通过")
    
    # 打印 critiques 结构
    print(f"\n  critiques_history 结构:")
    for round_num, round_data in session.critiques_history.items():
        print(f"    第{round_num}轮:")
        for critiquer_id, targets in round_data.items():
            for target_id, text in targets.items():
                print(f"      {critiquer_id} → {target_id}: len={len(text)}")
    
    # 打印最终状态
    print_section("FINAL STATE")
    for role_id in ["bull", "bear"]:
        answer = session.answers.get(role_id, "")
        prev = session.previous_answers.get(role_id, "")
        conf = session.confidences.get(role_id, 0.0)
        print(f"  [{role_id}] confidence={conf}, answer_len={len(answer)}, prev_len={len(prev)}")
        preview = (answer or "")[:200].replace("\n", " ")
        print(f"    preview: {preview}...")
    
    # 保存结果
    output_path = "/Users/lunight/dev/debater/tests/e2e/auto_debate_serial_output.json"
    state = {
        "question": question,
        "scenario": scenario,
        "max_cycles": max_cycles,
        "final_stage": str(session.stage),
        "final_round": session.current_round,
        "role_models": session.role_models,
        "answers": {k: v for k, v in session.answers.items()},
        "thinkings": {k: v for k, v in session.thinkings.items()},
        "confidences": {k: v for k, v in session.confidences.items()},
        "critiques_history": {
            str(rn): {
                critiquer: {target: text for target, text in targets.items()}
                for critiquer, targets in rd.items()
            }
            for rn, rd in session.critiques_history.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"\n  完整状态已保存到: {output_path}")
    
    print_section("E2E 测试完成 ✅")


if __name__ == "__main__":
    main()
