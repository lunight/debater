"""交互式命令行多模型辩论（流式版）

每一步都暂停，模型输出逐 token 实时显示。
"""

import sys
from debater.session import DebateSession


def print_separator(char="=", length=70):
    print(char * length)


def main():
    print("\n" + "=" * 70)
    print("🤖 多模型辩论系统 (命令行流式版)")
    print("=" * 70)
    
    question = input("\n📋 请输入问题: ").strip()
    if not question:
        print("❌ 问题不能为空")
        return
    
    print("\n场景选择:")
    print("  1. 问题分析")
    print("  2. 财务数据解读")
    print("  3. 事件分析")
    print("  4. 风险评估")
    scenario_choice = input("请选择 (1-4, 默认1): ").strip() or "1"
    scenarios = {
        "1": "question_analysis",
        "2": "financial_interpretation",
        "3": "event_analysis",
        "4": "risk_assessment",
    }
    scenario = scenarios.get(scenario_choice, "question_analysis")
    
    context = input("\n📎 上下文资料 (可选, 直接回车跳过): ").strip()
    
    print("\n" + "=" * 70)
    print("🚀 启动辩论...")
    print("=" * 70)
    
    session = DebateSession(
        question=question,
        scenario_type=scenario,
        context=context,
        max_rounds=3,
    )
    
    # 第1轮：流式生成
    print("\n🔄 第 1 轮：模型正在生成初始回答...\n")
    
    # 流式输出每个模型
    for event in session.generate_stream():
        etype = event["type"]
        
        if etype == "start":
            print_separator()
            print(f"🤖 {event['model_name']} 开始生成...")
            print_separator()
            print("\n💭 ", end="", flush=True)
        
        elif etype == "token":
            # 逐 token 打印
            token = event["token"]
            # 处理换行，保持缩进
            if "\n" in token:
                parts = token.split("\n")
                for i, part in enumerate(parts):
                    if i > 0:
                        print("\n   ", end="", flush=True)
                    print(part, end="", flush=True)
            else:
                print(token, end="", flush=True)
        
        elif etype == "model_done":
            model_id = event["model_id"]
            conf = session.confidences.get(model_id, 0)
            print(f"\n\n📊 置信度: {conf * 100:.0f}%")
            print(f"✅ {event['model_id']} 生成完成\n")
        
        elif etype == "error":
            print(f"\n❌ 错误: {event['error']}")
            # 不再 return，让其他模型继续
    
    # 主循环
    while True:
        ui = session._build_ui_state()
        
        print_separator("-", 70)
        print(f"📍 {ui['stage_info']} | {ui['round_info']}")
        print_separator("-", 70)
        
        # 检查是否结束
        if ui.get("is_done", False):
            print_separator()
            print("📊 最终统一结论：")
            print_separator()
            print(ui["final_answer"])
            print_separator()
            print("✅ 辩论结束")
            break
        
        # 用户选择
        print("\n请选择操作:")
        print("  1. ▶️  继续下一轮")
        print("  2. ✍️  我要评判")
        print("  3. 🏁 结束辩论")
        choice = input("输入 1/2/3: ").strip()
        
        if choice == "1":
            print("\n🔄 继续下一轮...")
            # 评审阶段
            print("\n📝 模型互相评审中...")
            ui = session.critique()
            
            # 显示评审
            for model_id in session._models:
                print_separator()
                print(f"🤖 {model_id} 的评审意见：")
                print_separator()
                for other_id, critique in session.critiques.get(model_id, {}).items():
                    print(f"\n对 {other_id} 的评审：")
                    print(critique[:500] + "..." if len(critique) > 500 else critique)
            
            input("\n按回车继续...")
            
            # 修订阶段
            print("\n🔄 模型正在根据评审修订...")
            ui = session.revise()
            
            # 显示修订后的输出
            for model_id in session._models:
                print_separator()
                print(f"🤖 {model_id} 修订后：")
                print_separator()
                thinking = session.thinkings.get(model_id, "")
                if thinking:
                    print("\n💭 思考过程：")
                    for line in thinking.split("\n"):
                        print(f"   {line}")
                print("\n📋 正式输出：")
                print(session.answers.get(model_id, ""))
                print(f"\n📊 置信度: {session.confidences.get(model_id, 0) * 100:.0f}%")
        
        elif choice == "2":
            critique = input("\n✍️  请输入你的评判意见:\n> ").strip()
            if critique:
                print("\n🔄 提交评判，模型正在修订...")
                ui = session.submit_user_critique(critique)
                print(f"✅ 修订完成，当前阶段: {ui['stage_info']}")
            else:
                print("⚠️ 评判意见为空，跳过")
                
        elif choice == "3":
            print("\n🔄 正在聚合最终结论...")
            ui = session.finish()
            
        else:
            print("⚠️ 无效输入，请重新选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已退出")
        sys.exit(0)
