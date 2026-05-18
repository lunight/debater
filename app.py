"""多模型辩论系统 - Gradio 可视化界面 (带日志版)"""

import gradio as gr
import traceback
from debater.session import DebateSession


def _log(msg):
    """打印日志到终端"""
    print(f"[DEBATER] {msg}")


def start_debate(question, scenario, context):
    _log("=== 开始辩论被点击 ===")
    try:
        if not question or not question.strip():
            _log("错误: 问题为空")
            return _error_state("❌ 请输入问题")
        
        _log(f"问题: {question[:50]}...")
        _log(f"场景: {scenario}")
        
        session = DebateSession(
            question=question,
            scenario_type=scenario,
            context=context or "",
            max_rounds=3,
        )
        _log("会话创建成功")
        
        _log("调用 generate()...")
        ui = session.generate()
        _log(f"generate() 完成，stage={session.stage}, round={session.current_round}")
        _log(f"kimi_output 长度: {len(ui.get('kimi_output', ''))}")
        _log(f"deepseek_output 长度: {len(ui.get('deepseek_output', ''))}")
        
        return _to_outputs(session, ui)
        
    except Exception as e:
        _log(f"!!! 异常: {e}")
        traceback.print_exc()
        return _error_state(f"❌ 启动失败: {str(e)}\n\n{traceback.format_exc()}")


def on_continue(session):
    _log("=== 继续下一轮被点击 ===")
    try:
        if session is None:
            return _error_state("请先点击'开始辩论'")
        _log(f"当前 stage: {session.stage}")
        ui = session.continue_next()
        _log(f"继续完成，stage={session.stage}")
        return _to_outputs(session, ui)
    except Exception as e:
        _log(f"!!! 异常: {e}")
        traceback.print_exc()
        return _error_state(f"❌ 继续失败: {str(e)}\n\n{traceback.format_exc()}")


def on_critique_show(session):
    _log("=== 我要评判被点击 ===")
    try:
        if session is None:
            return _error_state("请先点击'开始辩论'")
        ui = session._build_ui_state()
        _log(f"显示评判输入框")
        return (
            session,
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(value=f"{ui['stage_info']} - 请输入你的评判意见"),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), gr.update(value=""),
            gr.update(visible=False),
        )
    except Exception as e:
        _log(f"!!! 异常: {e}")
        traceback.print_exc()
        return _error_state(f"❌ 失败: {str(e)}")


def on_critique_submit(session, text):
    _log("=== 提交评判被点击 ===")
    try:
        if session is None:
            return _error_state("请先点击'开始辩论'")
        if not text or not text.strip():
            return _error_state("评判意见不能为空")
        _log(f"用户评判: {text[:50]}...")
        ui = session.submit_user_critique(text)
        _log(f"评判提交完成，stage={session.stage}")
        return _to_outputs(session, ui)
    except Exception as e:
        _log(f"!!! 异常: {e}")
        traceback.print_exc()
        return _error_state(f"❌ 评判提交失败: {str(e)}\n\n{traceback.format_exc()}")


def on_finish(session):
    _log("=== 结束辩论被点击 ===")
    try:
        if session is None:
            return _error_state("请先点击'开始辩论'")
        _log("调用 finish()...")
        ui = session.finish()
        _log(f"聚合完成，final_answer 长度: {len(ui.get('final_answer', ''))}")
        return _to_outputs(session, ui)
    except Exception as e:
        _log(f"!!! 异常: {e}")
        traceback.print_exc()
        return _error_state(f"❌ 聚合失败: {str(e)}\n\n{traceback.format_exc()}")


# ============ 辅助函数 ============

def _to_outputs(session, ui):
    """统一构建输出"""
    _log("构建输出...")
    
    thinking = lambda t: f"> 💭 **思考过程**\n> \n> {t.replace(chr(10), chr(10)+'> ')}" if t else ""
    
    show_controls = ui.get("stage", "") in [
        "pause_after_generate", "pause_after_critique", "pause_after_revise"
    ]
    is_done = ui.get("is_done", False)
    
    _log(f"show_controls={show_controls}, is_done={is_done}")
    _log(f"kimi_conf={ui.get('kimi_conf', '-')}, deepseek_conf={ui.get('deepseek_conf', '-')}")
    _log(f"stage_info={ui.get('stage_info', '')[:50]}...")
    
    return (
        session,
        thinking(ui.get("kimi_thinking", "")),
        ui.get("kimi_output", ""),
        thinking(ui.get("deepseek_thinking", "")),
        ui.get("deepseek_output", ""),
        ui.get("kimi_conf", "-"),
        ui.get("deepseek_conf", "-"),
        ui.get("stage_info", ""),
        ui.get("round_info", "-"),
        ui.get("final_answer", ""),
        gr.update(visible=show_controls),
        gr.update(visible=show_controls),
        gr.update(visible=show_controls),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=is_done),
    )


def _error_state(msg):
    """错误状态"""
    _log(f"错误状态: {msg[:100]}...")
    return (
        None,
        "", "", "", "",
        "-", "-",
        msg, "-", "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
    )


# ============ UI ============

def create_app():
    with gr.Blocks(title="多模型辩论系统") as app:
        
        gr.Markdown("# 🤖 多模型辩论系统")
        gr.Markdown("kimi-for-coding vs deepseek-v4-pro")
        
        # 输入
        question = gr.Textbox(label="📋 问题", lines=2, placeholder="输入你要分析的问题...")
        scenario = gr.Dropdown(
            label="场景",
            choices=[("问题分析","question_analysis"),("财务数据解读","financial_interpretation"),
                     ("事件分析","event_analysis"),("风险评估","risk_assessment")],
            value="question_analysis",
        )
        context = gr.Textbox(label="📎 上下文（可选）", lines=3, placeholder="粘贴财务数据、新闻等...")
        start_btn = gr.Button("🚀 开始辩论", variant="primary")
        
        gr.Markdown("---")
        
        # 模型输出
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 kimi-for-coding")
                kimi_think = gr.Markdown()
                kimi_out = gr.Markdown()
                kimi_conf = gr.Textbox(label="置信度", interactive=False)
            
            with gr.Column():
                gr.Markdown("### 🤖 deepseek-v4-pro")
                ds_think = gr.Markdown()
                ds_out = gr.Markdown()
                ds_conf = gr.Textbox(label="置信度", interactive=False)
        
        # 阶段信息
        stage_info = gr.Textbox(label="当前阶段", value="准备就绪", interactive=False)
        round_info = gr.Textbox(label="轮次", value="-", interactive=False)
        
        # 控制按钮
        with gr.Row():
            btn_continue = gr.Button("▶️ 继续下一轮", visible=False)
            btn_critique = gr.Button("✍️ 我要评判", visible=False)
            btn_finish = gr.Button("🏁 结束辩论", visible=False)
        
        # 评判输入
        with gr.Row(visible=False) as critique_box:
            critique_text = gr.Textbox(label="你的评判意见", lines=3, scale=4)
            critique_submit = gr.Button("提交评判", variant="primary", scale=1)
        
        # 最终结果
        with gr.Row(visible=False) as final_box:
            final_out = gr.Markdown(label="最终结论")
        
        # 状态
        session_state = gr.State(None)
        
        # 输出列表（共 16 个）
        all_out = [
            session_state,
            kimi_think, kimi_out, ds_think, ds_out,
            kimi_conf, ds_conf,
            stage_info, round_info, final_out,
            btn_continue, btn_critique, btn_finish,
            critique_box, critique_text, final_box,
        ]
        
        # 绑定事件
        start_btn.click(fn=start_debate, inputs=[question, scenario, context], outputs=all_out)
        btn_continue.click(fn=on_continue, inputs=[session_state], outputs=all_out)
        btn_critique.click(fn=on_critique_show, inputs=[session_state], outputs=all_out)
        critique_submit.click(fn=on_critique_submit, inputs=[session_state, critique_text], outputs=all_out)
        btn_finish.click(fn=on_finish, inputs=[session_state], outputs=all_out)
    
    return app


if __name__ == "__main__":
    _log("启动 Gradio 服务...")
    app = create_app()
    _log("界面创建完成，准备 launch...")
    app.launch(share=True, server_name="127.0.0.1", server_port=7860, show_error=True)
