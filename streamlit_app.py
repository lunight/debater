"""多模型辩论系统 - Streamlit 界面

产品形态：
- Phase 0: Brainstorm（头脑风暴）→ 分析问题、向用户提问
- Phase 1+: 正式辩论（生成→评审→修订）
- 历史轮次从上到下排列，最新内容追加到最下面
- 思考过程灰色显示
- 每轮末尾有 summary 框
- 底部固定事实框
"""

import os
import streamlit as st
from pathlib import Path

from debater.roles import ROLES, get_role_display_name
from debater.session import Stage

st.set_page_config(page_title="多模型辩论系统", layout="wide")

# ============ CSS ============
st.markdown("""
<style>
.thinking-box {
    color: #888888;
    font-size: 13px;
    line-height: 1.6;
    border-left: 3px solid #cccccc;
    padding-left: 12px;
    margin-bottom: 16px;
}
.model-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    background: #fafafa;
}
.model-card + .model-card {
    margin-top: 12px;
    border-top: 2px solid #d0d0d0;
}
.summary-box {
    border: 1px solid #4a90d9;
    border-radius: 6px;
    padding: 12px;
    margin-top: 12px;
    background: #f0f7ff;
    color: #333333;
    font-size: 14px;
    line-height: 1.6;
}
.summary-box h4 {
    margin: 0 0 8px 0;
    color: #2c5aa0;
    font-size: 14px;
}
.facts-box {
    border: 2px solid #2e7d32;
    border-radius: 8px;
    padding: 16px;
    margin-top: 20px;
    background: #e8f5e9;
    color: #333333;
}
.facts-box h4 {
    margin: 0 0 10px 0;
    color: #1b5e20;
}
.fact-item {
    background: white;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 4px 0;
    border-left: 3px solid #4caf50;
    color: #333333;
}
.loading-box {
    color: #999999;
    font-size: 14px;
    padding: 20px;
    text-align: center;
}
/* 两栏分隔 */
.model-col-left {
    border-right: 1px solid #e0e0e0;
    padding-right: 16px;
}
.model-col-right {
    padding-left: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============ 初始化 ============
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.session = None
    st.session_state.round_num = 0
    st.session_state.phase = "init"  # init → brainstorm → debate → done
    st.session_state.show_critique = False
    st.session_state.brainstorm_done = False
    st.session_state.new_fact = ""
    # 角色-模型映射：bull / bear 各分配一个模型
    st.session_state.role_models = {"bull": "kimi_k2.6", "bear": "deepseek_v4"}
    # 工具状态与历史记录
    st.session_state.tool_status = {}
    st.session_state.tool_history = []

TYPE_NAMES = {
    "brainstorm": "💡 头脑风暴",
    "generate": "📝 初始生成",
    "critique": "🔍 互相评审",
    "revise": "✏️ 修订完善",
    "aggregate": "📊 最终聚合",
}


def get_available_model_options():
    """获取可用模型选项列表（用于角色分配）"""
    """获取可用模型选项列表"""
    from debater.config import get_config
    cfg = get_config()
    models = cfg.get_available_models()
    return [(mid, mcfg.name) for mid, mcfg in models.items()]


# ============ 辅助函数 ============

def fmt_thinking(text):
    """格式化思考过程为灰色样式"""
    if not text:
        return ""
    escaped = text.replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")
    return f"""<div class="thinking-box">
<b>💭 思考过程</b><br><br>{escaped}
</div>"""


def fmt_output(text):
    """格式化正式输出（移除 summary 块，隐藏 tool_call/tool_result XML）"""
    if not text:
        return ""
    import re
    # 移除 <summary> 块，因为会单独显示
    clean = re.sub(r'<summary>.*?</summary>', '', text, flags=re.DOTALL)
    # 隐藏 tool_call 和 tool_result 标签（避免显示 raw XML，它们会在 sidebar 状态面板中显示）
    clean = re.sub(r'<tool(?:_call)?\s+name="[^"]+">\s*.*?\s*</tool(?:_call)?>', '', clean, flags=re.DOTALL)
    clean = re.sub(r'<tool_result\s+name="[^"]+">\s*.*?\s*</tool_result>', '', clean, flags=re.DOTALL)
    escaped = clean.replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")
    return f"""<div style="font-size:15px;line-height:1.7;">{escaped}</div>"""


def fmt_summary(text):
    """格式化 summary 为蓝色框"""
    if not text:
        return ""
    escaped = text.replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")
    return f"""<div class="summary-box">
<h4>📝 本轮要点</h4>
{escaped}
</div>"""


def render_model_card(model_id, model_name, item):
    """渲染单个模型卡片"""
    thinking = item.get(f"{model_id}_thinking", "")
    output = item.get(f"{model_id}_output", "")
    summary = item.get(f"{model_id}_summary", "")
    conf = item.get(f"{model_id}_conf", 0)
    
    content = ""
    if thinking:
        content += fmt_thinking(thinking)
    if output:
        content += fmt_output(output)
    if summary:
        content += fmt_summary(summary)
    if not content:
        content = '<div style="color:#999;">（无内容）</div>'
    
    st.markdown(f"**🤖 {model_name}**", unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)
    st.caption(f"置信度: {conf * 100:.0f}%")


def render_history_item(item, item_idx=None):
    """渲染单个历史轮次（支持动态模型）"""
    with st.container():
        st.markdown(f"**第 {item['round']} 轮 · {TYPE_NAMES.get(item['type'], item['type'])}**")
        
        # 获取该历史条目中的角色-模型映射（优先使用 item 中记录的，否则用当前配置）
        role_models = item.get("_role_models", st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"}))
        model_keys = list(role_models.values())
        cols = st.columns(len(model_keys))
        for i, (role_id, model_id) in enumerate(role_models.items()):
            safe_id = model_id.replace(".", "_")
            # 尝试从 item 中获取角色显示名
            model_name = item.get(f"{safe_id}_name", get_role_display_name(role_id, model_id))
            status = item.get(f"{safe_id}_status", "success")
            status_icon = {"success": "✅", "error": "❌", "pending": "⏳"}.get(status, "✅")
            with cols[i]:
                st.markdown(f'<div style="font-size:12px;color:#888;margin-bottom:4px;">{status_icon} {status}</div>', unsafe_allow_html=True)
                render_model_card(safe_id, model_name, {
                    f"{safe_id}_thinking": item.get(f"{safe_id}_thinking", ""),
                    f"{safe_id}_output": item.get(f"{safe_id}_output", ""),
                    f"{safe_id}_summary": item.get(f"{safe_id}_summary", ""),
                    f"{safe_id}_conf": item.get(f"{safe_id}_conf", 0),
                })
                # 重新输出按钮（只对 generate / revise 阶段显示）
                if item_idx is not None and item.get("type") in ("generate", "brainstorm", "revise"):
                    btn_key = f"retry_btn_{item['round']}_{role_id}_{item_idx}"
                    if st.button("🔄 重新输出", key=btn_key, help="对该角色重新生成输出，可补充新信息"):
                        st.session_state.retry_target = {
                            "item_idx": item_idx,
                            "role_id": role_id,
                            "round_type": item["type"],
                        }
                        st.rerun()
    
    st.divider()


def run_stream(session, stream_gen, phs, tool_phs, round_type=None):
    """通用流式收集（支持动态模型数量）
    
    Args:
        session: DebateSession
        stream_gen: 流式生成器
        phs: dict[model_id -> st.empty() placeholder] 用于 token 流
        tool_phs: dict[model_id -> st.empty() placeholder] 用于 tool 状态（独立，不被 token 覆盖）
        round_type: 轮次类型。若为 None，则从流事件中的 ``step_type`` 字段自动推断。
    """
    model_ids = list(phs.keys())
    texts = {mid: "" for mid in model_ids}
    started = {mid: False for mid in model_ids}
    role_status = {mid: "pending" for mid in model_ids}  # pending / success / error
    
    # 每轮开始时清空工具状态（避免旧状态干扰）
    st.session_state.tool_status = {}
    
    # 初始化加载状态
    for mid, ph in phs.items():
        ph.markdown('<div class="loading-box">🔄 正在连接 API...</div>', unsafe_allow_html=True)
    
    inferred_type = round_type
    for event in stream_gen:
        if inferred_type is None and event.get("step_type"):
            inferred_type = event["step_type"]
        etype = event["type"]
        
        if etype == "start":
            mid = event["model_id"]
            rid = event.get("role_id", mid)
            if rid in phs:
                phs[rid].empty()
                started[rid] = True
        
        elif etype == "token":
            mid = event["model_id"]
            rid = event.get("role_id", mid)
            acc = event["accumulated"]
            if rid in phs:
                texts[rid] = acc
                phs[rid].markdown(fmt_output(acc), unsafe_allow_html=True)
        
        elif etype == "tool_executing":
            mid = event.get("model_id", "")
            rid = event.get("role_id", mid)
            msg = event.get("message", "执行工具中...")
            executions = event.get("executions", [])
            has_error = event.get("has_error", False)
            fallback_notices = event.get("fallback_notices", [])
            errors = event.get("errors", [])
            
            # 保存到 session_state，sidebar 会持久显示
            if "tool_status" not in st.session_state:
                st.session_state.tool_status = {}
            st.session_state.tool_status[rid] = {
                "message": msg,
                "has_error": has_error,
                "errors": errors,
                "fallback_notices": fallback_notices,
                "executions": executions,
            }
            
            # 追加到 tool_history（输入栏上方的历史记录区）
            from datetime import datetime
            if "tool_history" not in st.session_state:
                st.session_state.tool_history = []
            st.session_state.tool_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "role_id": rid,
                "model_id": mid,
                "round": st.session_state.get("round_num", "?"),
                "message": msg,
                "has_error": has_error,
                "executions": executions,
            })
            
            # 同时尝试在主区域显示（作为即时反馈）
            target_ph = None
            if tool_phs:
                target_ph = tool_phs.get(rid)
            if not target_ph and phs:
                target_ph = phs.get(rid)
            if target_ph:
                # 三种状态：全部失败 / 部分回退 / 全部成功
                if has_error and not fallback_notices:
                    status_icon = "❌"
                    status_color = "#d32f2f"
                elif fallback_notices:
                    status_icon = "🔄"
                    status_color = "#f57c00"
                else:
                    status_icon = "✅"
                    status_color = "#2e7d32"
                
                # 构建每个 tool 调用的详细卡片
                exec_html = ""
                for exec in executions:
                    tname = exec.get("tool_name", "unknown")
                    params = exec.get("params", {})
                    status = exec.get("status", "unknown")
                    preview = exec.get("result_preview", "")
                    error_detail = exec.get("error_detail", "")
                    sources = exec.get("sources", [])
                    fb = exec.get("fallback_notice", "")
                    duration = exec.get("duration", 0)
                    
                    # 状态图标
                    if status == "success":
                        s_icon = "✅"
                        s_color = "#2e7d32"
                    elif status == "error":
                        s_icon = "❌"
                        s_color = "#d32f2f"
                    else:
                        s_icon = "⚠️"
                        s_color = "#f57c00"
                    
                    # 参数字符串
                    params_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
                    
                    # 耗时显示
                    duration_str = f"⏱ {duration:.1f}s" if duration else ""
                    
                    # 结果或错误
                    result_block = ""
                    if error_detail:
                        result_block = f'<div style="color:#d32f2f;font-size:12px;margin-top:4px;"><b>错误:</b> {error_detail.replace(chr(10), "<br>")}</div>'
                    elif preview:
                        result_block = f'<div style="color:#333;font-size:12px;margin-top:4px;"><b>结果:</b> {preview.replace(chr(10), "<br>")}</div>'
                    
                    # 来源
                    sources_block = ""
                    if sources:
                        sources_block = f'<div style="color:#666;font-size:11px;margin-top:4px;">📎 来源: {", ".join(sources[:3])}</div>'
                    
                    # 回退通知
                    fb_block = ""
                    if fb:
                        fb_block = f'<div style="color:#f57c00;font-size:11px;margin-top:4px;">🔄 {fb}</div>'
                    
                    exec_html += (
                        f'<div style="border:1px solid #ddd;border-radius:4px;padding:8px;margin:6px 0;background:#fafafa;">'
                        f'<div style="font-weight:bold;font-size:13px;color:#333;">'
                        f'{s_icon} {tname}({params_str})'
                        f'<span style="float:right;font-size:11px;color:#888;font-weight:normal;">{duration_str}</span></div>'
                        f'{result_block}{sources_block}{fb_block}'
                        f'</div>'
                    )
                
                # 构建详情提示（回退 + 错误摘要）
                detail_lines = []
                if fallback_notices:
                    detail_lines.append(f"回退: {'; '.join(fallback_notices)}")
                if errors:
                    detail_lines.append(f"错误: {'; '.join(errors)}")
                detail_html = "<br>".join(detail_lines)
                
                target_ph.markdown(
                    f'<div style="background:#f5f5f5;border-left:4px solid {status_color};padding:8px 12px;margin:4px 0;font-size:13px;color:#555;">'
                    f'{status_icon} <b>{msg}</b>'
                    + (f'<br><span style="font-size:12px;color:#777;">{detail_html}</span>' if detail_html else '') +
                    f'{exec_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        elif etype == "model_done":
            mid = event["model_id"]
            rid = event.get("role_id", mid)
            role_status[rid] = "success"
            if rid in phs:
                thinking = session.thinkings.get(rid, "")
                output = session.answers.get(rid, "")
                summary = session.summaries.get(rid, "")
                content = fmt_thinking(thinking) + fmt_output(output) + fmt_summary(summary)
                phs[rid].markdown(content, unsafe_allow_html=True)
        
        elif etype == "error":
            mid = event['model_id']
            rid = event.get("role_id", mid)
            role_status[rid] = "error"
            st.error(f"❌ {mid} 失败: {event['error']}")
            # 关键修复：不再 return None，让其他模型继续处理
            # 记录失败状态到 sidebar
            if "tool_status" not in st.session_state:
                st.session_state.tool_status = {}
            st.session_state.tool_status[rid] = {
                "message": f"❌ 模型调用失败: {event['error'][:100]}",
                "has_error": True,
                "errors": [event['error']],
                "fallback_notices": [],
                "executions": [],
            }
        
        elif etype == "partial_error":
            for err in event.get("errors", []):
                st.warning(f"⚠️ {err}")
            # 不返回 None，继续收集已完成的内容
    
    # 动态构建结果字典
    role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
    effective_type = inferred_type or round_type or "unknown"
    result = {
        "round": session.current_round,
        "type": effective_type,
        "type_name": TYPE_NAMES.get(effective_type, effective_type),
        "_model_keys": model_ids,
        "_role_models": dict(role_models),  # 记录创建时的角色-模型映射，防止后续模型切换导致 key 不匹配
    }
    for role_id, mid in role_models.items():
        safe_id = mid.replace(".", "_")
        result[f"{safe_id}_thinking"] = session.thinkings.get(role_id, "")
        result[f"{safe_id}_output"] = session.answers.get(role_id, "")
        result[f"{safe_id}_summary"] = session.summaries.get(role_id, "")
        result[f"{safe_id}_conf"] = session.confidences.get(role_id, 0)
        result[f"{safe_id}_status"] = role_status.get(role_id, "pending")
        # 记录模型显示名，供历史渲染使用
        cfg = session._config.models.get(mid)
        result[f"{safe_id}_name"] = cfg.name if cfg else mid
    
    return result


def render_facts_box(session):
    """渲染底部事实框"""
    if session is None:
        return
    
    with st.container():
        st.markdown("---")
        st.markdown("""<div class="facts-box">
<h4>📋 已知事实（用户补充 & 模型确认）</h4>
</div>""", unsafe_allow_html=True)
        
        facts = session.known_facts if session.known_facts else []
        if facts:
            for i, fact in enumerate(facts):
                cols = st.columns([10, 1])
                with cols[0]:
                    st.markdown(f'<div class="fact-item">{fact}</div>', unsafe_allow_html=True)
                with cols[1]:
                    if st.button("🗑️", key=f"del_fact_{i}"):
                        session.known_facts.pop(i)
                        st.rerun()
        else:
            st.caption("暂无已知事实。在下方添加或让模型在分析中提出需要补充的数据。")
        
        # 添加新事实
        new_fact = st.text_input("➕ 添加新事实", placeholder="输入一个已知事实...", key="fact_input")
        if st.button("添加", key="btn_add_fact") and new_fact.strip():
            session.known_facts.append(new_fact.strip())
            st.rerun()


def build_aggregate_item(session, round_num, role_models):
    """构建聚合阶段的历史记录项（基于角色-模型映射）"""
    item = {
        "round": round_num,
        "type": "aggregate",
        "type_name": "📊 最终聚合",
        "_role_models": role_models,
    }
    for role_id, mid in role_models.items():
        safe = mid.replace(".", "_")
        item[f"{safe}_name"] = get_role_display_name(role_id, mid)
        item[f"{safe}_output"] = session.answers.get(role_id, "")
        item[f"{safe}_conf"] = session.confidences.get(role_id, 0)
    return item


# ============ 侧边栏：Save / Load / 总结 ============
with st.sidebar:
    st.markdown("### ⚙️ 配置管理")
    
    # 扫描配置文件
    from debater.config import list_config_files
    config_files = list_config_files()
    if config_files:
        config_options = {Path(p).name: p for p in config_files}
        selected_config = st.selectbox("📄 配置文件", list(config_options.keys()), index=0)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 加载"):
                try:
                    cfg_path = config_options[selected_config]
                    # 如果已有 session，刷新 session 的配置
                    if st.session_state.session:
                        st.session_state.session.reload_config(cfg_path)
                    else:
                        # 没有 session 时只刷新全局配置
                        from debater.config import reload_config
                        reload_config(cfg_path)
                    
                    # 更新可用模型列表：保留角色配置中仍然有效的模型
                    from debater.config import get_config
                    cfg = get_config()
                    available = list(cfg.models.keys())
                    role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
                    for role_id in list(role_models.keys()):
                        if role_models[role_id] not in available:
                            # 该模型已不可用，用第一个可用模型替换
                            role_models[role_id] = available[0] if available else ""
                    st.session_state.role_models = role_models
                    
                    st.success(f"✅ 已加载: {selected_config}")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载失败: {e}")
        with c2:
            if st.button("📋 复制"):
                st.session_state.show_copy_config = True
        
        # 复制配置弹窗
        if st.session_state.get("show_copy_config", False):
            st.markdown("---")
            st.markdown("**📋 复制配置**")
            new_name = st.text_input("新配置文件名（不含 .yaml）", placeholder="config_prod", key="copy_config_name")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 确认复制", key="btn_confirm_copy"):
                    if new_name.strip():
                        try:
                            from debater.config import copy_config
                            src_path = config_options[selected_config]
                            new_path = copy_config(src_path, new_name.strip())
                            st.success(f"✅ 已复制: {Path(new_path).name}")
                            st.session_state.show_copy_config = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"复制失败: {e}")
                    else:
                        st.error("请输入文件名")
            with c2:
                if st.button("❌ 取消", key="btn_cancel_copy"):
                    st.session_state.show_copy_config = False
                    st.rerun()
    else:
        st.caption("未找到配置文件")
    
    st.markdown("---")
    st.markdown("### 🎭 角色配置")
    
    model_options = get_available_model_options()
    model_ids = [opt[0] for opt in model_options]
    model_names = [opt[1] for opt in model_options]
    
    role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
    changed = False
    
    for role_id in ["bull", "bear"]:
        role = ROLES[role_id]
        current_mid = role_models.get(role_id, model_ids[0] if model_ids else "")
        
        st.markdown(f"**{role.emoji} {role.name} ({role_id.upper()})**")
        try:
            current_idx = model_ids.index(current_mid)
        except ValueError:
            current_idx = 0
        
        selected_name = st.selectbox(
            f"{role_id}_model",
            model_names,
            index=current_idx,
            key=f"sidebar_role_select_{role_id}",
            label_visibility="collapsed",
        )
        selected_mid = model_ids[model_names.index(selected_name)]
        
        if selected_mid != current_mid:
            role_models[role_id] = selected_mid
            changed = True
            if st.session_state.session:
                st.session_state.session.set_role_model(role_id, selected_mid)
        
        st.caption(f"{role.description}")
    
    st.session_state.role_models = role_models
    
    if changed:
        st.success(f"✅ 已更新角色配置")
        st.rerun()
    
    # ============ Sidebar 工具执行状态面板（持久化，不受 rerun 影响）============
    st.markdown("---")
    st.markdown("### 🔧 工具执行状态")
    
    tool_status = st.session_state.get("tool_status", {})
    if tool_status:
        for mid, status in tool_status.items():
            has_error = status.get("has_error", False)
            fallback_notices = status.get("fallback_notices", [])
            msg = status.get("message", "")
            executions = status.get("executions", [])
            
            if has_error and not fallback_notices:
                status_icon = "❌"
                status_color = "#d32f2f"
            elif fallback_notices:
                status_icon = "🔄"
                status_color = "#f57c00"
            else:
                status_icon = "✅"
                status_color = "#2e7d32"
            
            # 构建精简卡片
            exec_lines = []
            for ex in executions:
                tname = ex.get("tool_name", "unknown")
                status_ex = ex.get("status", "unknown")
                duration = ex.get("duration", 0)
                dur_str = f"⏱ {duration:.1f}s" if duration else ""
                s_icon = "✅" if status_ex == "success" else ("❌" if status_ex == "error" else "⚠️")
                exec_lines.append(f"{s_icon} {tname} {dur_str}")
            
            exec_text = "<br>".join(exec_lines) if exec_lines else ""
            fb_text = "<br>".join(fallback_notices) if fallback_notices else ""
            
            st.markdown(
                f'<div style="border-left:3px solid {status_color};padding:6px 10px;margin:6px 0;background:#fafafa;font-size:12px;">'
                f'<b>{status_icon} {mid}</b><br><span style="color:#555;">{msg}</span>'
                + (f'<br><span style="color:#f57c00;font-size:11px;">{fb_text}</span>' if fb_text else '')
                + (f'<br><span style="color:#666;font-size:11px;">{exec_text}</span>' if exec_text else '')
                + '</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("暂无工具执行记录")
    
    # 新增模型快捷入口
    if st.button("➕ 新增模型", key="btn_new_model_sidebar"):
        st.session_state.show_new_model = True
    
    if st.session_state.get("show_new_model", False):
        st.markdown("---")
        st.markdown("**➕ 新增模型到配置**")
        new_model_id = st.text_input("模型 ID（英文，如 gpt4o）", key="new_m_id")
        new_model_name = st.text_input("显示名称", key="new_m_name")
        new_model_api_id = st.text_input("API Model ID", key="new_m_api_id")
        new_model_api_key = st.text_input("API Key", type="password", key="new_m_key")
        new_model_base_url = st.text_input("Base URL", key="new_m_url")
        c1, c2 = st.columns(2)
        with c1:
            new_model_temp = st.number_input("Temperature", value=0.3, min_value=0.0, max_value=2.0, step=0.1, key="new_m_temp")
        with c2:
            new_model_maxtok = st.number_input("Max Tokens", value=60000, min_value=1, max_value=200000, step=1000, key="new_m_maxtok")
        new_model_critique = st.selectbox("Critique Style", ["standard", "adversarial"], key="new_m_critique")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 保存模型", key="btn_save_new_model"):
                if new_model_id.strip() and new_model_api_key.strip():
                    try:
                        import yaml
                        from debater.config import load_config_raw, save_config, reload_config
                        cfg_data = load_config_raw()
                        cfg_data.setdefault("models", {})
                        cfg_data["models"][new_model_id.strip()] = {
                            "name": new_model_name.strip() or new_model_id.strip(),
                            "provider": "anthropic",
                            "model_id": new_model_api_id.strip() or new_model_id.strip(),
                            "api_key": new_model_api_key.strip(),
                            "base_url": new_model_base_url.strip(),
                            "temperature": float(new_model_temp),
                            "max_tokens": int(new_model_maxtok),
                            "role": "proposer",
                            "critique_style": new_model_critique,
                        }
                        save_config(cfg_data)
                        reload_config()
                        if st.session_state.session:
                            st.session_state.session._client.reload()
                            st.session_state.session._config = reload_config()
                        st.session_state.show_new_model = False
                        st.success(f"✅ 模型 {new_model_id.strip()} 已保存")
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败: {e}")
                else:
                    st.error("模型 ID 和 API Key 不能为空")
        with c2:
            if st.button("❌ 取消", key="btn_cancel_new_model"):
                st.session_state.show_new_model = False
                st.rerun()
    
    st.caption(f"共 {len(model_ids)} 个可用模型，上限 4 个 Debater")
    
    st.markdown("---")
    st.markdown("### 💾 辩论管理")
    
    # 扫描保存文件
    saves_dir = Path("/Users/lunight/dev/debater/saves")
    save_files = []
    if saves_dir.exists():
        save_files = sorted(saves_dir.glob("debate_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    # 加载已有辩论
    if save_files:
        options = {f.name.replace("debate_", "").replace(".json", ""): str(f) for f in save_files[:10]}
        selected = st.selectbox("📂 加载辩论", list(options.keys()), index=0)
        if st.button("📂 加载选中辩论"):
            from debater.session import DebateSession, Stage
            save_path = options[selected]
            try:
                loaded_session = DebateSession.load(save_path)
                st.session_state.session = loaded_session
                st.session_state.round_num = loaded_session.current_round
                # 根据 stage 恢复 phase
                if loaded_session.stage == Stage.DONE:
                    st.session_state.phase = "done"
                elif loaded_session.stage in [Stage.GENERATING, Stage.CRITIQUING, Stage.REVISING]:
                    st.session_state.phase = "debate"
                else:
                    st.session_state.phase = "debate"
                st.session_state.history = []  # 历史不恢复，从当前状态继续
                st.session_state.show_critique = False
                st.session_state.brainstorm_done = True
                st.success(f"✅ 已加载辩论（第 {loaded_session.current_round} 轮）")
                st.rerun()
            except Exception as e:
                st.error(f"加载失败: {e}")
    else:
        st.caption("暂无保存的辩论")
    
    st.markdown("---")
    
    # 显示已加载的 Skills
    st.markdown("### 🛠️ Skills & Tools")
    try:
        from debater.skills import SkillRegistry
        from debater.tools import ToolRegistry
        
        sr = SkillRegistry()
        sr.load()
        skills = sr.list_all()
        if skills:
            st.caption(f"已加载 {len(skills)} 个 skill")
            for s in skills[:5]:
                st.caption(f"• {s.name}")
        else:
            st.caption("未加载 skill")
        
        tr = ToolRegistry(base_dir=os.getcwd())
        tr.register_default_tools()
        tools = tr.list_all()
        if tools:
            st.caption(f"可用 tools: {', '.join(t.name for t in tools)}")
        else:
            st.caption("无可用 tools")
    except Exception:
        st.caption("Skills/Tools 加载中...")
    
    st.markdown("---")
    
    # 保存当前辩论
    session = st.session_state.get("session")
    if session and session.memory_manager:
        if st.button("💾 保存当前辩论"):
            try:
                save_path = session.save()
                st.success(f"✅ 已保存: `{Path(save_path).name}`")
            except Exception as e:
                st.error(f"保存失败: {e}")
        
        # 生成中途总结
        if st.button("📝 生成当前总结"):
            try:
                mid_path = session.memory_manager.generate_mid_obsidian(session, session.current_round)
                st.success(f"✅ 总结已生成: `{Path(mid_path).name}`")
            except Exception as e:
                st.error(f"生成失败: {e}")
    else:
        st.caption("开始辩论后可保存/总结")

# ============ 页面标题 & Tabs ============
st.title("🤖 多模型辩论系统")
st.caption("Phase 0: 头脑风暴 → Phase 1+: 正式辩论")

tab_debate, tab_config = st.tabs(["🤖 辩论", "⚙️ 配置管理"])

# ============ Tab 1: 辩论 ============
with tab_debate:

    # ============ 历史记录区 ============
    for idx, item in enumerate(st.session_state.history):
        render_history_item(item, item_idx=idx)

    # ============ 重新输出面板 ============
    retry_target = st.session_state.get("retry_target")
    if retry_target and st.session_state.session:
        session = st.session_state.session
        role_id = retry_target["role_id"]
        round_type = retry_target["round_type"]
        item_idx = retry_target.get("item_idx")
        role = ROLES.get(role_id, ROLES["bear"])
        
        st.markdown("---")
        st.markdown(f"### 🔄 重新输出 · {role.emoji} {role.name}")
        st.info(f"阶段: {TYPE_NAMES.get(round_type, round_type)} | 角色: {role.name}")
        
        extra = st.text_area("补充新信息（可选）", placeholder="提供额外的上下文或修正要求，帮助模型生成更好的输出...", key="retry_extra")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ 确认重试", type="primary", key="retry_confirm"):
                st.session_state.retry_confirmed = True
                st.session_state.retry_extra_text = extra
                st.rerun()
        with c2:
            if st.button("❌ 取消", key="retry_cancel"):
                del st.session_state.retry_target
                st.session_state.pop("retry_confirmed", None)
                st.session_state.pop("retry_extra_text", None)
                st.rerun()
        
        if st.session_state.get("retry_confirmed"):
            extra_text = st.session_state.get("retry_extra_text", "")
            st.session_state.pop("retry_confirmed", None)
            st.session_state.pop("retry_extra_text", None)
            
            with st.spinner(f"正在重新生成 {role.name} 的输出..."):
                # 清空该角色的旧状态
                session.answers[role_id] = ""
                session.thinkings[role_id] = ""
                session.summaries[role_id] = ""
                session.confidences[role_id] = 0.0
                
                # 调用重试流式方法
                retry_phs = {role_id: st.empty()}
                retry_tool_phs = {role_id: st.empty()}
                result = run_stream(
                    session,
                    session.retry_generate_role_stream(role_id, extra_context=extra_text, round_type=round_type),
                    retry_phs,
                    retry_tool_phs,
                    round_type,
                )
                
                # 更新历史记录
                if result and item_idx is not None and item_idx < len(st.session_state.history):
                    old_item = st.session_state.history[item_idx]
                    # 使用历史记录中记录的模型映射来确定写入 key（兼容模型切换后的旧历史记录）
                    historical_role_models = old_item.get("_role_models", session.role_models)
                    current_role_models = st.session_state.get("role_models", session.role_models)
                    
                    historical_model_id = historical_role_models.get(role_id, role_id)
                    historical_safe_id = historical_model_id.replace(".", "_")
                    
                    current_model_id = current_role_models.get(role_id, role_id)
                    current_safe_id = current_model_id.replace(".", "_")
                    
                    old_item[f"{historical_safe_id}_thinking"] = result.get(f"{current_safe_id}_thinking", "")
                    old_item[f"{historical_safe_id}_output"] = result.get(f"{current_safe_id}_output", "")
                    old_item[f"{historical_safe_id}_summary"] = result.get(f"{current_safe_id}_summary", "")
                    old_item[f"{historical_safe_id}_conf"] = result.get(f"{current_safe_id}_conf", 0)
                    old_item[f"{historical_safe_id}_status"] = result.get(f"{current_safe_id}_status", "success")
                
                del st.session_state.retry_target
                st.success(f"✅ {role.name} 重新生成完成")
                st.rerun()
    
    # ============ 当前活动区 ============
    st.markdown("---")
    current_title = st.empty()

    # 动态创建 placeholder，每个角色一个独立的列
    role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
    cols = st.columns(len(role_models))
    live_phs = {}
    tool_phs = {}
    for i, (role_id, mid) in enumerate(role_models.items()):
        with cols[i]:
            # 显示角色名称
            display_name = get_role_display_name(role_id, mid)
            st.markdown(f"**{display_name}**", unsafe_allow_html=True)
            tool_phs[role_id] = st.empty()  # tool 状态占位符（独立，不被 token 覆盖）
            live_phs[role_id] = st.empty()

    # ============ Tool Call 历史记录区（输入栏上方） ============
    st.markdown("---")
    st.markdown("### 🔧 工具调用记录")
    
    tool_history = st.session_state.get("tool_history", [])
    if tool_history:
        # 只显示最近 20 条，避免过长
        for record in tool_history[-20:]:
            mid = record.get("model_id", "unknown")
            ts = record.get("timestamp", "")
            msg = record.get("message", "")
            has_error = record.get("has_error", False)
            executions = record.get("executions", [])
            round_num = record.get("round", "?")
            
            if has_error:
                status_icon = "❌"
                status_color = "#d32f2f"
            elif "回退" in msg:
                status_icon = "🔄"
                status_color = "#f57c00"
            else:
                status_icon = "✅"
                status_color = "#2e7d32"
            
            # 构建执行详情
            exec_lines = []
            for ex in executions:
                tname = ex.get("tool_name", "unknown")
                status = ex.get("status", "unknown")
                duration = ex.get("duration", 0)
                preview = ex.get("result_preview", "")[:80]
                error = ex.get("error_detail", "")[:80]
                s_icon = "✅" if status == "success" else ("❌" if status == "error" else "⚠️")
                dur_str = f"⏱ {duration:.1f}s" if duration else ""
                if error:
                    exec_lines.append(f"{s_icon} {tname} {dur_str} | ❌ {error}")
                elif preview:
                    exec_lines.append(f"{s_icon} {tname} {dur_str} | {preview}")
                else:
                    exec_lines.append(f"{s_icon} {tname} {dur_str}")
            
            exec_text = "<br>".join(exec_lines) if exec_lines else ""
            
            st.markdown(
                f'<div style="border-left:3px solid {status_color};padding:6px 12px;margin:4px 0;background:#fafafa;font-size:12px;">'
                f'<b>{status_icon} {mid}</b> <span style="color:#888;font-size:11px;">第{round_num}轮 · {ts}</span><br>'
                f'<span style="color:#555;">{msg}</span>'
                + (f'<br><span style="color:#666;font-size:11px;">{exec_text}</span>' if exec_text else '')
                + '</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("暂无工具调用记录")
    
    # ============ 输入控制区（底部） ============
    st.markdown("---")

    if st.session_state.phase == "init":
        st.markdown("### 📋 第一步：输入你的问题")
    
        c1, c2 = st.columns([2, 1])
        with c1:
            question = st.text_area("问题", placeholder="例如：分析某公司2023年盈利能力...", key="q", height=80)
        with c2:
            scenario = st.selectbox("场景", [
                ("问题分析", "question_analysis"),
                ("财务数据解读", "financial_interpretation"),
                ("事件分析", "event_analysis"),
                ("风险评估", "risk_assessment"),
            ], format_func=lambda x: x[0], key="s")
    
        context = st.text_area("已知背景资料（可选）", placeholder="粘贴已有信息...", key="c")
    
        if st.button("💡 开始头脑风暴", type="primary"):
            if not question:
                st.error("请输入问题")
            else:
                from debater.session import DebateSession
                scenario_val = scenario[1] if isinstance(scenario, tuple) else scenario
                session = DebateSession(question, scenario_val, context)
                # 应用用户选择的模型配置
                # 应用用户选择的角色-模型配置
                for role_id, mid in st.session_state.role_models.items():
                    session.set_role_model(role_id, mid)
                st.session_state.session = session
                st.session_state.round_num = 1
                st.session_state.phase = "brainstorm"
                # 将初始 context 作为第一条事实
                if context.strip():
                    session.known_facts.append(f"【初始背景】{context.strip()}")
                st.rerun()


    elif st.session_state.phase == "brainstorm":
        session = st.session_state.session
    
        if not st.session_state.brainstorm_done:
            current_title.markdown("### 💡 Phase 0：头脑风暴 - 模型正在分析问题...")
        
            result = run_stream(session, session.brainstorm_stream_parallel(), live_phs, tool_phs, "brainstorm")
            if result:
                st.session_state.history.append(result)
                st.session_state.brainstorm_done = True
            st.rerun()
    
        else:
            current_title.markdown("### 💡 Phase 0：头脑风暴完成")
        
            # 显示所有角色的 brainstorm 结果
            role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
            cols = st.columns(len(role_models))
            for i, (role_id, model_id) in enumerate(role_models.items()):
                with cols[i]:
                    st.markdown(f"**{get_role_display_name(role_id, model_id)}**")
                    output = session.answers.get(role_id, "")
                    summary = session.summaries.get(role_id, "")
                    st.markdown(output)
                    if summary:
                        st.markdown(fmt_summary(summary), unsafe_allow_html=True)
        
            st.markdown("---")
            st.markdown("### 📋 请补充信息（可选）")
            st.info("模型提出了以上问题和建议。你可以补充更多信息，让后续分析更精准。")
        
            extra = st.text_area("补充背景资料", placeholder="根据模型建议，补充缺失的信息...", key="extra")
        
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 补充完成，开始正式辩论", type="primary"):
                    if extra.strip():
                        session.context += f"\n\n【用户补充信息】\n{extra}"
                        session.known_facts.append(f"【用户补充】{extra.strip()}")
                    st.session_state.phase = "debate"
                    st.session_state.round_num = 1
                    st.rerun()
            with c2:
                if st.button("⏭️ 无需补充，直接开始"):
                    st.session_state.phase = "debate"
                    st.session_state.round_num = 1
                    st.rerun()


    elif st.session_state.phase == "debate":
        session = st.session_state.session
        role_models = st.session_state.get("role_models", {"bull": "kimi_k2.6", "bear": "deepseek_v4"})
    
        # 显示当前阶段
        if session:
            ui = session._build_ui_state()
            st.info(f"📍 {ui['stage_info']} | {ui['round_info']}")
    
        # ====== 自动模式流转（优先处理）======
        if st.session_state.get("auto_mode"):
            # 自动模式状态条
            auto_col1, auto_col2 = st.columns([4, 1])
            with auto_col1:
                status_text = "🤖 自动辩论进行中..."
                if st.session_state.get("auto_pause_reason") == "need_info":
                    status_text = "⏸️ 自动辩论暂停：等待补充信息"
                st.info(status_text)
            with auto_col2:
                if st.button("⏹ 取消自动", key="btn_cancel_auto"):
                    st.session_state.auto_mode = False
                    st.session_state.auto_pause_reason = None
                    st.session_state.auto_info_needed = None
                    st.session_state.auto_judge_done = None
                    st.rerun()
            
            # 暂停等待补充信息
            if st.session_state.get("auto_pause_reason") == "need_info":
                st.warning("⚠️ 裁判模型判断需要补充关键信息才能继续")
                st.markdown(f"**需要补充：** {st.session_state.get('auto_info_needed', '')}")
                if st.button("▶️ 补充完成，继续自动辩论", key="btn_resume_auto"):
                    st.session_state.auto_pause_reason = None
                    st.session_state.auto_info_needed = None
                    st.session_state.auto_judge_done = False
                    st.rerun()
                # 不继续自动流转，等待用户操作或手动按钮
            
            else:
                # 判断是否需要 judge
                if session.stage == Stage.PAUSE_AFTER_REVISE and not st.session_state.get("auto_judge_done"):
                    current_title.markdown("### ⚖️ 自动裁判判断中...")
                    with st.spinner("裁判模型正在评估辩论质量..."):
                        try:
                            judge_result = session.auto_judge()
                        except Exception as e:
                            st.error(f"自动裁判失败: {e}")
                            judge_result = {"action": "continue", "reason": "裁判调用失败，fallback 到继续", "info_needed": ""}
                    
                    st.session_state.auto_judge_reason = judge_result.get("reason", "")
                    
                    if judge_result["action"] == "continue":
                        st.success(f"✅ 裁判决定：继续辩论。原因：{judge_result['reason'][:200]}")
                        st.session_state.auto_judge_done = True
                        st.rerun()
                    
                    elif judge_result["action"] == "stop":
                        st.success(f"🏁 裁判决定：结束辩论。原因：{judge_result['reason'][:200]}")
                        st.session_state.auto_mode = False
                        st.session_state.auto_judge_done = None
                        current_title.markdown("### 📊 正在聚合最终结论...")
                        session.finish()
                        st.session_state.history.append(build_aggregate_item(session, session.current_round, role_models))
                        st.session_state.phase = "done"
                        st.rerun()
                    
                    elif judge_result["action"] == "need_info":
                        st.warning(f"⚠️ 裁判决定：需要补充信息。原因：{judge_result['reason'][:200]}")
                        st.session_state.auto_mode = False
                        st.session_state.auto_pause_reason = "need_info"
                        st.session_state.auto_info_needed = judge_result.get("info_needed", "")
                        st.rerun()
                
                else:
                    # 执行一步（由 session 根据 stage 自动决定）
                    step_type_map = {
                        Stage.INIT: ("📝 初始生成", "generate"),
                        Stage.PAUSE_AFTER_GENERATE: ("🔍 bear 评审 bull", "critique"),
                        Stage.PAUSE_AFTER_CRITIQUE: ("✏️ bull 修订完善", "revise"),
                        Stage.PAUSE_AFTER_REVISE: ("🔍 bear 评审 bull", "critique"),
                    }
                    step_title, _ = step_type_map.get(session.stage, ("执行中", "unknown"))
                    current_title.markdown(f"### {step_title}（自动） | 第 {session.current_round} 轮")
                    result = run_stream(session, session.step_stream(), live_phs, tool_phs)
                    if result:
                        st.session_state.history.append(result)
                    
                    # 如果执行完变成 PAUSE_AFTER_REVISE，下一轮需要 judge
                    if session.stage == Stage.PAUSE_AFTER_REVISE:
                        st.session_state.auto_judge_done = False
                    st.rerun()
        
        # ====== 手动流程 ======
        if not st.session_state.get("auto_mode"):
            if session.stage == Stage.INIT:
                if st.button("🚀 开始辩论", type="primary", key="btn_start_debate"):
                    current_title.markdown(f"### 📝 第 {session.current_round} 轮：初始生成")
                    result = run_stream(session, session.step_stream(), live_phs, tool_phs)
                    if result:
                        st.session_state.history.append(result)
                    st.rerun()
            
            elif session.stage == Stage.PAUSE_AFTER_GENERATE:
                st.markdown("### 🎮 请选择下一步操作")
                c1, c2, c3, c4 = st.columns(4)
            
                if c1.button("🔍 bear 评审", key="btn_critique"):
                    current_title.markdown(f"### 🔍 第 {session.current_round} 轮：bear 评审 bull")
                    result = run_stream(session, session.step_stream(), live_phs, tool_phs)
                    if result:
                        st.session_state.history.append(result)
                    st.rerun()
            
                if c2.button("✍️ 我要评判", key="btn_user_critique"):
                    st.session_state.show_critique = True
                
                if c3.button("🤖 自动辩论", key="btn_auto"):
                    st.session_state.auto_mode = True
                    st.session_state.auto_judge_done = True  # 从 generate 进入，下一步 critique，无需 judge
                    st.rerun()
            
                if c4.button("🏁 结束辩论", key="btn_finish_gen"):
                    current_title.markdown("### 📊 正在聚合最终结论...")
                    session.finish()
                    st.session_state.history.append(build_aggregate_item(session, session.current_round, role_models))
                    st.session_state.phase = "done"
                    st.rerun()
            
            elif session.stage == Stage.PAUSE_AFTER_CRITIQUE:
                st.markdown("### 🎮 请选择下一步操作")
                c1, c2, c3, c4 = st.columns(4)
            
                if c1.button("✅ bull 修订", key="btn_revise"):
                    current_title.markdown(f"### ✏️ 第 {session.current_round} 轮：bull 修订完善")
                    result = run_stream(session, session.step_stream(), live_phs, tool_phs)
                    if result:
                        st.session_state.history.append(result)
                    st.rerun()
            
                if c2.button("✍️ 我要评判", key="btn_user_critique_2"):
                    st.session_state.show_critique = True
                
                if c3.button("🤖 自动辩论", key="btn_auto_crit"):
                    st.session_state.auto_mode = True
                    st.session_state.auto_judge_done = True  # 从 critique 进入，下一步 revise，无需 judge
                    st.rerun()
            
                if c4.button("🏁 结束辩论", key="btn_finish_crit"):
                    current_title.markdown("### 📊 正在聚合最终结论...")
                    session.finish()
                    st.session_state.history.append(build_aggregate_item(session, session.current_round, role_models))
                    st.session_state.phase = "done"
                    st.rerun()
            
            elif session.stage == Stage.PAUSE_AFTER_REVISE:
                st.markdown("### 🎮 请选择下一步操作")
                c1, c2 = st.columns(2)
            
                if c1.button("🔄 继续下一轮", key="btn_next_round"):
                    current_title.markdown(f"### 🔍 第 {session.current_round + 1} 轮：bear 评审 bull")
                    result = run_stream(session, session.step_stream(), live_phs, tool_phs)
                    if result:
                        st.session_state.history.append(result)
                    st.rerun()
            
                if c2.button("🏁 结束辩论", key="btn_finish_revise"):
                    current_title.markdown("### 📊 正在聚合最终结论...")
                    session.finish()
                    st.session_state.history.append(build_aggregate_item(session, session.current_round, role_models))
                    st.session_state.phase = "done"
                    st.rerun()
            
            # 用户评判输入
            if st.session_state.show_critique:
                st.markdown("---")
                critique = st.text_area("你的评判意见", placeholder="指出模型分析中的问题、补充遗漏信息...", key="critique_input")
                c1, c2 = st.columns([1, 3])
                with c1:
                    if st.button("提交评判", key="btn_submit"):
                        if critique.strip():
                            current_title.markdown(f"### ✏️ 第 {session.current_round} 轮：bull 根据用户评判修订")
                            result = run_stream(session, session.revise_stream_single("bull", user_critique=critique), live_phs, tool_phs, "revise")
                            if result:
                                result["type_name"] = "✏️ 修订完善（含用户评判）"
                                st.session_state.history.append(result)
                            st.session_state.show_critique = False
                            st.rerun()
                        else:
                            st.error("评判意见不能为空")
                with c2:
                    if st.button("取消", key="btn_cancel"):
                        st.session_state.show_critique = False
                        st.rerun()


    elif st.session_state.phase == "done":
        session = st.session_state.session
        if session and session.final_answer:
            st.success("✅ 辩论已结束")
            st.markdown("### 📊 最终统一结论")
            st.markdown(session.final_answer)
        
            # 生成 Obsidian 文档
            if session.memory_manager:
                obsidian_path = session.finish_debate()
                if obsidian_path:
                    st.info(f"📁 Obsidian 文档已生成: `{obsidian_path}`")
                
                    # 显示 working_memory 内容预览
                    wm = session.memory_manager.read_working_memory()
                    if wm:
                        with st.expander("📋 查看 Working Memory"):
                            st.markdown(wm)
    
        if st.button("🔄 重新开始"):
            st.session_state.history = []
            st.session_state.session = None
            st.session_state.round_num = 0
            st.session_state.phase = "init"
            st.session_state.show_critique = False
            st.session_state.brainstorm_done = False
            st.rerun()

    # ============ 底部事实框（始终显示） ============
    if st.session_state.session:
        render_facts_box(st.session_state.session)


# ============ Tab 2: 配置管理 ============
with tab_config:
    st.markdown("### ⚙️ 配置管理")
    
    import yaml
    from debater.config import load_config_raw, save_config, copy_config, list_config_files, get_config, reload_config
    from pathlib import Path
    
    # 当前加载的配置文件
    config_files = list_config_files()
    current_cfg_path = config_files[0] if config_files else None
    if current_cfg_path:
        st.info(f"当前配置文件: `{Path(current_cfg_path).name}`")
    
    # 加载原始配置数据
    try:
        cfg_data = load_config_raw(current_cfg_path)
    except Exception as e:
        st.error(f"加载配置失败: {e}")
        cfg_data = {}
    
    # --- 模型配置编辑 ---
    st.markdown("#### 🤖 模型参数")
    
    # 配置管理 tab 中新增模型
    if st.button("➕ 新增模型", key="btn_new_model_tab"):
        st.session_state.show_new_model_tab = True
    
    if st.session_state.get("show_new_model_tab", False):
        with st.expander("➕ 填写新模型信息", expanded=True):
            nm_id = st.text_input("模型 ID（英文）", key="nm_tab_id")
            nm_name = st.text_input("显示名称", key="nm_tab_name")
            nm_api_id = st.text_input("API Model ID", key="nm_tab_api_id")
            nm_key = st.text_input("API Key", type="password", key="nm_tab_key")
            nm_url = st.text_input("Base URL", key="nm_tab_url")
            c1, c2 = st.columns(2)
            with c1:
                nm_temp = st.number_input("Temperature", value=0.3, min_value=0.0, max_value=2.0, step=0.1, key="nm_tab_temp")
            with c2:
                nm_maxtok = st.number_input("Max Tokens", value=60000, min_value=1, max_value=200000, step=1000, key="nm_tab_maxtok")
            nm_critique = st.selectbox("Critique Style", ["standard", "adversarial"], key="nm_tab_critique")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 保存", key="nm_tab_save"):
                    if nm_id.strip() and nm_key.strip():
                        try:
                            cfg_data = load_config_raw(current_cfg_path)
                            cfg_data.setdefault("models", {})
                            cfg_data["models"][nm_id.strip()] = {
                                "name": nm_name.strip() or nm_id.strip(),
                                "provider": "anthropic",
                                "model_id": nm_api_id.strip() or nm_id.strip(),
                                "api_key": nm_key.strip(),
                                "base_url": nm_url.strip(),
                                "temperature": float(nm_temp),
                                "max_tokens": int(nm_maxtok),
                                "role": "proposer",
                                "critique_style": nm_critique,
                            }
                            save_config(cfg_data, current_cfg_path)
                            reload_config(current_cfg_path)
                            if st.session_state.session:
                                st.session_state.session._client.reload()
                                st.session_state.session._config = reload_config(current_cfg_path)
                            st.session_state.show_new_model_tab = False
                            st.success(f"✅ 模型 {nm_id.strip()} 已保存")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {e}")
                    else:
                        st.error("模型 ID 和 API Key 不能为空")
            with c2:
                if st.button("❌ 取消", key="nm_tab_cancel"):
                    st.session_state.show_new_model_tab = False
                    st.rerun()
    
    models_data = cfg_data.get("models", {})
    
    edited_models = {}
    for model_id, model_cfg in models_data.items():
        with st.expander(f"🤖 {model_id} — {model_cfg.get('name', model_id)}"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("名称", value=model_cfg.get("name", ""), key=f"cfg_name_{model_id}")
                provider = st.selectbox("Provider", ["anthropic", "openai", "other"], 
                                       index=["anthropic", "openai", "other"].index(model_cfg.get("provider", "anthropic")),
                                       key=f"cfg_provider_{model_id}")
                model_api_id = st.text_input("Model ID", value=model_cfg.get("model_id", ""), key=f"cfg_model_id_{model_id}")
                api_key = st.text_input("API Key", value=model_cfg.get("api_key", ""), type="password", key=f"cfg_api_key_{model_id}")
                base_url = st.text_input("Base URL", value=model_cfg.get("base_url", ""), key=f"cfg_base_url_{model_id}")
            with c2:
                temperature = st.number_input("Temperature", value=float(model_cfg.get("temperature", 0.3)), 
                                             min_value=0.0, max_value=2.0, step=0.1, key=f"cfg_temp_{model_id}")
                max_tokens = st.number_input("Max Tokens", value=int(model_cfg.get("max_tokens", 4096)), 
                                            min_value=1, max_value=200000, step=1000, key=f"cfg_maxtok_{model_id}")
                role = st.selectbox("Role", ["proposer", "aggregator"], 
                                   index=["proposer", "aggregator"].index(model_cfg.get("role", "proposer")),
                                   key=f"cfg_role_{model_id}")
                critique_style = st.selectbox("Critique Style", ["standard", "adversarial"],
                                             index=["standard", "adversarial"].index(model_cfg.get("critique_style", "standard")),
                                             key=f"cfg_critique_{model_id}")
                extra_body_str = st.text_area("Extra Body (YAML)", 
                                             value=yaml.dump(model_cfg.get("extra_body", {}), allow_unicode=True, default_flow_style=False) if model_cfg.get("extra_body") else "",
                                             key=f"cfg_extra_{model_id}")
            
            # 解析 extra_body
            extra_body = None
            if extra_body_str.strip():
                try:
                    extra_body = yaml.safe_load(extra_body_str)
                except Exception:
                    st.warning("extra_body YAML 解析失败，将忽略")
            
            edited_models[model_id] = {
                "name": name,
                "provider": provider,
                "model_id": model_api_id,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "role": role,
                "critique_style": critique_style,
            }
            if extra_body:
                edited_models[model_id]["extra_body"] = extra_body
    
    # --- 辩论配置编辑 ---
    st.markdown("#### 🎙️ 辩论参数")
    debate_data = cfg_data.get("debate", {})
    c1, c2 = st.columns(2)
    with c1:
        consensus_threshold = st.number_input("共识阈值", value=float(debate_data.get("consensus_threshold", 0.85)),
                                             min_value=0.0, max_value=1.0, step=0.05, key="cfg_consensus")
    with c2:
        min_confidence = st.number_input("最小置信度", value=float(debate_data.get("min_confidence", 0.70)),
                                        min_value=0.0, max_value=1.0, step=0.05, key="cfg_min_conf")
    
    edited_debate = {
        "consensus_threshold": consensus_threshold,
        "min_confidence": min_confidence,
    }
    
    # --- 保存按钮 ---
    st.markdown("---")
    save_col1, save_col2 = st.columns([1, 3])
    with save_col1:
        if st.button("💾 保存配置", type="primary", key="btn_save_config"):
            new_cfg = {
                "models": edited_models,
                "debate": edited_debate,
                "scenarios": cfg_data.get("scenarios", {}),
            }
            try:
                save_config(new_cfg, current_cfg_path)
                # 重新加载全局配置
                reload_config(current_cfg_path)
                # 刷新 session 的客户端
                if st.session_state.session:
                    st.session_state.session._client.reload()
                    st.session_state.session._config = get_config()
                st.success("✅ 配置已保存并重新加载")
                st.rerun()
            except Exception as e:
                st.error(f"保存失败: {e}")
    
    with save_col2:
        # 以 YAML 格式预览
        with st.expander("📄 查看完整配置 YAML"):
            preview_cfg = {
                "models": edited_models,
                "debate": edited_debate,
                "scenarios": cfg_data.get("scenarios", {}),
            }
            st.code(yaml.dump(preview_cfg, allow_unicode=True, sort_keys=False, default_flow_style=False), language="yaml")
