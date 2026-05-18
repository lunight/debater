"""Prompt 模板

针对股票分析四大场景定义专业的系统角色和提示词模板。
"""

from typing import Dict

from .roles import ROLES, get_role

# ==================== 场景角色定义 ====================

SCENARIO_ROLES: Dict[str, str] = {
    "question_analysis": """你是一位资深金融分析师，擅长对复杂商业和投资问题进行结构化分析。
你的分析风格是：逻辑严密、多角度审视、重视数据支撑、敢于提出反直觉观点。
在团队讨论中，你会认真审视同事的分析，指出逻辑漏洞或遗漏维度，同时开放地接受合理质疑。""",

    "financial_interpretation": """你是一位资深财务分析师，精通财务报表分析和财务建模。
你擅长从资产负债表、利润表、现金流量表中发现关键信号，识别财务健康状况和潜在风险。
在团队讨论中，你会严格检查同事对财务数据的解读是否准确，计算是否有误，并补充关键的财务比率分析。""",

    "event_analysis": """你是一位资深行业研究员，擅长分析突发事件对公司和行业的连锁影响。
你关注事件的直接效应、间接效应、长期结构性变化，以及市场反应与基本面的背离。
在团队讨论中，你会挑战同事对事件影响程度的判断，提出被忽视的传导路径，并校准时间维度上的影响预期。""",

    "risk_assessment": """你是一位资深风险管理专家，擅长识别、量化和评估各类投资与经营风险。
你的特点是：保守但不悲观，注重尾部风险，关注风险之间的相关性和级联效应。
在团队讨论中，你会指出同事对风险的低估，提出压力测试情景，并评估风险缓释措施的有效性。""",
}


# 质疑者角色（专用于 deepseek 的对抗性评审）
ADVERSARIAL_ROLES: Dict[str, str] = {
    "question_analysis": """你是一位严格的审查者，专门负责找出分析中的错误、漏洞和遗漏。
你不是来赞同别人的，你是来挑错的。你的任务是确保最终结论经得起检验。
你必须：核查事实、检验逻辑、确保不跑题、检查是否有遗漏的问题未回答。""",

    "financial_interpretation": """你是一位严格的财务审计师，专门负责找出财务分析中的错误。
你不是来赞同别人的，你是来挑错的。任何数据引用、计算过程、比率解读都必须经得起检验。
你必须：核查数据来源、检验计算、确保不跑题、检查是否有遗漏的财务指标未分析。""",

    "event_analysis": """你是一位严格的事实核查员，专门负责找出事件分析中的夸大、误判和遗漏。
你不是来赞同别人的，你是来挑错的。任何因果推断、影响评估都必须有充分依据。
你必须：核查事实、检验逻辑链条、确保不跑题、检查是否有遗漏的影响路径未考虑。""",

    "risk_assessment": """你是一位严格的风险审计师，专门负责找出风险评估中的盲区和低估。
你不是来赞同别人的，你是来挑错的。任何风险判断都必须经过压力测试。
你必须：核查风险识别完整性、检验缓释措施有效性、确保不跑题、检查是否有遗漏的风险场景未覆盖。""",
}


# ==================== 生成节点 Prompt ====================

# 模块级常量：工具使用策略（与工具列表解耦，可独立注入 user prompt）
TOOL_USAGE_STRATEGY = """
【工具使用策略】
1. **先独立思考**：在调用任何工具之前，先基于已有信息和你的专业知识构建完整的分析框架。
2. **识别信息缺口**：明确列出你需要但不知道的具体数据点（如"某公司2024年Q3营收具体数字"）。
3. **按需调用工具**：只在你确定缺少某个**具体事实或数据**时才调用搜索工具，不要把整个问题丢给搜索。
4. **调用格式要求（极其重要）**：
   - 当你决定调用工具时，**直接输出 XML 标签**，系统会自动解析执行
   - **必须严格使用 `<tool_call>` 标签格式**，不要写成 `<tool>` 或其他变体
   - **禁止写自然语言描述意图**，如"我准备用搜索工具...""让我查一下..."——这些不会被识别
   - 正确格式：`<tool_call name="web_search"><query>关键词</query></tool_call>`
5. **输出工具后必须继续分析（非常重要）**：
   - ⚠️ **只输出工具标签是不够的！** 系统执行工具后会再次调用你，但你第一次输出时就应该包含自己的分析思路
   - 理想做法：先给出你的初步分析框架，在需要数据支撑的地方插入工具调用
   - 工具标签只是你分析中的一个"数据请求点"，前后都应该有你自己的推理和观点
6. **正确 vs 错误示例**：

   ❌ **错误输出 1：只写自然语言（会被系统忽略）**：
   ```
   不过我需要先搜索一下深圳的生活成本、人均消费支出等数据作为背景参考。
   ```

   ❌ **错误输出 2：只输出工具标签，没有自己的分析（不合格）**：
   ```
   <tool_call name="web_search"><query>深圳 人均消费支出 2024</query></tool_call>
   <tool_call name="web_search"><query>深圳 房价 2024</query></tool_call>
   ```

   ✅ **正确输出 A：先分析，再按需插入工具**：
   ```
   针对这个问题，我计划从三个维度展开分析：
   1. 深圳三口之家的月均生活成本（含房贷/租金、教育、日常开销）
   2. 基于 FIRE 4%法则计算所需资产规模
   3. 不同资产配置方案下的可行性评估

   目前我缺少的具体数据：深圳2024年家庭月均消费支出、平均房贷利率。
   <tool_call name="web_search"><query>深圳 三口之家 每月支出 2024 房贷 教育</query></tool_call>

   在等待数据的同时，我先基于已有知识构建分析框架...
   ```

   ✅ **正确输出 B：不需要工具，直接给出完整分析**：
   ```
   基于已有信息，我的分析如下...
   （完整推理过程，不提及搜索）
   ```

7. **禁止行为**：
   - 不要直接把用户的问题原文复制给搜索工具
   - 不要在没有分析框架前就调用工具
   - 不要连续多次调用同一个工具查同一个主题
   - 绝对不要只输出工具标签而不给出自己的分析
   - **绝对不要写"我需要搜索一下""让我查一下"之类的过渡句，要么直接输出 XML 标签，要么直接给出分析**
"""


def build_generate_prompt(
    scenario_type: str,
    question: str,
    context: str,
    round_num: int,
    memory_context: str = "",
    skill_context: str = "",
    tool_context: str = "",
    debater_role: str = "",
    include_tool_strategy: bool = True,
) -> str:
    """构建初始生成/修订阶段的 prompt
    
    Args:
        include_tool_strategy: 是否在 user prompt 中注入 ❌/✅ 正误示例和策略要求。
            当 tool_context 已放在 system_prompt 中时，设 tool_context="" 并保留
            include_tool_strategy=True，可避免工具说明重复同时保留策略约束。
    """
    
    role = SCENARIO_ROLES.get(scenario_type, SCENARIO_ROLES["question_analysis"])
    
    # 注入辩论角色定义（如果提供了）
    role_prompt = ""
    if debater_role:
        role_obj = get_role(debater_role)
        role_prompt = f"""
【你的角色定位】
{role_obj.generate_prompt}
---
"""
    
    context_section = f"""
【补充背景资料】
{context}
""" if context.strip() else ""

    memory_section = f"""
【此前辩论摘要】
{memory_context}
""" if memory_context.strip() else ""

    skill_section = f"""
{skill_context}
""" if skill_context.strip() else ""

    tool_section = f"""
{tool_context}
""" if tool_context.strip() else ""

    tool_usage_strategy = TOOL_USAGE_STRATEGY if include_tool_strategy else ""

    summary_instruction = """
【输出末尾总结要求】
在回答的最后，必须用以下格式输出一个总结块（方便系统提取展示给用户）：

<summary>
【需要补充的数据】
- 列出你认为还缺少、需要用户提供的关键数据或信息
- 如果没有，写"无"

【存在的问题/不确定性】
- 列出你当前分析中存在的不确定性、假设前提或逻辑盲区
- 如果没有，写"无"

【当前思考结论】
- 用 2-3 句话概括你当前对该问题的核心判断
</summary>
"""

    if round_num == 1:
        return f"""{role}
{role_prompt}
现在有一个问题需要你分析。

【问题】
{question}
{context_section}

请给出你的专业分析。要求：
1. 先完整展示你的思考推理过程（放在 <thinking> 标签内）
2. 然后给出结构清晰的正式分析
3. 给出明确的结论或判断
4. 说明你的置信度（0%~100%）以及依据
5. 标注你可能存在信息不足或不确定的地方
{memory_section}
{skill_section}
{tool_section}
{tool_usage_strategy}
{summary_instruction}

请用中文回答，格式如下：

<thinking>
（你的完整推理思考过程，可以包括：
- 对问题的拆解和理解
- 关键数据的提取和计算
- 不同角度的权衡和排除
- 推理中的犹豫和修正
- 最终判断的形成逻辑）
</thinking>

**分析过程**：
（正式分析内容）

**结论**：
（明确结论）

**置信度**：X%
（说明依据）

**不确定点**：
（如有）

<summary>
...
</summary>
"""
    else:
        # 修订轮次，会传入评审意见
        return f"""{role}
{role_prompt}
这是第 {round_num} 轮分析。请根据上一轮收到的评审意见，修正或完善你的分析。

【问题】
{question}
{context_section}
{memory_section}
{skill_section}
{tool_section}
{tool_usage_strategy}
{summary_instruction}

请用中文回答，格式同上（包含 <thinking>、正式分析、置信度、<summary> 总结块）。
"""


# ==================== 评审节点 Prompt ====================

# 独立分析指导（要求审查者先形成自己的观点，再评审他人）
INDEPENDENT_ANALYSIS_INSTRUCTION = """
【独立分析要求】

在评审他人之前，你必须先形成自己的独立判断：

1. **独立思考**：基于原始问题和已知信息，快速构建你自己的分析框架和初步观点。
   - 不要先看对方的答案，先问自己：如果由我独自分析这个问题，我会怎么看？
   - 列出你认为的关键维度、核心变量和最重要的判断依据。

2. **刻意差异化**：如果你的初步观点与对方高度一致，主动寻找一个不同的切入角度。
   - 对方从财务角度分析，你就从行业竞争或政策风险角度审视。
   - 对方乐观，你至少尝试构造一个悲观情景来测试其结论的稳健性。
   - 差异化不是为反对而反对，而是确保问题被多维度审视。

3. **基于独立观点进行评审**：
   - 评审不是"挑错游戏"，而是"我的分析 vs 你的分析"的对比。
   - 如果对方在某个点上比我分析得更深，我承认并吸收。
   - 如果对方遗漏了我认为关键的维度，我明确指出并补充我的推理。
"""

# 对抗性检查清单（注入 critique prompt）
ADVERSARIAL_CHECKLIST = """
你在评审时，必须执行以下五项检查：

【事实核查】
- 对方引用的数据、事实是否准确？
- 是否有数据来源？如果没有，明确标记为"未经证实"
- 如果你知道正确数据，请指出

【逻辑检查】
- 推理链条是否完整？是否有跳跃？
- 前提假设是否合理？是否隐含了未声明的假设？
- 因果推断是否成立？是否存在混淆相关与因果？

【深度追问（Relentless Grill）】
对每个关键论证点， relentless 追问直到触及底层：
- 对方的每个核心结论，连续追问"为什么"——这一层的依据是什么？再追问"为什么"——更深层的依据是什么？直到触及原始证据或不可再分的假设
- 遍历决策树的每个关键分支：对方的分析是否覆盖了所有重要路径？有没有遗漏的分支未考虑？（例如：对方只分析了乐观情景，是否考虑了基准/悲观情景？）
- 如果对方使用了模糊、超载或未定义的术语，立即要求澄清："你说'XX'，具体指什么？"
- 如果对方涉及技术实现、代码结构或系统设计的断言，检查其是否与事实一致；发现矛盾立即指出

【跑题检查】
- 对方回答是否紧扣原始问题？
- 是否引入了无关议题？
- 核心问题是否被回避？

【漏答检查】
- 用户（或历史评审）提出的每个问题，对方是否都回应了？
- 如果有遗漏，明确列出"未回应的问题"
"""


def build_critique_prompt(
    scenario_type: str,
    question: str,
    context: str,
    my_answer: str,
    other_answers: Dict[str, str],
    memory_context: str = "",
    critique_history: str = "",
    skill_context: str = "",
    tool_context: str = "",
    critiquer_role: str = "",
) -> str:
    """构建标准评审阶段的 prompt"""
    
    role = SCENARIO_ROLES.get(scenario_type, SCENARIO_ROLES["question_analysis"])
    
    # 注入评审者角色定义（如果提供了）
    role_prompt = ""
    if critiquer_role:
        role_obj = get_role(critiquer_role)
        role_prompt = f"""
【你的评审角色定位】
{role_obj.critique_prompt}
---
"""
    
    others_section = "\n\n".join([
        f"【{name} 的分析】\n{ans}"
        for name, ans in other_answers.items()
    ])
    
    memory_section = f"""
【此前辩论摘要】
{memory_context}
""" if memory_context.strip() else ""
    
    history_section = f"""
【历史评审记录】
{critique_history}

> 提示：以上是以往各轮次的评审摘要。请在本次评审中注意：
> - 避免重复指出对方已经修正过的问题
> - 关注评审意见的演进和变化趋势
> - 如果对方对同一问题的回应有进步，请给予认可
""" if critique_history.strip() else ""
    
    skill_section = f"""
{skill_context}
""" if skill_context.strip() else ""

    tool_section = f"""
{tool_context}
""" if tool_context.strip() else ""
    
    # 根据目标数量动态调整措辞（串行模式下通常只有1个目标）
    target_count = len(other_answers)
    if target_count == 1:
        target_name = list(other_answers.keys())[0]
        review_intro = f"""请按以下步骤对 **{target_name}** 的分析师进行评审：

> 重要提示：以上【你的分析】是你自己之前的观点，仅作为对比参考。你**只需要评审 {target_name} 的分析**，不要在评审中讨论或评判你自己的分析。"""
        review_step2 = f"""**第二步：对比评审——针对 {target_name} 的分析**
针对该分析师的分析，给出："""
        review_close = f"请对 {target_name} 的分析师进行评审，保持理性客观，用中文回答。"
    else:
        review_intro = "请按以下步骤对每位分析师进行评审："
        review_step2 = """**第二步：对比评审**
针对每一位其他分析师，分别给出："""
        review_close = "请对每位分析师分别评审，保持理性客观，用中文回答。"
    
    return f"""{role}
{role_prompt}
你现在进入评审环节。但评审不是被动挑错——你必须先有自己的独立观点，再与他人对比。

【原始问题】
{question}
{memory_section}

{INDEPENDENT_ANALYSIS_INSTRUCTION}

【你的分析】
{my_answer}

{others_section}
{history_section}
{skill_section}
{tool_section}

{review_intro}

**第一步：独立审视**
- 基于你自己的分析框架，列出你认为该问题最关键的 2-3 个判断维度。
- 简要说明你的初步立场（与你之前给出的分析保持一致）。

{review_step2}
1. **认同点**：对方哪些分析深化了你的理解，或比你考虑得更周全
2. **差异点**：你的分析与对方在哪些关键维度上存在分歧？分歧的本质是什么？
3. **质疑点**：对方的推理中哪些环节你认为有问题（数据、逻辑、假设、遗漏）
4. **补充点**：从你的差异化视角，可以补充哪些对方未覆盖的关键信息
5. **对结论的影响**：对方的观点是否改变了你的判断？如果有，为什么？

{review_close}
"""


def build_adversarial_critique_prompt(
    scenario_type: str,
    question: str,
    context: str,
    my_answer: str,
    other_answers: Dict[str, str],
    memory_context: str = "",
    critique_history: str = "",
    skill_context: str = "",
    tool_context: str = "",
) -> str:
    """构建对抗性评审 prompt（严格版，专用于 deepseek）
    
    deepseek 的角色从"分析师"切换为"审查者"，重点是挑错而非认同。
    """
    role = ADVERSARIAL_ROLES.get(scenario_type, ADVERSARIAL_ROLES["question_analysis"])
    
    others_section = "\n\n".join([
        f"【{name} 的分析】\n{ans}"
        for name, ans in other_answers.items()
    ])
    
    memory_section = f"""
【此前辩论摘要】
{memory_context}
""" if memory_context.strip() else ""
    
    history_section = f"""
【历史评审记录】
{critique_history}
""" if critique_history.strip() else ""
    
    skill_section = f"""
{skill_context}
""" if skill_context.strip() else ""

    tool_section = f"""
{tool_context}
""" if tool_context.strip() else ""
    
    # 根据目标数量动态调整措辞
    target_count = len(other_answers)
    if target_count == 1:
        target_name = list(other_answers.keys())[0]
        self_review_note = f"""
> 重要提示：以上【你的分析】是你自己之前的观点，仅作为对比参考。你**只需要严格评审 {target_name} 的分析**，不要在评审中讨论或评判你自己的分析。
"""
        review_close = f"现在，请对 **{target_name}** 的分析师进行严格评审。"
    else:
        self_review_note = ""
        review_close = "现在，请对每一位其他分析师进行严格评审。"
    
    return f"""{role}

原始问题：{question}

{memory_section}

{INDEPENDENT_ANALYSIS_INSTRUCTION}

【你的分析】
{my_answer}

{others_section}
{history_section}
{skill_section}
{tool_section}

{self_review_note}
{ADVERSARIAL_CHECKLIST}

{review_close}

**评审原则**：
- 你不是在"批改作业"，你是在用"我的分析"对比"你的分析"。
- 如果对方在某个点上分析得比你深，直接承认，不要硬挑错。
- 如果你发现对方的分析框架有盲区（而你的独立分析覆盖了这些盲区），明确指出。
- 刻意寻找差异化角度：如果对方从 A 角度切入，你从 B/C 角度检验其结论是否稳健。

评审格式（对每位分析师，先独立审视，再逐条评审）：

**一、独立审视（你自己的分析框架）**
- 你认为该问题最关键的 2-3 个判断维度是什么？
- 你的初步立场是什么？

**二、对比评审**
1. **对方比你强的地方**：对方在哪些维度上分析得比你深入？你吸收了什么？
2. **差异与分歧**：你的分析与对方在哪些关键点上存在分歧？分歧的本质是什么？
3. **事实核查**：指出数据错误、未经证实的断言
4. **逻辑问题**：指出推理跳跃、隐含假设、因果混淆
5. **跑题/回避**：指出是否偏离问题核心、是否回避关键问题
6. **漏答清单**：列出对方未回应的问题（如果有）
7. **你的判断**：对方的结论有多大可信度？如果与你的结论冲突，你更相信谁？为什么？

请用中文回答。直接、尖锐、但基于独立分析，不是为反对而反对。
"""


# ==================== 修订节点 Prompt ====================

def build_revise_prompt(
    scenario_type: str,
    question: str,
    context: str,
    my_previous_answer: str,
    critiques_on_me: Dict[str, str],  # 其他模型对我的评审
    memory_context: str = "",
    critique_history: str = "",
    skill_context: str = "",
    tool_context: str = "",
    my_role: str = "",
) -> str:
    """构建修订阶段的 prompt"""
    
    role = SCENARIO_ROLES.get(scenario_type, SCENARIO_ROLES["question_analysis"])
    
    # 注入修订者角色定义（如果提供了）
    role_prompt = ""
    if my_role:
        role_obj = get_role(my_role)
        role_prompt = f"""
【你的角色定位】
{role_obj.generate_prompt}
---
"""
    
    critiques_section = "\n\n".join([
        f"【{name} 对你的评审】\n{crit}"
        for name, crit in critiques_on_me.items()
    ])
    
    memory_section = f"""
【此前辩论摘要】
{memory_context}
""" if memory_context.strip() else ""
    
    history_section = f"""
【历史评审演进】
{critique_history}

> 提示：以上是其他模型在以往轮次中对你的评审。请注意：
> - 如果历史评审中的问题你已修正，请在修订说明中明确回应
> - 如果历史评审指出的问题反复出现，请重点处理
> - 关注评审意见的聚焦点变化（从表面问题到深层逻辑）
""" if critique_history.strip() else ""
    
    skill_section = f"""
{skill_context}
""" if skill_context.strip() else ""

    tool_section = f"""
{tool_context}
""" if tool_context.strip() else ""
    
    return f"""{role}
{role_prompt}
请根据收到的评审意见，修订你的分析。

【原始问题】
{question}
{memory_section}

【你之前的分析】
{my_previous_answer}
{history_section}
{skill_section}
{tool_section}

【收到的评审意见】
{critiques_section}

修订要求：
1. 先展示你的思考推理过程（放在 <thinking> 标签内）
2. 保留你原分析中仍然正确的部分
3. 修正被合理指出的错误或偏差
4. 整合有价值的补充信息
5. 如果评审意见不合理，说明你的理由
6. 给出更新后的置信度（0%~100%）

请用中文回答，格式如下：

<thinking>
（你的完整推理思考过程）
</thinking>

**修订说明**：
（哪些点被修改、为什么）

**更新后的分析**：
（完整修订后的分析）

**更新后的结论**：
（明确结论）

**更新后的置信度**：X%
（说明依据）
"""


# ==================== Brainstorm Prompt ====================

def build_brainstorm_prompt(
    question: str,
    context: str,
    debater_role: str = "",
) -> str:
    """构建头脑风暴阶段的 prompt
    
    在正式分析前，让模型先拆解问题、识别信息缺口、向用户提问。
    """
    from .roles import get_role
    
    role_prompt = ""
    if debater_role:
        role_obj = get_role(debater_role)
        role_prompt = f"【你的角色定位】\n{role_obj.generate_prompt}\n---\n"
    
    context_section = f"""
【已知背景资料】
{context}
""" if context.strip() else ""
    
    return f"""{role_prompt}你是一位资深金融分析师。在正式给出分析结论之前，你需要先对问题进行结构化拆解，确保分析覆盖所有关键维度，并识别出还缺少哪些必要信息。

【用户提出的问题】
{question}
{context_section}

请进行头脑风暴式分析：
1. **问题定性**：这个问题的本质是什么？属于哪种分析类型？
2. **关键维度**：涉及哪些必须考虑的维度？（如财务、行业、竞争、政策、风险等）
3. **信息缺口**：基于现有信息，还缺少哪些关键数据或背景？
4. **建议追问**：如果用户能提供更多信息，你会追问哪 2-3 个最关键的问题？

请用中文回答，格式如下：

<thinking>
（你的完整思考推理过程）
</thinking>

**问题定性**：
（一句话概括问题本质）

**关键维度**：
1. ...
2. ...
3. ...

**信息缺口**：
1. ...
2. ...

**建议追问**：
1. ...
2. ...
3. ...
"""


# ==================== 自动裁判节点 Prompt ====================

def build_auto_judge_prompt(
    question: str,
    context: str,
    all_answers: Dict[str, str],
    all_confidences: Dict[str, float],
    all_critiques: Dict[str, Dict[str, str]],
    round_num: int,
    max_rounds: int = 5,
) -> str:
    """构建自动裁判判断 prompt
    
    让聚合器模型判断当前辩论状态，决定下一步：
    - continue: 继续下一轮辩论
    - stop: 结论足够清晰，可以结束
    - need_info: 缺少关键信息，需要用户补充
    """
    
    answers_section = "\n\n".join([
        f"【{name}】(置信度: {all_confidences.get(name, 0)*100:.0f}%)\n{ans}"
        for name, ans in all_answers.items()
    ])
    
    critiques_section = ""
    if all_critiques:
        lines = []
        for critiquer, targets in all_critiques.items():
            for target, text in targets.items():
                lines.append(f"- {critiquer} → {target}: {text[:200]}...")
        critiques_section = "\n".join(lines)
    
    return f"""你是一位资深投资委员会主席，负责判断多轮辩论是否应该继续、结束，还是需要补充信息。

【原始问题】
{question}

【已知背景】
{context}

【当前轮次】
第 {round_num} 轮 / 最多 {max_rounds} 轮

【各方当前观点】
{answers_section}

【最近一轮评审摘要】
{critiques_section or "（无）"}

请作为裁判，客观评估当前状态，输出以下 JSON 格式（不要添加任何其他文字，只输出 JSON）：

{{
  "action": "continue|stop|need_info",
  "reason": "详细说明你的判断依据",
  "info_needed": "如果 action 是 need_info，具体说明需要什么信息；否则为空字符串"
}}

判断标准：

**continue**（继续辩论）——满足以下任一条件：
- 各方结论存在明显分歧，需要进一步讨论
- 置信度普遍低于 80%
- 审查者指出了尚未被回应的重大问题
- 有重要的分析维度尚未覆盖

**stop**（结束辩论）——需同时满足：
- 各方核心结论基本一致（即使细节有分歧）
- 各模型置信度均高于 80%
- 审查者没有提出新的重大异议
- 已经覆盖了问题的关键维度

**need_info**（需要补充信息）——满足：
- 存在一个或多个关键事实/数据缺失
- 缺少这些信息的条件下，任何结论都是不可靠的
- 明确说明需要什么具体信息

请只输出 JSON，不要输出任何其他内容。"""


# ==================== 聚合节点 Prompt ====================

def build_aggregate_prompt(
    question: str,
    context: str,
    all_answers: Dict[str, str],
    all_confidences: Dict[str, float],
    all_rounds: int,
) -> str:
    """构建最终聚合输出的 prompt"""
    
    history_section = "\n\n".join([
        f"【{name}】(最终置信度: {all_confidences.get(name, 0)*100:.0f}%)\n{ans}"
        for name, ans in all_answers.items()
    ])
    
    return f"""你是一位资深投资委员会主席。多轮讨论已结束，请你综合各方观点，给出最终的投资级结论。

【原始问题】
{question}

【讨论背景】
{context}

【各方最终观点】
{history_section}

【讨论轮次】
{all_rounds} 轮

请输出：
1. **最终结论**（明确、可执行的判断）
2. **结论依据**（综合各方观点的核心理由）
3. **关键分歧点**（即使最终有结论，也请记录讨论中各方的主要分歧）
4. **风险提示**（需要特别关注的剩余不确定性）
5. **建议后续跟踪点**（什么信息变化会导致结论改变）

请用专业、简洁的中文撰写，适合写入投资备忘录。
"""
