# 多模型辩论系统 (Debater) — 设计与实现文档

> 版本: 2026-05-19
> 状态: 生产中

---

## 1. 系统概述

### 1.1 产品定位

**Debater** 是一个面向金融/投研领域的多模型辩论系统。通过让多个大语言模型（LLM）扮演不同角色，对同一问题进行多轮辩论、互相评审、逐步修订，最终输出经过充分审视的综合性结论。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 多模型并行 | 同时调用 Kimi、DeepSeek 等多个模型，对比不同视角的分析 |
| 角色-模型解耦 | Bull（支持者）/ Bear（质疑者）角色与具体模型解耦，同一模型可扮演两个角色 |
| 流式实时展示 | Token 级流式输出，用户实时看到每个角色的思考过程 |
| 工具调用 (Tool Calling) | 模型可自主调用搜索工具获取实时数据，验证关键论点；支持系统回退搜索 |
| 多轮辩论 | generate → critique → revise → judge 的严格串行交替闭环流程 |
| Memory 系统 | 三层记忆架构（Working/Stage/Obsidian），支持长程辩论 |
| Skill 注入 | 按场景自动加载专业 Skill（如财务分析、竞争分析框架） |
| 单角色重试 | 历史记录中可对单个角色重新生成输出 |
| 完成状态指示 | 每轮每个角色显示 ✅ success / ❌ error / ⏳ pending 状态 |

### 1.3 技术栈

- **Python 3.9+**
- **Streamlit 1.50** — 交互式 Web UI
- **LangGraph** — 辩论工作流编排（底层状态机，参考实现）
- **Anthropic SDK** — 统一 API 接口（兼容 Kimi、DeepSeek 等）
- **Tavily + DuckDuckGo** — 网络搜索工具链（默认使用 DDG）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI 层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Brainstorm │  │   Generate  │  │   Critique  │  │  Aggregate │  │
│  │   (Phase 0) │  │  (Phase 1+) │  │  (Phase 1+) │  │   (Final)  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DebateSession (手动控制流)                       │
│                                                                      │
│   状态以 role_id（"bull"/"bear"）为 key，而非 model_id               │
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │ brainstorm_  │───▶│ generate_    │───▶│ critique_    │───▶ ... │
│   │ stream_para. │    │ stream_para. │    │ stream_para. │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              流式事件队列 (token / tool_executing /          │   │
│   │              model_done / error / done)                     │   │
│   │              所有事件携带 role_id + model_id                │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LLMClient (统一客户端)                        │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────┐   │
│   │ call()      │  │ call_stream()│  │ call_parallel()           │   │
│   │ 同步调用    │  │ 流式调用     │  │ 并行同步调用(call_id)      │   │
│   └─────────────┘  └─────────────┘  └───────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  模型适配层：Kimi (api.kimi.com)  /  DeepSeek (api.deepseek) │   │
│   │  Anthropic SDK 兼容接口，支持 text / thinking / reasoning    │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Skill    │   │   Tool    │   │  Memory   │
            │ Registry  │   │ Registry  │   │  Manager  │
            └───────────┘   └───────────┘   └───────────┘
```

---

## 3. 核心模块详解

### 3.1 配置系统 (`debater/config.py`)

#### 3.1.1 设计目标

- 支持多配置文件动态切换
- 模型配置集中管理（API Key、Base URL、Temperature 等）
- 场景配置与辩论参数独立

#### 3.1.2 核心类

```python
class ModelConfig:
    id: str                # 模型标识（如 "kimi_k2.6"）
    name: str              # 显示名称
    provider: str          # 提供商（anthropic 兼容）
    model_id: str          # 实际模型 ID（如 "kimi-for-coding"）
    api_key: str           # API Key（优先）或环境变量名
    base_url: str          # 自定义 Base URL
    temperature: float     # 采样温度
    max_tokens: int        # 最大输出 token
    role: str              # proposer | aggregator
    extra_body: dict       # 模型特定参数（如 reasoning_effort）
    critique_style: str    # standard | adversarial

class AppConfig:
    models: Dict[str, ModelConfig]
    debate: DebateConfig    # consensus_threshold, min_confidence
    scenarios: dict         # 场景定义
```

#### 3.1.3 配置示例 (`config.yaml`)

```yaml
models:
  kimi_k2.6:
    name: "kimi-coding-k2.6"
    provider: "anthropic"
    model_id: "kimi-for-coding"
    base_url: "https://api.kimi.com/coding"
    max_tokens: 60000
    role: "proposer"
    extra_body:
      reasoning_effort: high

debate:
  consensus_threshold: 0.85
  min_confidence: 0.70
```

---

### 3.2 LLM 客户端 (`debater/llm_client.py`)

#### 3.2.1 设计目标

- **统一接口**：所有模型通过 Anthropic SDK 兼容方式接入
- **流式支持**：Token 级实时输出，支持 reasoning/thinking 内容提取
- **超时保护**：防止模型陷入 reasoning loop 导致前端无限卡住
- **异常隔离**：单模型失败不掐断其他模型
- **流式失败 fallback**：流式调用失败后自动 fallback 到同步调用

#### 3.2.2 核心类

```python
class LLMClient:
    def call(self, model_id, system_prompt, user_prompt) -> LLMResponse
    def call_stream(self, model_id, ...) -> Generator[(token, accumulated), None, LLMResponse]
    def call_parallel(self, calls: List[dict]) -> Dict[str, LLMResponse]
```

#### 3.2.3 流式超时机制（关键设计）

```
┌─────────────────────────────────────────────────────────────┐
│  call_stream() 内部架构                                      │
│                                                              │
│   ┌──────────────┐         ┌─────────────────────────────┐  │
│   │ _producer()  │────────▶│  result_queue (Queue)       │  │
│   │ daemon thread│  token  │  maxsize=0 (无界)           │  │
│   │              │         │                             │  │
│   │ messages.    │         │  ┌───────────────────────┐  │  │
│   │ stream()     │         │  │ Consumer (主线程)      │  │  │
│   │              │         │  │                       │  │  │
│   │ yield token  │         │  │ while True:           │  │  │
│   │ → queue.put  │         │  │   event, payload =    │  │  │
│   │              │         │  │     queue.get(        │  │  │
│   └──────────────┘         │  │       timeout=300s    │  │  │
│                            │  │     )                 │  │  │
│                            │  │   if timeout:         │  │  │
│                            │  │     raise RuntimeError│  │  │
│                            │  └───────────────────────┘  │  │
│                            └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策：**

1. **Daemon 线程**：生产者使用 `threading.Thread(daemon=True)`，避免 `ThreadPoolExecutor.shutdown(wait=True)` 在流未关闭时阻塞主线程
2. **300 秒 Token 间隔超时**：消费者通过 `queue.get(timeout=300)` 检测，300 秒内无新 token 则判定模型卡住（DeepSeek reasoning 可能持续数分钟）
3. **异常透传**：生产者捕获所有异常，通过 `("error", str(e), traceback)` 事件发送给消费者
4. **消费者支持 2/3/N 元组通用解包**（2026-05-11 修复）： producer 可能发送 `("token", text)`、`("done", None)` 或 `("error", msg, traceback)`，consumer 统一处理
5. **流式失败 fallback**：流式调用失败后（连接错误最多重试 1 次），fallback 到同步调用，一次性 yield 完整文本
6. **`tool_use` content block 处理**（2026-05-11 新增）：捕获 `content_block_start` 事件中 `type=tool_use` 的 content block，将其序列化为 XML 格式的 `<tool_call>` 标签输出

#### 3.2.4 Token 提取逻辑

```python
# 支持多模型输出格式（优先级：text > thinking > reasoning_content）
token = ""
if hasattr(event, "delta") and event.delta:
    delta = event.delta
    if hasattr(delta, "text") and delta.text:
        token = delta.text                    # 标准文本 token
    elif hasattr(delta, "thinking") and delta.thinking:
        token = delta.thinking                # Kimi thinking
    elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
        token = delta.reasoning_content       # DeepSeek reasoning
```

---

### 3.2.5 公共引擎 (`debater/engine.py`)

#### 3.2.5.1 设计目标

- **消除代码重复**：提取 `session.py` 和 `nodes.py` 共用的纯工具函数
- **纯函数设计**：无状态、无 side effect，便于单元测试
- **统一解析逻辑**：置信度提取、thinking/summary 解析、Judge 输出解析

#### 3.2.5.2 核心函数

```python
# 从模型输出中提取置信度（支持数据缺失强制降权）
def extract_confidence(text: str) -> float

# 提取 <thinking> 标签并清理 tool_call XML
def extract_thinking(text: str) -> Tuple[str, str]

# 提取 <summary> 总结块
def extract_summary(text: str) -> str

# 清理文本中的 tool_call / tool_result XML 标签
def clean_tool_tags(text: str) -> str

# 解析 Judge 模型输出（4 层 fallback：json块 → 裸json → 关键词提取 → 默认）
def parse_judge_output(text: str) -> dict
```

#### 3.2.5.3 Judge 解析的 4 层 Fallback

```
模型输出文本
    │
    ▼
┌─────────────────────┐
│ 1. ```json 代码块   │──成功→ json.loads
│    (最常见)         │
└──────────┬──────────┘
           │ 失败
           ▼
┌─────────────────────┐
│ 2. 裸 JSON { ... }  │──成功→ json.loads
│    (模型省略标记)   │
└──────────┬──────────┘
           │ 失败
           ▼
┌─────────────────────┐
│ 3. 关键词提取       │──在文本中搜索 "stop"/"need_info"
│    (模型用自然语言) │
└──────────┬──────────┘
           │ 未匹配
           ▼
┌─────────────────────┐
│ 4. 默认 fallback    │──action="continue"
│    (安全保守)       │
└─────────────────────┘
```

---

### 3.3 辩论会话 (`debater/session.py`)

#### 3.3.1 设计目标

- **手动控制流**：每个阶段结束后暂停，等待用户决策（继续/跳过/结束）
- **流式并行**：generate / brainstorm 阶段多角色同时生成，Token 实时展示
- **串行交替**：critique → revise → judge 严格串行，角色按轮次交替（奇数轮 bear→bull，偶数轮 bull→bear）
- **角色-模型解耦**：状态以 role_id 为 key，同一模型可扮演两个角色
- **状态持久**：支持 Streamlit 的 `st.session_state` 模式
- **工具调用双轮+回退**：第一轮生成 → 检测 tool call / 回退搜索 → 执行 → 第二轮注入结果

#### 3.3.2 角色-模型解耦（核心架构）

```python
# DebateSession 核心状态（以 role_id 为 key）
self.role_models: Dict[str, str]       # role_id → model_id, 如 {"bull": "kimi_k2.6", "bear": "deepseek_v4"}
self.answers: Dict[str, str]           # role_id → 正式分析文本
self.thinkings: Dict[str, str]         # role_id → <thinking> 内容
self.confidences: Dict[str, float]     # role_id → 置信度
self.summaries: Dict[str, str]         # role_id → <summary> 内容
self.previous_answers: Dict[str, str]  # role_id → 完整原始输出
self.critiques: Dict[str, Dict[str, str]]  # role_id → {target_role_id: critique_text}
```

**同一模型双角色**：当 bull 和 bear 都选同一模型时，两个独立线程并发调用，结果分别存入 `answers["bull"]` / `answers["bear"]`。LLM 调用通过 `call_id=role_id` 区分。

#### 3.3.3 状态机

```
                    ┌─────────────┐
                    │    INIT     │
                    └──────┬──────┘
                           │ 用户点击"开始头脑风暴"
                           ▼
                    ┌─────────────┐
         ┌─────────│  GENERATING │◄────────┐
         │         │ (Brainstorm)│         │
         │         └──────┬──────┘         │
         │                │                │
         │                ▼                │
         │    ┌───────────────────────┐    │
         │    │ PAUSE_AFTER_GENERATE  │    │
         │    │   (用户确认/补充事实)  │    │
         │    └───────────┬───────────┘    │
         │                │ 用户点击"开始辩论"
         │                ▼                │
         │    ┌───────────────────────┐    │
         │    │    GENERATING         │────┘
         │    │   (Formal Debate)     │
         │    └───────────┬───────────┘
         │                │
         │                ▼
         │    ┌───────────────────────┐
         │    │ PAUSE_AFTER_GENERATE  │
         │    └───────────┬───────────┘
         │                │ 用户点击"评审"
         │                ▼
         │    ┌───────────────────────┐
         │    │     CRITIQUING        │
         │    │   (交替方向, 串行)    │
         │    └───────────┬───────────┘
         │                │
         │                ▼
         │    ┌───────────────────────┐
         │    │ PAUSE_AFTER_CRITIQUE  │
         │    └───────────┬───────────┘
         │                │ 用户点击"修订"
         │                ▼
         │    ┌───────────────────────┐
         │    │      REVISING         │
         │    │    (被评审方, 串行)   │
         │    └───────────┬───────────┘
         │                │
         │                ▼
         │    ┌───────────────────────┐
         │    │ PAUSE_AFTER_REVISE    │────┐
         │    │  (未收敛 → 进入下一轮  │    │
         │    │   critique, round+=1) │    │
         │    └───────────────────────┘    │
         │                                 │
         └─────────────────────────────────┘
                          │ 用户点击"聚合"
                          ▼
              ┌───────────────────────┐
              │     AGGREGATING       │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │        DONE           │
              └───────────────────────┘
```

#### 3.3.4 流式事件架构

所有流式方法采用统一的事件驱动架构，**所有事件均携带 `role_id` 和 `model_id`**。通过 `step_stream()` 统一入口调用时，事件还携带 `step_type` 字段（"generate"/"critique"/"revise"），供 UI 层自动推断当前步骤类型：

```python
# 事件类型
event = {
    "type": "token",           # token 事件
    "model_id": "kimi_k2.6",
    "role_id": "bull",         # ← 新增
    "token": "单个 token",
    "accumulated": "累积文本",
}

event = {
    "type": "tool_executing",  # 工具执行事件
    "model_id": "deepseek_v4",
    "role_id": "bear",
    "message": "✅ 工具执行完成",
    "executions": [...],
    "has_error": False,
    "fallback_notices": [],    # 回退来源标注
}

event = {
    "type": "model_done",      # 模型完成事件
    "model_id": "kimi_k2.6",
    "role_id": "bull",
    "full_text": "完整输出",
}

event = {
    "type": "error",           # 错误事件
    "model_id": "deepseek_v4",
    "role_id": "bear",
    "error": "流式调用失败: ...",
}

event = {
    "type": "partial_error",   # 部分错误（汇总）
    "errors": ["model1: ...", "model2: ..."],
}

event = {
    "type": "done",            # 全部完成
    "ui_state": {...},
}

# step_stream() 注入的元数据字段
event = {
    "type": "token",
    "step_type": "critique",   # ← step_stream() 自动注入：generate/critique/revise
    "model_id": "deepseek_v4",
    "role_id": "bear",
    ...
}
```

#### 3.3.5 统一单步入口 `step_stream()` / `step()`

为避免 UI 层和底层各维护一套状态机，引入统一的单步执行入口：

```python
def step_stream(self) -> Generator[Dict, None, None]:
    """根据当前 stage 自动执行下一步"""
    if self.stage == Stage.INIT:
        yield from self.generate_stream_parallel()          # step_type="generate"
    elif self.stage == Stage.PAUSE_AFTER_GENERATE:
        yield from self.critique_stream_single(*self._get_critique_pair())  # step_type="critique"
    elif self.stage == Stage.PAUSE_AFTER_CRITIQUE:
        yield from self.revise_stream_single(self._get_reviser())        # step_type="revise"
    elif self.stage == Stage.PAUSE_AFTER_REVISE:
        self.current_round += 1
        yield from self.critique_stream_single(*self._get_critique_pair())  # step_type="critique"

def step(self) -> Dict[str, Any]:
    """同步版：消费 step_stream() 后返回 ui_state"""
    for _ in self.step_stream():
        pass
    return self._build_ui_state()
```

**设计原则**：UI 层只调用 `step_stream()` / `step()`，不自己判断该执行哪个方法。`DebateSession` 是唯一的状态机。

#### 3.3.6 生产者-消费者模型（基于角色）

```python
# 每个角色一个生产者线程（而非每个模型）
threads = []
for role_id, model_id in self.role_models.items():
    t = threading.Thread(target=stream_single, args=(role_id, model_id), daemon=True)
    t.start()
    threads.append(t)
thread_infos = list(self.role_models.items())

# 消费者主循环：按角色数等待
done_count = 0
while done_count < len(thread_infos):
    try:
        event = event_queue.get(timeout=300)
    except queue.Empty:
        # 超时检测：检查哪些线程仍然存活
        break
    
    yield event
    if event["type"] in ("model_done", "error"):
        done_count += 1
```

**关键设计**：
- `done_count < len(thread_infos)` 确保同一模型双角色时正确等待两个线程
- 超时后检查线程存活状态，区分"API 卡住"和"线程崩溃"

#### 3.3.6 工具调用双轮+回退流程

```
第一轮生成
    │
    ▼
模型输出包含 <tool_call name="web_search">...
    │
    ▼
has_tool_calls() → True
    │
    ▼
发送 "🔧 检测到 X 个工具调用" 事件到 UI
    │
    ▼
extract_and_execute() 并行执行（30s 超时，最多10个）
    │
    ▼
发送执行结果事件到 UI（含 duration、status、sources）
    │
    ▼
构建 followup prompt → 第二轮 call_stream()
    │
    ▼
第二轮输出为空？→ 自动回退到第一轮输出（2026-05-11 新增）
    │
    ▼
第二轮 Kimi 返回 "high risk"？→ 使用简化 prompt（不含搜索结果）重试

┌────────────────────────────────────────────────────────────┐
│  回退搜索路径（模型未输出 XML 但表达了搜索意图）              │
│                                                              │
│  关键词检测："搜索"/"查询"/"查一下"/"获取数据"/"用工具"/      │
│  "调用工具"/"事实核查"                                        │
│      │                                                       │
│      ▼                                                       │
│  自动构造：`<tool_call name="web_search">`                   │
│            `<query>{question[:100]}</query>`                 │
│            `</tool_call>`                                    │
│      │                                                       │
│      ▼                                                       │
│  执行搜索 → 进入第二轮                                        │
│  UI 显示："🔄 模型未输出工具标签，系统自动执行回退搜索"        │
└────────────────────────────────────────────────────────────┘
```

**工具失败处理策略：**

- **全部失败**：在 followup prompt 中注入 `【🚨 工具调用全部失败——关键数据缺失】`，强制置信度低于 50%
- **部分回退**（如 Tavily → DDG）：在 followup prompt 中标注回退来源
- **全部成功**：正常注入结果，模型基于新数据完善分析
- **回退搜索**：模型未按格式输出 XML 时，系统自动搜索补充数据（查询词使用 `html.escape()` 转义，防止 XML 被破坏）
- **第二轮空输出防护**：若第二轮输出为空或仅空白字符，自动回退到第一轮输出
- **Kimi "high risk" 拒绝处理**：第二轮若因内容安全策略被拒绝，捕获 `RuntimeError` 并检测 `"high risk"`，使用不含搜索结果的简化 prompt 自动重试

---

### 3.4 提示词系统 (`debater/prompts.py`)

#### 3.4.1 四层提示词架构（已调整）

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: 系统级工具/技能上下文 (system prompt 尾部动态注入)   │
│  - Tool context（可用工具说明、XML 格式示例）                  │
│  - 注：2025-05-11 从 user prompt 迁移到 system prompt         │
│        以提高模型（尤其 DeepSeek）对工具调用格式的遵循度        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 场景角色 + 辩论角色 (system prompt)                 │
│  - SCENARIO_ROLES: 资深金融分析师/财务分析师/行业研究员等      │
│  - DebateRole.generate_prompt: Bull/Bear 角色定位              │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 任务 Prompt (user prompt)                           │
│  - build_brainstorm_prompt(): 头脑风暴（注入 debater_role）    │
│  - build_generate_prompt(): 初始生成（注入 debater_role）      │
│  - build_critique_prompt(): 互相评审（注入 critiquer_role）    │
│  - build_adversarial_critique_prompt(): 对抗性评审             │
│  - build_revise_prompt(): 修订完善（注入 my_role）             │
│  - build_aggregate_prompt(): 最终聚合                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 输出格式约束                                        │
│  - <thinking>...</thinking> 思考过程                         │
│  - <summary>...</summary> 末尾总结（含置信度）                │
│  - <tool_call>...</tool_call> 工具调用（系统级注入说明）      │
└─────────────────────────────────────────────────────────────┘
```

#### 3.4.2 角色化 Prompt 参数

所有 prompt 构建函数支持角色注入：

```python
def build_generate_prompt(..., debater_role: str = "") -> str
    # 注入 ROLES[debater_role].generate_prompt

def build_critique_prompt(..., critiquer_role: str = "") -> str
    # 注入 ROLES[critiquer_role].critique_prompt

def build_revise_prompt(..., my_role: str = "") -> str
    # 注入 ROLES[my_role].generate_prompt

def build_brainstorm_prompt(..., debater_role: str = "") -> str
    # 注入 ROLES[debater_role].generate_prompt
```

#### 3.4.3 工具使用策略 Prompt

工具说明现在通过 `ToolRegistry.get_prompt_context()` 生成，格式：

```
你有以下工具可用：

<tool name="web_search">
  <description>网络搜索工具</description>
  <parameters>
    <parameter name="query" type="string" required="true">搜索关键词</parameter>
  </parameters>
</tool>

如需使用工具，请按以下格式输出（会被自动执行并返回结果）：
<tool_call name="tool_name">
<参数名>参数值</参数名>
</tool_call>

例如搜索工具：
<tool_call name="web_search">
<query>保利物业 2025 毛利率</query>
</tool_call>

⚠️ 重要：决定调用工具时，直接输出上面的 XML 格式，不要写自然语言描述。
```

---

### 3.5 角色系统 (`debater/roles.py`)

#### 3.5.1 设计目标

将"模型"与"角色"解耦：
- **角色**是业务逻辑（Bull 支持者 / Bear 质疑者），定义生成和评审时的行为准则
- **模型**只是执行角色的演员（kimi、deepseek、claude 等均可）
- 同一模型可以被分配给两个不同角色，实现并发辩论

#### 3.5.2 核心类

```python
@dataclass
class DebateRole:
    id: str                   # "bull" | "bear"
    name: str                 # "支持者" | "质疑者"
    emoji: str                # "🐂" | "🐻"
    description: str
    generate_prompt: str      # 生成阶段的角色系统提示
    critique_prompt: str      # 评审阶段的角色指导

ROLES = {
    "bull": DebateRole(...),   # 乐观分析师：发现价值、强化论证
    "bear": DebateRole(...),   # 严格审查者：找出漏洞、压力测试
}
```

---

### 3.6 工具系统 (`debater/tools/`)

#### 3.6.1 架构概览

```
debater/tools/
├── base.py           # Tool / ToolResult / ToolExecutor 基类
├── registry.py       # ToolRegistry（注册表）
├── search.py         # TavilySearchTool（主搜索）
├── ddg_search.py     # DDGSearchTool（免费回退，当前默认）
├── fetch.py          # WebFetchTool（网页抓取）
├── github_search.py  # GitHubSearchTool（代码搜索）
└── filesystem.py     # FileReadTool / FileGlobTool / CodeSearchTool
```

#### 3.6.2 Prompt-based Tool Calling

采用 XML 标签方式，模型在输出中嵌入 `<tool_call>` 标签：

```xml
<tool_call name="web_search">
<query>保利物业 2025 毛利率</query>
</tool_call>
```

**解析流程：**

1. `ToolExecutor.has_tool_calls(text)` — 正则检测是否存在 `<tool_call>`
2. `ToolExecutor.extract_and_execute(text)` — 提取所有 tool call，并行执行
   - **数量上限**：单轮最多执行 10 个 tool call（防止模型异常输出数百个导致卡死）
3. `ToolExecutor.build_tool_result_message(results)` — 构建 `<tool_result>` 注入 prompt

#### 3.6.3 搜索工具链（Tavily → DuckDuckGo）

```
用户查询
    │
    ▼
┌─────────────────────┐
│  TavilySearchTool   │
│  - 15s HTTP 超时     │
│  - max_results=5     │
│  - search_depth=advanced
└──────────┬──────────┘
           │
      成功 │              失败
           ▼                 ▼
    ┌──────────┐      ┌─────────────────┐
    │ 返回结果  │      │ DDGSearchTool    │
    │ + sources │      │ - 15s 执行超时   │
    └──────────┘      │ - 自动语言检测   │
                      │   zh→cn-zh, en→us-en
                      └────────┬────────┘
                               │
                          成功 │         失败
                               ▼            ▼
                        ┌──────────┐  ┌─────────────────────┐
                        │ 返回结果  │  │ 返回合并错误信息     │
                        │ [回退标注]│  │ Tavily + DDG 都失败 │
                        └──────────┘  └─────────────────────┘
```

**注**：当前默认使用 DDG（DuckDuckGo），Tavily 因配额问题已停用。

#### 3.6.4 并行执行与超时

```python
# 每个 tool call 一个线程，整体 30s 超时
pool = ThreadPoolExecutor()
futures = {pool.submit(_run_one, m): m for m in matches[:MAX_TOOL_CALLS]}  # 最多10个

for future in futures:
    try:
        results.append(future.result(timeout=30))
    except FutureTimeoutError:
        # 记录超时结果
    except Exception:
        # 记录异常结果
finally:
    pool.shutdown(wait=False)  # 避免死锁
```

**关键设计**：
- `shutdown(wait=False)` 防止嵌套 ThreadPoolExecutor 死锁
- `MAX_TOOL_CALLS = 10` 防止模型异常输出大量 `<tool_call>` 导致系统卡死

---

### 3.7 技能系统 (`debater/skills/`)

#### 3.7.1 Skill 加载机制

Skill 采用 Claude Code 标准格式：YAML frontmatter + Markdown body。

```markdown
---
name: 3-statement-model
description: 构建三表财务模型的完整框架
---

# 三表财务模型

## 收入表...
```

**加载路径：**
1. `~/.config/agents/skills/`（用户级）
2. `./skills/`（项目级）

**场景映射：**

| 场景 | Skill |
|------|-------|
| question_analysis | brainstorming |
| financial_interpretation | 3-statement-model |
| event_analysis | competitive-analysis |
| risk_assessment | systematic-debugging |

#### 3.7.2 Prompt 注入

Skill 内容在 prompt 中作为 Layer 2 尾部注入：

```python
【专业 Skill 指导】
{skill.description}

{skill.body[:3000]}
```

---

### 3.8 记忆系统 (`debater/memory.py`)

#### 3.8.1 三层记忆架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Working Memory                                    │
│  - 文件: memories/working_memory.md                         │
│  - 更新频率: 每 5 轮（第 2 轮提前激活）                      │
│  - 内容: 核心事实、活跃观点、关键分歧、待跟踪事项             │
│  - 特点: 持续增量更新，保留辩论连贯性                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Stage Memory                                      │
│  - 文件: memories/stage_memory/stage_{N}.md                 │
│  - 更新频率: 每 10 轮                                       │
│  - 操作: 1) 保存完整快照  2) 压缩 Working Memory 为骨架版    │
│  - 特点: 防止 Working Memory 无限膨胀                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Obsidian                                          │
│  - 文件: memories/obsidian/debate_{id}_{date}.md            │
│  - 触发: 辩论结束时一次性生成                                │
│  - 内容: 结构化投资备忘录，含 [[双向链接]]                    │
│  - 结构: 问题定义 → 核心结论 → 关键实体 → 核心论据 → 分歧 → 风险 → 跟踪 │
└─────────────────────────────────────────────────────────────┘
```

#### 3.8.2 Working Memory 更新流程

```
最近 5 轮数据
    │
    ▼
构建 rounds_text（摘要）
    │
    ▼
调用 Aggregator 模型（LLM）
    │
    ▼
生成更新后的 Working Memory Markdown
    │
    ▼
写入 memories/working_memory.md
```

#### 3.8.3 启用开关

Memory 系统默认**关闭**（`enable_memory=False`），以减少不必要的 LLM 调用。仅在长程辩论（>5 轮）或需要生成 Obsidian 文档时建议开启：

```python
session = DebateSession(
    question="...",
    scenario_type="question_analysis",
    context="",
    enable_memory=True,  # 开启记忆系统
)
```

关闭时：
- `memory_manager` 保持为 `None`
- `_update_memory()` 和 `_maybe_summarize()` 直接返回，不调用 LLM
- 不生成 `working_memory.md` 或 `stage_memory`
- `finish_debate()` 仍可用，但不会生成 Obsidian 文档

---

### 3.9 Streamlit UI (`streamlit_app.py`)

#### 3.9.1 页面结构

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                                                    │
│  ├── 模型选择（kimi / deepseek）→ 分配给 bull / bear         │
│  ├── 配置管理（切换/保存配置）                               │
│  ├── 🔧 工具执行状态（实时，按 role_id）                     │
│  └── 📊 辩论统计                                            │
├─────────────────────────────────────────────────────────────┤
│  Main Content                                               │
│  ├── 历史轮次展示（从上到下排列）                            │
│  │   └── Round N: [🐂 支持者 · model] [🐻 质疑者 · model]   │
│  │       含完成状态标签（✅/❌/⏳）+ 🔄 重新输出按钮          │
│  ├── 当前轮次实时流式输出                                    │
│  │   └── [🐂 Bull] ████████████ (实时 token)                │
│  │   └── [🐻 Bear] ██████░░░░░░ (实时 token)                │
│  ├── 🔧 工具调用记录（输入栏上方，最近 20 条）                │
│  ├── 底部固定事实框（用户可补充已知事实）                     │
│  └── 控制按钮（开始/评审/修订/聚合/自动模式）                │
└─────────────────────────────────────────────────────────────┘
```

#### 3.9.2 状态管理

```python
st.session_state = {
    "session": DebateSession(...),       # 核心会话对象（唯一状态机）
    "history": [...],                     # 历史轮次记录
    "phase": "init",                      # init → brainstorm → debate → done
    "live_texts": {},                     # 当前轮次各角色累积文本（key=role_id）
    "tool_status": {},                    # 侧边栏工具状态（key=role_id）
    "tool_history": [],                   # 工具调用历史记录
    "role_models": {"bull": "kimi_k2.6", "bear": "deepseek_v4"},
    "auto_mode": False,                   # 自动辩论模式开关
    "auto_judge_done": None,              # auto_mode: 本轮 judge 是否已完成
    "auto_pause_reason": None,            # auto_mode: 暂停原因（如 need_info）
    # 注：已移除 auto_next_action，状态流转完全由 session.stage 驱动
}
```

#### 3.9.3 流式渲染 (`run_stream`)

```python
def run_stream(session, stream_gen, phs, tool_phs, round_type=None):
    """通用流式收集器
    
    round_type 为 None 时，自动从事件中的 step_type 字段推断。
    这让 UI 层无需知道当前执行的是 generate/critique/revise。
    """
    # phs / tool_phs 的 key 是 role_id（"bull"/"bear"）
    texts = {rid: "" for rid in phs.keys()}
    role_status = {rid: "pending" for rid in phs.keys()}
    
    inferred_type = round_type
    for event in stream_gen:
        if inferred_type is None and event.get("step_type"):
            inferred_type = event["step_type"]   # ← 自动推断
        
        etype = event["type"]
        rid = event.get("role_id", event["model_id"])
        
        if etype == "token":
            texts[rid] = event["accumulated"]
            phs[rid].markdown(fmt_output(accumulated), ...)
            
        elif etype == "tool_executing":
            st.session_state.tool_status[rid] = {...}
            st.session_state.tool_history.append({...})
            
        elif etype == "model_done":
            role_status[rid] = "success"
            
        elif etype == "error":
            role_status[rid] = "error"
            st.error(f"❌ {rid} 失败: {event['error']}")
    
    effective_type = inferred_type or round_type or "unknown"
    result = {
        "round": session.current_round,       # ← 使用 session.current_round 避免不同步
        "type": effective_type,
        "type_name": TYPE_NAMES.get(effective_type, effective_type),
        "_role_status": role_status,
        ...
    }
    return result
```

#### 3.9.4 手动模式与自动模式

**手动模式**：根据 `session.stage` 显示对应按钮，执行统一调用 `session.step_stream()`：

| stage | 显示按钮 | 执行 |
|-------|----------|------|
| `INIT` | "🚀 开始辩论" | `step_stream()` → generate |
| `PAUSE_AFTER_GENERATE` | "🔍 bear 评审" / "✍️ 我要评判" / "🤖 自动辩论" / "🏁 结束辩论" | `step_stream()` → critique |
| `PAUSE_AFTER_CRITIQUE` | "✅ bull 修订" / "✍️ 我要评判" / "🤖 自动辩论" / "🏁 结束辩论" | `step_stream()` → revise |
| `PAUSE_AFTER_REVISE` | "🔄 继续下一轮" / "🏁 结束辩论" | `step_stream()` → round+=1, critique |

**自动模式**：循环 `step_stream()` + `judge()`，不再维护 `auto_next_action`：

```python
if session.stage == PAUSE_AFTER_REVISE and not auto_judge_done:
    judge_result = session.auto_judge()
    # continue → auto_judge_done = True → rerun
    # stop    → finish() → done
    # need_info → 暂停，等待用户补充
else:
    result = run_stream(session, session.step_stream(), ...)
    history.append(result)
    if session.stage == PAUSE_AFTER_REVISE:
        auto_judge_done = False  # 下一轮需要先 judge
    rerun()
```

#### 3.9.5 单角色重试

历史记录卡片中显示"🔄 重新输出"按钮（仅对 generate / brainstorm / revise 类型显示）：
- 点击后弹出 text_area 让用户补充新信息
- 调用 `session.retry_generate_role_stream(role_id, extra_context, round_type)`
- **round_type 感知**：brainstorm 类型使用 `build_brainstorm_prompt`，generate/revise 使用 `build_generate_prompt`
- 该角色重新走完整的第一轮→工具调用→第二轮流程
- 替换历史记录中对应角色的输出

**历史记录 key 匹配机制（2026-05-14 修复）**：
- 每轮历史记录项记录 `_role_models`（创建时的角色-模型映射）
- retry 更新时，使用历史记录中的 `_role_models` 确定写入 key，使用当前 `role_models` 确定读取 key
- 解决模型切换后旧历史记录项内容"消失"的问题

---

## 4. 数据流

### 4.1 完整辩论数据流

```
用户输入问题 + 上下文
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 0: Brainstorm                                          │
│ - 两个角色并行分析问题                                       │
│ - 识别信息缺口，向用户提问                                   │
│ - 用户可补充"已知事实"                                       │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Generate (Round 1)                                  │
│ - 两个角色并行生成初始分析                                   │
│ - 自动检测 tool call，执行搜索（含回退搜索）                 │
│ - 基于搜索结果生成第二轮完善分析                             │
│ - 输出格式: <thinking> + 正文 + <summary>                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Critique (串行: 交替方向)                           │
│ - 只有 bear 评审 bull 的回答                                 │
│ - Bear: 挑错、找漏洞、事实核查                               │
│ - 不再并行双方互相评审，保证一方基于最新输出反应              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Revise (串行: 被评审方修订)                         │
│ - 只有 bull 根据 bear 的评审意见修订回答                     │
│ - 置信度重新评估                                             │
│ - bear 的答案保持 generate 阶段结果不变                      │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 4: Judge                                               │
│ - 检查是否收敛（置信度 > threshold 或达到最大轮次）           │
│ - 未收敛: 回到 Critique (round += 1)，不重新 generate        │
│ - 已收敛: 进入 Aggregate                                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 5: Aggregate                                           │
│ - 调用 Aggregator 模型综合所有轮次                           │
│ - 输出最终结论 + 推理过程 + 分歧点                            │
│ - 生成 Obsidian 投资备忘录                                   │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 工具调用数据流

```
模型输出文本
    │
    ▼
┌─────────────────┐
│ has_tool_calls  │───Yes──▶ 正常工具调用流程
│ (正则检测)      │
└────────┬────────┘
         │ No
         ▼
┌──────────────────────────┐
│ 检测自然语言搜索意图      │
│ (关键词: 搜索/查询/查一下) │───Yes──▶ 系统回退搜索
└────────┬─────────────────┘
         │ No
         ▼
    直接结束（无工具调用）

正常工具调用流程:
┌─────────────────┐
│ 发送 "检测到工具"│───▶ UI 显示 "🔧 检测到 X 个工具调用..."
│ 事件到队列      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ extract_and_    │
│ execute()       │───并行执行所有 tool call（ThreadPoolExecutor，最多10个）
│ (30s 超时)      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 发送执行结果    │───▶ UI 显示工具卡片（耗时、状态、来源）
│ 事件到队列      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 构建 followup   │
│ prompt          │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 第二轮 call_    │───若仍有 tool call → 第三轮强提示
│ stream()        │───若第三轮仍有 → 报错，不沉默
└─────────────────┘
```

---

## 5. 关键设计决策

### 5.1 为什么采用手动控制流而非全自动 LangGraph？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **LangGraph 全自动** | 代码简洁，自动流转 | 用户无法介入，无法补充事实，无法看到中间过程 |
| **手动控制流 (当前)** | 每阶段暂停，用户可补充事实、调整方向 | 代码复杂度高，状态管理繁琐 |

**决策**：采用手动控制流作为 UI 层主逻辑，LangGraph 仅作为底层参考实现（`nodes.py` + `graph.py`）。

### 5.2 为什么采用 Prompt-based Tool Calling 而非 Function Calling？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Function Calling** | 结构化参数，模型原生支持 | 不同模型 API 差异大，Kimi/DeepSeek 支持程度不一 |
| **Prompt-based XML** | 统一接口，所有模型支持 | 需要正则解析，容错性要求更高 |

**决策**：采用 `<tool_call>` XML 标签方式，通过 `ToolExecutor` 统一解析。已在 Kimi 上验证兼容。DeepSeek 存在不输出 XML 的问题（见 9.1）。

### 5.3 为什么事件队列超时后需要 drain？

**问题**：`event_queue.get(timeout=300)` 超时后 `break`，队列中剩余的 `tool_executing`、`model_done` 事件被丢弃。

**根因**：`_execute_tool_calls` 执行期间（最长 30s），消费者线程在等待队列事件。如果工具执行刚好在超时边界完成，`tool_executing` 事件入队时消费者已超时退出。

**解决**：
1. 超时从 30s 提升到 300s（适应 DeepSeek reasoning）
2. `break` 后增加 `drain` 循环，消费队列中所有剩余事件

### 5.4 为什么 DeepSeek 流式使用 Daemon 线程？

**问题**：`ThreadPoolExecutor` 的 `with` 上下文在退出时调用 `shutdown(wait=True)`。如果生产者的 SSE 流未正常关闭（DeepSeek 卡住），`shutdown` 会无限等待。

**解决**：使用 `threading.Thread(daemon=True)` 作为生产者，主线程通过 `queue.get(timeout=300)` 检测超时后正常退出，daemon 线程在后台自然消亡。

### 5.5 为什么将 tool_context 从 user prompt 迁移到 system prompt？

**问题**：DeepSeek 对 user prompt 中的工具使用说明遵循度低，频繁输出自然语言搜索意图（"让我先搜索一下..."）而非 XML 标签。

**决策**：将工具说明（可用工具列表 + XML 格式示例）合并到 system_prompt 尾部，利用模型对 system prompt 的更高遵循度。

**效果**：Kimi 遵循度提升明显；DeepSeek 仍存在问题（见 9.1）。

### 5.6 为什么引入角色-模型解耦？

**问题**：早期架构中模型 ID 直接作为状态 key，导致：
1. 同一模型无法同时扮演两个角色（ bull 和 bear 都用 deepseek 会冲突）
2. UI 显示的是模型名而非角色立场，用户难以区分
3. 评审 prompt 中无法注入角色化指导

**决策**：引入 `DebateRole`（bull/bear），状态全部以 `role_id` 为 key，`role_models` 映射 role→model。LLM 调用通过 `call_id=role_id` 区分同一模型的不同角色调用。

---

## 6. 错误处理与超时策略

### 6.1 三层超时保护

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: LLM 流式超时 (llm_client.call_stream)             │
│  - 机制: queue.get(timeout=300)                             │
│  - 触发: 300 秒内无新 token                                 │
│  - 行为: 抛出 RuntimeError，发送 error 事件                 │
│  - 备注: DeepSeek reasoning 可能持续数分钟                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 单工具执行超时 (ToolExecutor.extract_and_execute) │
│  - 机制: future.result(timeout=30)                          │
│  - 触发: 单个 tool call 执行超过 30 秒                      │
│  - 行为: 记录超时结果，继续处理其他 tool                     │
│  - 备注: 单轮最多 10 个 tool call                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 事件队列超时 (session.py 消费者循环)               │
│  - 机制: event_queue.get(timeout=300)                       │
│  - 触发: 300 秒内无新事件（角色线程卡住）                   │
│  - 行为: 检测存活线程，区分 API 卡住 vs 线程崩溃             │
│  - 备注: 超时后 drain 剩余事件                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 单模型失败隔离

```python
# stream_single 中的异常处理
try:
    for token, accumulated in self._client.call_stream(...):
        ...
    # 工具调用...
    # model_done 事件...
except Exception as e:
    # 发送 error 事件，但不影响其他角色
    event_queue.put({"type": "error", "role_id": role_id, "model_id": model_id, "error": str(e)})

# run_stream 中的处理
elif etype == "error":
    st.error(f"❌ {rid} 失败: {event['error']}")
    # 注意：不 return，让其他角色继续
```

### 6.3 工具链降级

```
Tavily (高质量搜索)
    │
    ├── 成功 → 返回 Tavily 结果
    │
    └── 失败 ──▶ DuckDuckGo (免费回退)
            │
            ├── 成功 → 返回 DDG 结果（标注回退）
            │
            └── 失败 → 返回合并错误信息
                      → 在 followup prompt 中标注"关键数据缺失"
                      → 强制置信度 < 50%
```

---

## 7. 扩展性设计

### 7.1 添加新模型

1. 在 `config.yaml` 中新增模型配置
2. 确保模型支持 Anthropic SDK 兼容接口
3. （可选）在 `llm_client.py` 的 token 提取逻辑中适配新模型的输出格式

### 7.2 添加新工具

1. 继承 `Tool` 基类，实现 `execute()` 方法
2. 在 `ToolRegistry.register_default_tools()` 中注册
3. 工具自动出现在 prompt 的可用工具列表中

### 7.3 添加新场景

1. 在 `config.yaml` 的 `scenarios` 中定义
2. 在 `prompts.py` 的 `SCENARIO_ROLES` 中定义系统角色
3. 在 `SCENARIO_SKILL_MAP` 中映射 Skill（可选）

### 7.4 添加新 Skill

1. 创建 `SKILL.md` 文件（YAML frontmatter + Markdown body）
2. 放置在 `~/.config/agents/skills/` 或 `./skills/`
3. 在 `SCENARIO_SKILL_MAP` 中建立场景映射（可选）

### 7.5 添加新角色

1. 在 `debater/roles.py` 的 `ROLES` 中新增 `DebateRole`
2. 定义 `generate_prompt` 和 `critique_prompt`
3. 在 UI 的 `role_models` 选择器中暴露新角色

---

## 8. 测试策略

### 8.1 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_config.py       # 配置加载
│   ├── test_session.py      # 会话状态机
│   ├── test_prompts.py      # 提示词构建
│   ├── test_memory.py       # 记忆系统
│   ├── test_skills/         # Skill 系统
│   └── test_tools/          # 工具系统
├── integration/             # 集成测试
└── e2e/                     # 端到端测试
```

### 8.2 关键测试覆盖

| 模块 | 测试重点 |
|------|----------|
| `config.py` | 多配置文件切换、环境变量覆盖 |
| `session.py` | 状态机转换、流式事件顺序、tool call 双轮流程、角色-模型解耦 |
| `tools/base.py` | XML 解析（标准格式 + DeepSeek 格式）、并行超时、数量上限 |
| `tools/search.py` | Tavily 失败回退 DDG、超时处理 |
| `llm_client.py` | 流式 token 提取（text/thinking/reasoning）、300s 超时触发、同步 fallback |
| `roles.py` | 角色获取、显示名生成 |

**当前状态**：211 tests passing（含 retry、brainstorm、工具调用、角色-模型解耦、串行交替辩论、step_stream 统一入口等核心流程覆盖）。

---

## 9. 已知限制与未来优化

### 9.1 当前限制（高优先级）

1. **DeepSeek 不输出 XML tool call（已部分缓解）**
   - **现象**：DeepSeek 输出自然语言搜索意图（"让我先搜索一下..."）后停止，不输出 `<tool_call>` 标签
   - **当前缓解**：系统回退搜索（检测到关键词后自动构造搜索）+ `include_tool_strategy` 在 user prompt 中注入 ❌/✅ 正误示例
   - **状态**：回退搜索机制已稳定运行；Kimi 对 XML 遵循度较好，DeepSeek 仍偶有偏差

2. **Prompt length bloat（critiques_history）**
   - **现象**：`critiques_history` 按 `round → critiquer → target` 存储，O(R×M²) 增长，第 5 轮超 8k tokens
   - **影响**：导致 API 调用成本上升，模型注意力分散
   - **建议修复**：截断到最近 2 轮，或生成压缩摘要替代完整历史

3. **Thread 积累**
   - **现象**：Daemon 线程在 DeepSeek 卡住后不会立即清理，可能积累（实际影响有限，API 连接有底层超时）

### 9.2 已修复问题（归档）

| 问题 | 修复日期 | 修复内容 |
|------|----------|----------|
| retry 面板无响应 | 2026-05-14 | `_role_models` 缺失导致模型切换后 key 不匹配；`retry_generate_role_stream` 未区分 brainstorm/generate prompt |
| `result_queue` 3 元组解包崩溃 | 2026-05-11 | consumer 支持 2/3/N 元组通用解包 |
| brainstorm 无工具调用 | 2026-05-11 | `brainstorm_stream_parallel` 补充完整的工具检测+执行+第二轮逻辑 |
| `_role_of_model` 并发 bug | 2026-05-11 | 同一模型双角色时永远返回 "bull"；改为 `build_prompt` 直接传 `role_id` |
| 并行 critique/revise 改为串行交替 | 2026-05-17 | `critique_stream_parallel` → `critique_stream_single(bear→bull)`；`revise_stream_parallel` → `revise_stream_single(bull)`；generate 只做一次，后续 critique→revise→judge 循环 |
| UI 层与业务逻辑不一致 | 2026-05-17 | streamlit_app.py 自己维护 auto_next_action 状态机；重构为统一调用 `session.step_stream()` + `session.auto_judge()`；手动模式按 `session.stage` 显示按钮 |
| DDG 搜索 SSL 崩溃 | 2026-05-11 | `DDGS(verify=False)` 绕过 Python 3.9 + macOS `TLSv1_3` 不兼容 |
| 第二轮空输出 | 2026-05-11 | 空内容时自动回退到第一轮输出 |
| `tool_use` content block 丢失 | 2026-05-11 | `_call_stream_impl` 新增 `delta.partial_json` + `content_block_start` 处理 |
| 回退搜索 XML 破坏 | 2026-05-11 | `html.escape()` 转义特殊字符 |
| Kimi "high risk" 安全拒绝 | 2026-05-11 | 第二轮捕获 `RuntimeError` 并检测 `"high risk"`，自动使用不含搜索结果的简化 prompt 重试 |
| 空结果不进 Round 2 | 2026-05-11 | `_execute_tool_calls` 空结果时返回错误提示文本（非空 `tool_info["text"]`）

### 9.3 当前限制（低优先级）

4. **Working Memory 硬编码**：Memory 更新和压缩调用 LLM，成本较高
5. **Skill 注入长度**：Skill body 截断 3000 字符，可能丢失关键细节
6. **Obsidian 生成**：仅在辩论结束时生成，不支持中途导出

### 9.4 未来优化方向

1. **流式聚合**：当前 Aggregate 阶段是同步调用，可改为流式
2. **动态模型选择**：根据问题类型自动选择最合适的模型组合
3. **工具结果缓存**：相同查询缓存结果，减少 API 调用
4. **可视化增强**：辩论树状图、置信度变化曲线、分歧热力图
5. **多语言支持**：当前主要针对中文投研场景
6. **Native Function Calling**：为支持原生 function calling 的模型（如 Claude）提供备选路径

---

## 10. 附录

### 10.1 文件索引

| 文件 | 职责 |
|------|------|
| `streamlit_app.py` | Streamlit UI 主入口 |
| `debater/session.py` | 手动控制流 DebateSession（核心） |
| `debater/llm_client.py` | LLM 统一客户端 |
| `debater/config.py` | 配置加载与管理 |
| `debater/prompts.py` | 提示词模板 |
| `debater/roles.py` | 辩论角色定义（Bull/Bear） |
| `debater/memory.py` | 三层记忆系统 |
| `debater/tools/base.py` | Tool 基类与执行器 |
| `debater/tools/registry.py` | Tool 注册表 |
| `debater/tools/search.py` | Tavily 搜索 + DDG 回退 |
| `debater/skills/registry.py` | Skill 注册表 |
| `debater/skills/loader.py` | Skill 加载器 |
| `debater/nodes.py` | LangGraph 节点（参考实现） |
| `debater/graph.py` | LangGraph 图定义（参考实现） |
| `debater/state.py` | LangGraph 状态定义 |

### 10.2 关键配置项

```yaml
models:
  <model_id>:
    name: str              # 显示名称
    model_id: str          # API 模型 ID
    base_url: str          # API Base URL
    max_tokens: int        # 最大输出长度
    role: str              # proposer | aggregator
    critique_style: str    # standard | adversarial
    extra_body: dict       # 模型特定参数

debate:
  consensus_threshold: 0.85
  min_confidence: 0.70
```

### 10.3 状态 Key 约定

| 字典 | Key 类型 | 示例 |
|------|----------|------|
| `self.answers` | `role_id` | `answers["bull"]`, `answers["bear"]` |
| `self.thinkings` | `role_id` | `thinkings["bull"]` |
| `self.confidences` | `role_id` | `confidences["bear"]` |
| `self.critiques` | `role_id` → `target_role_id` | `critiques["bear"]["bull"]` |
| `self.critiques_history` | `round_num` → `critiquer` → `target` | `critiques_history[1]["bear"]["bull"]`（串行模式下只记录 bear→bull） |
| `self.role_models` | `role_id` → `model_id` | `role_models["bull"] = "kimi_k2.6"` |
| `call_parallel()` 结果 | `call_id` (= `role_id`) | `results["bull"]` |
| UI `phs` / `tool_phs` | `role_id` | `phs["bull"]` |
| `st.session_state.live_texts` | `role_id` | `live_texts["bear"]` |
