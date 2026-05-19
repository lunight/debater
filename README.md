# 多模型辩论系统 (Multi-Model Debate System)

基于 **LangGraph** 的多模型协作辩论框架。多个 LLM 模型围绕一个问题进行多轮讨论、互相评审、修订观点，最终输出统一结论。

## 核心特性

- 🏗️ **LangGraph 状态机**：清晰的节点流转，支持循环辩论、条件分支
- 🤖 **多模型并行**：多个模型同时生成、评审、修订
- 🔄 **多轮辩论**：模型互相质疑、补充、修正，逐步收敛
- 📊 **置信度追踪**：每个模型给出置信度，辅助判断共识达成
- 🛠️ **工具调用**：模型可调用 web_search、web_fetch 等工具获取实时信息
- 🧠 **记忆系统**：支持长期记忆（facts）和工作记忆（working memory）
- 🎯 **场景定制**：针对金融分析四大场景优化角色设定
- 💾 **完整追溯**：保存完整的推理过程和分歧点

## 架构

```
┌─────────────┐
│  Initialize │
└──────┬──────┘
       ↓
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Generate  │────→│   Critique  │────→│    Revise   │
│  (并行生成)  │     │  (并行评审)  │     │  (并行修订)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                ↓
                                          ┌─────────────┐
                                          │    Judge    │
                                          │   (裁判判断) │
                                          └──────┬──────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │ 未收敛                      │                            │ 已收敛/最大轮次
                    ↓                            ↓                            ↓
              ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
              │   Critique  │←─────────────│    Revise   │              │  Aggregate  │
              │   (下一轮)   │              │   (修订后)   │              │  (最终聚合)  │
              └─────────────┘              └─────────────┘              └──────┬──────┘
                                                                               ↓
                                                                        ┌─────────────┐
                                                                        │     END     │
                                                                        └─────────────┘
```

## 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 复制配置文件模板
cp config.yaml.template config.yaml
# 编辑 config.yaml，填入你的 API Key（或配置对应的环境变量）
```

## 配置

编辑 `config.yaml`：

```yaml
models:
  kimi_k2.6:
    name: "kimi-coding-k2.6"
    provider: "anthropic"
    model_id: "kimi-for-coding"
    api_key_env: "KIMI_API_KEY"      # 从环境变量读取（推荐）
    # api_key: "your-key-here"       # 或直接填写（不安全）
    base_url: "https://api.kimi.com/coding"
    temperature: 0.3
    max_tokens: 60000
    role: "proposer"
    extra_body:
      reasoning_effort: high

  deepseek_v4:
    name: "deepseek-v4"
    provider: "anthropic"
    model_id: "deepseek-v4-pro"
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com/anthropic"
    temperature: 0.3
    max_tokens: 60000
    role: "proposer"
    critique_style: "adversarial"
```

> `role: proposer` 表示参与辩论的模型，可配置多个。`role: aggregator` 为最终聚合模型（默认用第一个模型兼任）。

对应的环境变量（可选，写入 `.env` 文件）：

```bash
KIMI_API_KEY=your_kimi_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## 使用方式

### 1. CLI 命令行

```bash
# 基础用法
python -m debater.main "分析某公司2023年财报"

# 指定场景和上下文
python -m debater.main \
  "判断该公司盈利能力是否改善" \
  -c "营业收入：2021年100亿，2022年150亿，2023年180亿..." \
  -s financial_interpretation \
  -r 2

# 保存结果到 JSON
python -m debater.main "某事件对股价的影响" -o result.json
```

### 2. Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

### 3. Python API

```python
from debater.graph import run_debate

result = run_debate(
    question="分析该监管政策对行业的影响",
    context="政策内容...",
    scenario_type="event_analysis",
    max_rounds=2,
)

print(result["final_answer"])
print(result["confidences"])
```

### 4. 运行示例

```bash
python examples/stock_analysis_example.py
```

## 场景类型

| 场景 | 用途 | 模型角色 |
|------|------|---------|
| `question_analysis` | 一般性问题多角度分析 | 金融分析师 |
| `financial_interpretation` | 财务报表、指标解读 | 财务分析师 |
| `event_analysis` | 事件影响评估 | 行业研究员 |
| `risk_assessment` | 风险识别与评估 | 风险管理专家 |

## 辩论流程

1. **Generate（生成）**：所有模型并行给出初始分析和结论
2. **Critique（评审）**：双方按轮次交替评审（奇数轮 bear→bull，偶数轮 bull→bear）
3. **Revise（修订）**：被评审方根据评审意见修订自己的结论
4. **Judge（裁判）**：检查是否达成共识或达到最大轮次
5. **Aggregate（聚合）**：综合所有观点，输出最终结论

## 项目结构

```
debater/
├── config.yaml.template       # 配置模板（复制为 config.yaml 后编辑）
├── .env.example               # 环境变量示例
├── requirements.txt
├── streamlit_app.py           # Streamlit Web UI
├── interactive.py             # 交互式入口
├── debater/
│   ├── config.py              # 配置加载与管理
│   ├── llm_client.py          # 统一 LLM 客户端
│   ├── engine.py              # 公共引擎（置信度提取、Judge 解析等纯函数）
│   ├── state.py               # LangGraph 状态定义
│   ├── prompts.py             # Prompt 模板
│   ├── roles.py               # 角色定义
│   ├── nodes.py               # 各节点逻辑实现
│   ├── graph.py               # 图定义与编排
│   ├── session.py             # 会话管理
│   ├── memory.py              # 记忆系统
│   ├── logger.py              # 日志系统
│   ├── main.py                # CLI 入口
│   ├── skills/                # 技能系统
│   │   ├── loader.py
│   │   └── registry.py
│   └── tools/                 # 工具系统
│       ├── base.py
│       ├── search.py          # Tavily 搜索
│       ├── ddg_search.py      # DuckDuckGo 搜索
│       ├── fetch.py           # 网页抓取
│       ├── filesystem.py      # 文件系统工具
│       ├── github_search.py   # GitHub 搜索
│       └── registry.py
├── doc/
│   ├── DESIGN.md
│   └── FLOW.md
├── examples/
│   └── stock_analysis_example.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```
