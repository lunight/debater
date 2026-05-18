# 多模型辩论系统 (Multi-Model Debate System)

基于 **LangGraph** 的多模型协作辩论框架。多个 LLM 模型围绕一个问题进行多轮讨论、互相评审、修订观点，最终由聚合器输出统一结论。

## 核心特性

- 🏗️ **LangGraph 状态机**：清晰的节点流转，支持循环辩论、条件分支
- 🤖 **多模型并行**：多个模型同时生成、评审、修订，最大化效率
- 🔄 **多轮辩论**：模型互相质疑、补充、修正，逐步收敛到更优结论
- 📊 **置信度追踪**：每个模型给出置信度，辅助判断共识达成
- 🎯 **场景定制**：针对股票分析四大场景（问题分析、财务解读、事件分析、风险评估）优化角色设定
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

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 ANTHROPIC_API_KEY
```

## 配置模型

编辑 `config.yaml`：

```yaml
models:
  kimi_k2.6:
    name: "kimi-coding-k2.6"
    provider: "anthropic"
    model_id: "kimi-coding-k2.6"
    api_key_env: "ANTHROPIC_API_KEY"
    base_url: "https://your-api-base.com/v1"  # 如有自定义 base_url
    temperature: 0.3
    max_tokens: 4096
    role: "proposer"

  deepseek_v4:
    name: "deepseek-v4"
    provider: "anthropic"
    model_id: "deepseek-v4"
    api_key_env: "ANTHROPIC_API_KEY"
    base_url: "https://your-api-base.com/v1"
    temperature: 0.3
    max_tokens: 4096
    role: "proposer"
```

> `role: proposer` 表示参与辩论的模型。可以配置任意多个。`role: aggregator` 为最终聚合模型（默认第一个模型兼任）。

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

# 保存结果
python -m debater.main "某事件对股价的影响" -o result.json
```

### 2. Python API

```python
from debater.graph import run_debate

result = run_debate(
    question="分析该监管政策对行业的影响",
    context="政策内容...",
    scenario_type="event_analysis",  # 场景类型
    max_rounds=2,                   # 最大辩论轮次
)

print(result["final_answer"])          # 最终结论
print(result["confidences"])           # 各模型置信度
print(result["reasoning_process"])     # 完整推理过程
```

### 3. 运行示例

```bash
# 财务数据解读示例
python examples/stock_analysis_example.py financial

# 风险评估示例
python examples/stock_analysis_example.py risk

# 事件分析示例
python examples/stock_analysis_example.py event

# 运行全部示例
python examples/stock_analysis_example.py all
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
2. **Critique（评审）**：每个模型评审其他模型的观点，指出问题或补充
3. **Revise（修订）**：模型根据收到的评审意见修订自己的结论
4. **Judge（裁判）**：检查是否达成共识（置信度均 > 85%）或达到最大轮次
5. **Aggregate（聚合）**：综合所有观点，输出最终投资级结论

## 项目结构

```
debater/
├── config.yaml              # 模型与辩论配置
├── .env                     # API Key 环境变量
├── requirements.txt         # 依赖
├── debater/
│   ├── config.py            # 配置加载
│   ├── llm_client.py        # 统一 LLM 客户端（支持多模型并行）
│   ├── state.py             # LangGraph 状态定义
│   ├── prompts.py           # Prompt 模板（按场景定制）
│   ├── nodes.py             # 各节点逻辑实现
│   ├── graph.py             # 图定义与编排
│   └── main.py              # CLI 入口
└── examples/
    └── stock_analysis_example.py  # 股票分析示例
```

## 扩展建议

- **接入更多模型**：在 `config.yaml` 中增加模型配置即可
- **自定义场景**：在 `prompts.py` 的 `SCENARIO_ROLES` 中增加场景角色
- **Judge LLM-based**：当前 Judge 使用启发式规则，可改为调用 LLM 做更精细判断（修改 `nodes.py` 的 `judge` 方法）
- **持久化存储**：在 `aggregate` 节点后将结果写入数据库
- **Web 接口**：用 FastAPI 包装 `run_debate` 函数提供 HTTP API
