# 智能工厂 + 金融投顾 Agent Teams Demo

基于 CrewAI 的多域融合 Agent Teams 演示。

## 场景

一家中型电子制造企业，需要同时评估工厂运营和融资/投资策略。3 个 Agent 自主协作，输出跨域综合分析报告。

## Agent 团队

| Agent | 角色 | 职责 |
|-------|------|------|
| Factory Analyst | 智能工厂数据分析师 | 分析产线效率、设备风险、运营优化 |
| Finance Advisor | 金融投顾分析师 | 财务诊断、融资方案、资金策略 |
| Synthesizer | 跨域综合策略师 | 整合运营+金融视角，输出综合方案 |

## 执行流程

```
工厂分析 ──→ 金融分析 ──→ 跨域综合
(Task 1)     (Task 2)     (Task 3, 依赖 1&2)
```

## 快速开始

```bash
# 1. 设置 API Key（DeepSeek 或 OpenAI）
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.deepseek.com"  # 用 DeepSeek 时

# 2. 激活环境
source venv/bin/activate

# 3. 运行
python main.py
```

## 输出

运行后在 `output_report.md` 中生成综合策略报告。
