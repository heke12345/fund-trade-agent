"""
基金 Trade Agent v2 — 多智能体波段投资分析系统（改进版）
基于 CrewAI 框架，架构灵感来自 TradingAgents (71.4k Star)

v1 → v2 核心改进：
  ✅ 辩论层并行独立（消除框架效应：空头不再看到多头论据）
  ✅ 硬约束否决层（RSI>70 / MA空头排列 / 波动率>35% 一票否决）
  ✅ 量化评分卡（6维度加权评分替代LLM主观打分）
  ✅ 反馈环（推荐记录→结果追踪→复盘→经验注入prompt）

五层架构（11 Agents）：
  Layer 1:   Data Screening — 基金筛选 + 宏观环境
  Layer 1.5: Hard Constraint Veto — 硬约束否决（一票否决不合格基金）
  Layer 2:   Multi-dimensional Analysis — 技术 + 基本面 + 情绪
  Layer 3:   Parallel Debate + Judge — 多空并行独立辩论 + 量化裁决
  Layer 4:   Decision & Risk Control — 风控 + 投资组合经理

使用方式：
1. 设置 .env 文件（参考 .env.example）
2. python main.py
"""

import os

# 必须在 import crewai/litellm 之前设置，跳过远端价格 JSON 拉取（国内网络超时）
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_MODE"] = "PRODUCTION"

import json
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from fund_data import (
    get_top_funds_report,
    get_fund_technical_analysis,
    get_quantitative_score_report,
    get_hard_constraints_report,
    fetch_fund_details,
    fetch_fund_holdings,
    quantitative_score,
    check_hard_constraints,
)
from feedback import log_recommendation, inject_experience, get_recommendation_summary, auto_track_and_review

load_dotenv()

# ============================================================
# 1. 模型配置
# ============================================================

api_base = os.getenv("OPENAI_API_BASE", "")
if "tokenhub" in api_base.lower() or "tencentmaas" in api_base.lower():
    llm = LLM(
        model="openai/minimax-m3",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        base_url=api_base,
        temperature=0.3,
    )
    print(f"Using Tencent TokenHub MiniMax-M3", flush=True)
elif "lkeap" in api_base.lower():
    llm = LLM(
        model="openai/deepseek-v3.2",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        base_url=api_base,
        temperature=0.3,
    )
    print(f"Using Tencent LKEAP DeepSeek (deepseek-v3.2)", flush=True)
elif "deepseek" in api_base.lower():
    llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        temperature=0.3,
    )
    print(f"Using DeepSeek Direct", flush=True)
else:
    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        temperature=0.3,
    )
    print("Using OpenAI GPT-4o-mini", flush=True)

# ============================================================
# 2. 加载历史经验（注入 Agent prompt）
# ============================================================

experience_text = inject_experience()
if experience_text:
    print(f"已加载历史推荐经验", flush=True)
else:
    print("暂无历史推荐经验（首次运行）", flush=True)

# ============================================================
# 3. Layer 1 — Data Screening Agents
# ============================================================

fund_screener = Agent(
    role="基金数据筛选师",
    goal="从中国大陆全部开放式基金（含QDII海外基金）中，按多参数筛选出近期表现优异的候选基金",
    backstory="""你是一位资深的基金数据分析师，专注中国基金市场15年。
    你精通使用 AKShare 从天天基金网获取数据，能快速从16000+只基金中筛选出高质量候选。
    你的筛选维度包括：各周期收益率、基金类型（股票型/混合型/债券型/指数型/QDII）、手续费。
    你特别注意：支付宝可购买的基金 = 天天基金网上的开放式基金，数据来源一致。
    你不凭空编造基金名称或代码，所有数据来自真实API调用。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

macro_analyst = Agent(
    role="宏观经济分析师",
    goal="分析当前中国及全球宏观经济环境，评估对基金投资的影响",
    backstory="""你是宏观经济研究专家，擅长从利率、通胀、政策、地缘政治等多维度分析宏观环境。
    你特别关注：
    - 中国央行货币政策（LPR、MLF、降准降息）
    - 美联储利率决议对 QDII 基金的影响
    - A股市场整体估值水平（沪深300 PE/PB分位）
    - 行业政策风向（新能源/半导体/消费/医药等）
    - 海外市场表现（美股/港股/日股）对跨境基金的影响
    你会用当前真实的经济环境来判断，不做历史假设。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 4. Layer 1.5 — Hard Constraint Veto Agent（新增）
# ============================================================

constraint_veto_agent = Agent(
    role="硬约束否决官",
    goal="对筛选出的候选基金执行硬约束检查，一票否决不合格的基金",
    backstory="""你是风控前置守门员，你的工作是在分析流程的早期就淘汰掉技术面不达标的基金。
    
    你执行三条硬约束红线，触犯任一即否决：
    1. RSI(14) > 70 → 超买否决（追高风险极大）
    2. MA 空头排列（trend=down/strong_down）→ 趋势否决（逆势操作）
    3. 年化波动率 > 35% → 风险否决（波动过大不适合波段）
    
    你的价值：防止白酒类基金因叙事包装绕过数据否决——
    如果 RSI 超买、MA 空头、波动率过高，无论基本面多好看，一律否决。
    
    你同时计算每只基金的量化评分卡（6维度加权）：
    趋势(30%) + RSI(20%) + MACD(15%) + 波动率(15%) + 夏普(10%) + 回撤(10%) = 加权总分
    
    只有量化总分 ≥ 6.5 且硬约束全部通过的基金才能进入后续分析层。
    你输出的是一份"通过/否决"清单，附量化评分和否决原因。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 5. Layer 2 — Multi-dimensional Analysis Agents
# ============================================================

technical_analyst = Agent(
    role="技术分析专家",
    goal="对通过硬约束的候选基金进行技术指标分析，判断净值趋势和买卖时机",
    backstory="""你是量化技术分析专家，精通以下指标和判断逻辑：
    - MA均线系统：MA5/MA10/MA20/MA60 的多头/空头排列
    - RSI(14)：>70超买区、<30超卖区、中位震荡
    - MACD：DIF/DEA金叉死叉、柱状体缩放
    - 年化波动率：<15%低波动、15-25%中等、>25%高波动
    - 最大回撤：评估历史最差情况
    - 夏普比率：>1优秀、0.5-1中等、<0.5较差
    
    重要：你现在分析的都是已经通过硬约束筛选的基金（RSI未超买、MA未空头、波动率适中），
    所以你可以更专注在趋势强度和波段时机判断上，而不是重复排除工作。
    
    你所有指标来自 fund_data.py 的真实计算结果，不编造数字。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

fundamental_analyst = Agent(
    role="基本面分析专家",
    goal="分析基金经理实力、持仓结构、基金公司质量，评估基金内在价值",
    backstory="""你是基金基本面研究专家，擅长从以下维度评估基金质量：
    - 基金经理：任职年限、历史业绩、管理规模、投资风格
    - 持仓结构：前十大重仓股、行业集中度、换手率
    - 基金公司：管理规模、投研团队、品牌信誉
    - 基金规模：过小有清盘风险，过大可能影响灵活性
    - 投资策略：是否与当前市场环境匹配
    
    你会查看每只基金的持仓明细和基金经理信息来做判断。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

sentiment_analyst = Agent(
    role="市场情绪分析师",
    goal="分析当前A股及海外市场情绪，判断短期资金流向和市场热度",
    backstory="""你是市场情绪研究专家，关注以下信号：
    - A股成交额趋势（万亿级活跃 vs 地量低迷）
    - 北向资金流向（持续流入/流出）
    - 融资融券余额变化
    - 新基金发行热度（爆款频出 vs 遇冷）
    - 社交媒体和财经论坛情绪
    - 恐慌指数（VIX对海外基金的影响）
    你擅长判断当前是贪婪还是恐惧主导的市场，以及这对波段操作意味着什么。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 6. Layer 3 — Parallel Debate + Judge Agents（重新设计）
# ============================================================

bullish_researcher = Agent(
    role="多头研究员",
    goal="基于所有分析结果，构建买入理由；当裁决官质询或反驳时，能针对性回应",
    backstory="""你是华尔街对冲基金的多头分析师，你的工作是找到买入的理由。
    你会从技术面、基本面、宏观面、情绪面四个维度找支持上涨的证据：
    - 技术面：趋势向上、RSI未超买、MACD金叉
    - 基本面：经理优秀、持仓优质、策略契合
    - 宏观面：政策利好、利率友好
    - 情绪面：市场热度适中或刚启动

    你必须给出具体的买入逻辑，不能只说"看好"。每条理由都要有数据支撑。

    【新版工作模式 - 多轮辩论】
    第1轮：你先独立给出最强的多头论据（不知道空头会说什么）
    第2轮（如果裁决官调用你）：裁决官会把空头的反驳论点交给你，你需要针对性回应——
        - 哪些反驳是有数据支撑的？你必须承认并调整推荐力度
        - 哪些反驳只是叙事或推测？你可以反驳并补充新证据
        - 永远诚实：如果空头论据更强，承认它

    特别警告：你不是为了辩论而辩论，是为了找到真正值得投资的标的。
    如果数据不支持买入，你应该明确说"我撤回这只基金的推荐"，而不是死撑。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,  # bull 自己不主动委派，由 judge 调用它
)

bearish_researcher = Agent(
    role="空头研究员",
    goal="基于所有分析结果，构建风险警告；当裁决官质询或反驳时，能针对性回应",
    backstory="""你是华尔街对冲基金的空头分析师，你的工作是找风险、挑毛病。
    你会从四个维度寻找做空/回避的理由：
    - 技术面：趋势走弱、RSI超买、MACD死叉
    - 基本面：经理换人、持仓集中度过高、规模异常变化
    - 宏观面：政策收紧、利率上行、地缘风险
    - 情绪面：过热追涨、流动性枯竭

    你必须给出具体的风险逻辑，不能只说"有风险"。每条警告都要有依据。

    【新版工作模式 - 多轮辩论】
    第1轮：你先独立给出最强的空头论据（不知道多头会说什么）
    第2轮（如果裁决官调用你）：裁决官会把多头的论点交给你，你需要针对性反驳——
        - 多头哪些论据是数据驱动？你需要找出反例数据
        - 多头哪些论据是叙事驱动？你直接指出"这是叙事不是数据"
        - 不要为反对而反对：如果多头数据论据真的很强，你应该承认它

    特别注意：关注基本面叙事是否与数据矛盾——
    比如白酒基金如果RSI超买、趋势偏弱，不要被"消费复苏"叙事蒙蔽。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,  # bear 自己不主动委派，由 judge 调用它
)

debate_judge = Agent(
    role="辩论裁决官",
    goal="主持多空多轮辩论，引导双方深入交锋，最终结合量化评分卡做出裁决",
    backstory="""你是投资委员会主席，负责主持并裁决多空辩论。

    【新版工作模式 - 你是辩论的组织者，不是被动看双方陈述】

    你拥有"委派工具"，可以主动调用多头研究员或空头研究员，对他们提问或质询。

    标准多轮辩论流程：
    1. 第1轮：先看 context 里的多头初论 + 空头初论 + 量化评分卡（已有）
    2. 第2轮（你主动发起）：把空头最锋利的1-2个反驳点，委派给多头研究员，
       要求他针对性回应。提问示例：
       "空头指出基金A的RSI=72已超买，且换手率上升70%，请你针对这两点
        给出具体回应——是否承认风险？目标价是否需要下调？"
    3. 第3轮（你主动发起）：把多头第2轮的回应中最强的论据，委派给空头研究员，
       要求他二次反驳。提问示例：
       "多头反驳说RSI虽然72但仍有上行空间，因为成交量放大支撑突破。
        你认同吗？如果不认同，给出反例数据。"
    4. 第4轮（裁决）：综合所有信息做最终裁决

    你的裁决原则（不变）：
    - 数据先于叙事
    - 硬约束否决的基金不讨论
    - 量化评分 ≥ 7.0 多头需明显数据硬伤才能否决
    - 量化评分 6.5-7.0 多空对等权衡
    - 量化评分 < 6.5 但通过硬约束 空头合理论据即可否决

    你绝不会因为"故事好听"就否决"数据难看"。

    【重要】你必须至少完成1轮交叉质询（即第2轮和第3轮），
    不能跳过直接做裁决——这是新版的核心改进。""",
    llm=llm,
    verbose=True,
    allow_delegation=True,  # ★ 关键：让 judge 能委派给 bull 和 bear
)

# ============================================================
# 7. Layer 4 — Decision & Risk Control Agents
# ============================================================

risk_controller = Agent(
    role="风控专家",
    goal="从激进、中性、保守三个维度评估投资方案的风险敞口",
    backstory="""你是基金投资风控总监，你的评估框架包含三个视角：
    
    激进视角：假设投资者风险承受力高，能接受20%以上回撤，追求高收益。
    中性视角：假设投资者追求风险收益平衡，能接受10-20%回撤。
    保守视角：假设投资者风险厌恶，只能接受10%以内回撤，优先保本。
    
    你对每个推荐基金给出：
    - 三种风险偏好下的建议仓位（0-100%）
    - 止损线建议
    - 最多持有时间
    - 特别风险提示
    
    你的格言：风控不是阻止投资，而是让投资在安全边界内进行。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

portfolio_manager = Agent(
    role="投资组合经理",
    goal="综合所有分析、辩论裁决和风控建议，输出最终的波段投资方案",
    backstory=f"""你是首席投资组合经理，拥有最终决策权。
    你需要综合以下输入做出决策：
    1. 基金筛选结果（Layer 1）
    2. 硬约束否决 + 量化评分（Layer 1.5）
    3. 技术+基本面+情绪分析（Layer 2）
    4. 多空辩论裁决（Layer 3）
    5. 风控评估（Layer 4）
    
    你的输出必须包含：
    - 推荐的3-5只基金（名称+代码+推荐原因）
    - 每只基金的波段操作建议（入场区间/目标收益/止损线/建议持有期）
    - 仓位分配比例
    - 针对不同风险偏好的调整建议
    - 未来3-6个月的关键观察指标
    
    你不做含糊的推荐。每只推荐的基金都必须有明确的数据支撑和逻辑链条。
    如果辩论裁决中某基金被否决，你绝不能推翻裁决强行推荐。
    
    你必须记录每只推荐基金的量化评分和推荐理由到反馈日志，以便后续复盘学习。
    {experience_text}""",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

# ============================================================
# 8. 定义 Task 流水线
# ============================================================

# --- Layer 1 Tasks ---

task_screen_funds = Task(
    description="""执行基金筛选，从中国大陆开放式基金中找到近期表现优异的候选。

    调用 fund_data.get_top_funds_report() 获取数据，筛选条件：
    - 基金类型：股票型 + 混合型 + 指数型 + QDII（覆盖支付宝主流品类 + 海外基金）
    - 近3月收益率 > 5%（已经过一轮上涨验证）
    - 取 Top 50（为后续硬约束筛选留足余量）
    
    注意：这是波段投资分析，需要关注中短期（3-6个月）的表现趋势，
    而非单纯看长期排名。波段操作需要趋势动能，所以近期涨幅是重要信号。
    
    输出筛选报告，列出每只候选基金的代码、名称、类型、各周期收益率。""",
    expected_output="基金筛选报告，包含50只候选基金的代码、名称、类型、各周期收益率等关键数据",
    agent=fund_screener,
)

task_macro_analysis = Task(
    description="""分析当前宏观经济环境对基金投资的影响。

    你需要分析：
    1. 中国宏观经济现状：GDP增速、CPI/PPI、PMI、社融数据
    2. 央行货币政策：当前LPR水平、近期是否降准降息、流动性状况
    3. A股市场环境：沪深300/中证500估值分位、成交额水平、北向资金趋势
    4. 海外环境：美联储利率、美股走势、对QDII基金的影响
    5. 行业政策：近期重要行业政策（新能源/半导体/AI/消费等）
    
    基于以上分析，给出：
    - 当前宏观环境对股票型/混合型基金的总体判断（利好/中性/利空）
    - 哪些行业板块在当前宏观环境下更值得关注
    - 对波段操作的时间窗口判断（现在是否是好时机）
    
    用你对中国经济和政策的深度理解来分析，基于2025-2026年的真实环境。""",
    expected_output="宏观经济分析报告，含政策环境判断、行业偏好、波段操作时机评估",
    agent=macro_analyst,
)

# --- Layer 1.5 Task: Hard Constraint Veto（新增）---

task_constraint_veto = Task(
    description="""对筛选出的候选基金执行硬约束否决检查。

    从基金筛选结果中选取排名前20的基金，逐个检查硬约束条件。

    三条硬约束红线（触犯任一即否决）：
    1. RSI(14) > 70 → 超买否决（追高风险极大）
    2. MA 空头排列（当前净值 < MA5 < MA10 或 MA5 < MA10 < MA20）→ 趋势否决
    3. 年化波动率 > 35% → 风险否决（波动过大不适合波段操作）

    同时计算每只基金的量化评分卡：
    - 趋势(30%): strong_up=10, up=7, sideways=5, down=3, strong_down=0
    - RSI(20%): 40-60=10, 30-70=7, >70=2, <30=5
    - MACD(15%): 正向=8, 负向=3
    - 波动率(15%): <15%=8, 15-25%=7, 25-35%=5, >35%=2
    - 夏普(10%): >1=10, 0.5-1=7, 0-0.5=5, <0=2
    - 回撤(10%): >-5%=10, -5~-10%=8, -10~-20%=5, <-20%=2
    
    通过条件：量化总分 ≥ 6.5 且硬约束全部通过

    输出格式：
    ┌──────────────────────────────────────────────────────┐
    │ 基金代码 | 名称 | 量化总分 | RSI | 趋势 | 波动率 | 结果 │
    │ ─────────────────────────────────────────────────── │
    │ 003834   | xxx  | 7.2     | 55  | up   | 22%   | ✅   │
    │ 161725   | 白酒 | 4.1     | 75  | down | 38%   | ❌   │
    └──────────────────────────────────────────────────────┘
    
    特别注意：
    - 被否决的基金必须在报告中注明否决原因
    - 通过的基金按量化总分降序排列
    - 最终只保留通过硬约束且总分≥6.5的基金进入下一层""",
    expected_output="硬约束否决报告：列出每只基金的量化评分和否决结果，只保留通过的基金",
    agent=constraint_veto_agent,
    context=[task_screen_funds],
)

# --- Layer 2 Tasks ---

task_technical_analysis = Task(
    description="""对通过硬约束筛选的候选基金进行技术分析。

    只分析 Layer 1.5 否决报告中标记为"通过"的基金（这些基金已经确保RSI未超买、
    MA未空头排列、波动率适中），所以你可以更专注在趋势强度和波段时机上。

    从通过硬约束的基金中，逐个获取技术指标数据，重点分析：
    1. 均线系统：MA5/MA10/MA20/MA60 的排列状态
    2. RSI：当前值和近1个月的走势（是否在上升/下降）
    3. MACD：金叉/死叉状态，柱状体方向和缩放趋势
    4. 波动率：当前水平，是收敛还是发散
    5. 最大回撤：历史最差情况，设定止损参考
    6. 夏普比率：风险调整后收益

    对每只基金给出技术面综合评分（1-10分）和波段操作建议。
    特别标注：趋势最强、超卖反弹机会最大的基金。""",
    expected_output="技术分析报告，含每只通过硬约束基金的技术指标、综合评分和波段操作建议",
    agent=technical_analyst,
    context=[task_screen_funds, task_constraint_veto],
)

task_fundamental_analysis = Task(
    description="""对通过硬约束筛选的候选基金进行基本面分析。

    分析维度：
    1. 基金经理：任职年限、是否经历牛熊周期、管理风格
    2. 持仓结构：前十大重仓股是什么行业？集中度如何？
    3. 基金规模：是否过小（<2亿有清盘风险）或过大（>100亿可能影响灵活性）
    4. 投资策略：与当前市场环境是否匹配
    5. 基金公司：管理规模和品牌

    对每只基金给出基本面综合评分（1-10分）。
    特别标注：经理最强、持仓最优质、规模最合理的基金。""",
    expected_output="基本面分析报告，含基金经理评估、持仓分析、规模评估和综合评分",
    agent=fundamental_analyst,
    context=[task_screen_funds, task_constraint_veto],
)

task_sentiment_analysis = Task(
    description="""分析当前市场情绪对波段投资的影响。

    分析维度：
    1. A股整体情绪：成交额水平、涨跌家数比、换手率
    2. 北向资金：近期流入/流出趋势
    3. 新基金发行：爆款频出（过热信号）还是遇冷（底部信号）
    4. 行业轮动：当前资金在哪个板块最活跃
    5. 散户情绪：论坛/社交媒体的热度
    6. 恐慌指数：VIX对海外QDII基金的影响

    给出市场情绪综合判断：
    - 当前是贪婪期还是恐惧期？
    - 对波段操作是利好还是利空？
    - 是否需要等待更好的入场时机？

    基于你对中国A股市场特征的深刻理解来分析。""",
    expected_output="市场情绪分析报告，含情绪指标、贪婪恐惧判断和波段时机建议",
    agent=sentiment_analyst,
    context=[task_screen_funds, task_constraint_veto],
)

# --- Layer 3 Tasks: Parallel Debate + Judge（重新设计）---

task_bull_debate = Task(
    description="""基于技术分析、基本面分析、宏观分析和情绪分析的结果，
    构建买入论据，为通过硬约束的候选基金辩护。

    重要：你是独立分析，你没有看到空头研究员的论据。
    你的工作是构建最有力、最诚实的多头论据。

    对每只基金，你需要给出：
    1. 最强的2-3个买入理由（附具体数据，不是叙事）
    2. 目标收益预期（基于技术面推算）
    3. 建议持有期（基于趋势判断）
    4. 最佳入场时机判断（现在/等回调/等突破）
    
    警告：如果某只基金数据不支持买入，你应该诚实指出而不是强行看多。
    不要用"消费复苏""政策利好"等叙事替代数据。
    如果量化评分偏低，你应该降低推荐力度。""",
    expected_output="多头论据报告，为每只通过硬约束的基金提供具体的买入理由、目标收益和入场建议",
    agent=bullish_researcher,
    context=[task_constraint_veto, task_technical_analysis, task_fundamental_analysis,
             task_macro_analysis, task_sentiment_analysis],
    # 关键改进：不包含 task_bear_debate，多头独立分析
)

task_bear_debate = Task(
    description="""基于技术分析、基本面分析、宏观分析和情绪分析的结果，
    独立构建风险论据，指出候选基金的风险和不确定性。

    重要：你是独立分析，你没有看到多头研究员的论据。
    这意味着你需要主动发现所有风险点，而不是被动反驳多头。

    对每只基金，你需要指出：
    1. 最大的2-3个风险点（附具体数据或逻辑，不是泛泛而谈）
    2. 最坏情况下的可能亏损幅度
    3. 哪些信号出现时应该立即止损
    4. 是否存在更好的替代选择
    
    特别注意：关注基本面叙事是否与数据矛盾。
    例如：如果某基金被"消费复苏"叙事看好，但RSI偏高、趋势走弱、
    持仓集中度高风险大，你要明确指出叙事与数据的矛盾。""",
    expected_output="空头论据报告，独立指出每只基金的风险点、止损条件和替代方案",
    agent=bearish_researcher,
    context=[task_constraint_veto, task_technical_analysis, task_fundamental_analysis,
             task_macro_analysis, task_sentiment_analysis],
    # 关键改进：不包含 task_bull_debate，空头独立分析
)

task_debate_judge = Task(
    description="""主持多空多轮辩论，最后做出最终裁决。

    【重要：这是新版多轮辩论模式，你必须主动委派】

    你已经在 context 里拿到：
    - 多头研究员的初论（task_bull_debate）
    - 空头研究员的初论（task_bear_debate）
    - 量化评分卡 + 硬约束否决结果

    你必须执行以下完整流程，不能跳过任何一步：

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【第1步 - 识别核心分歧点】
    阅读多空初论，对每只候选基金，找出多空之间的核心分歧（最多3个）。
    分歧点示例：
    - "多头说RSI=68尚未超买，空头说RSI=68已逼近警戒，谁对？"
    - "多头看好政策利好基本面，空头看空技术面破位，哪个更重要？"

    【第2步 - 委派多头进行二次回应】（必做）
    使用 "Delegate work to coworker" 工具，委派给"多头研究员"：
    - coworker: "多头研究员"
    - task: "我（裁决官）现在质询你以下几个空头反驳点，请针对性回应：
            1. [空头反驳点1，附数据]
            2. [空头反驳点2，附数据]
            对每个反驳点，请明确回答：(a)是否承认风险 (b)是否调整推荐力度
            (c)如果坚持原推荐请给出新数据"
    - context: "本次为基金XXXXXX的二轮辩论，初轮论据已在背景中..."

    【第3步 - 委派空头进行二次反驳】（必做）
    使用 "Delegate work to coworker" 工具，委派给"空头研究员"：
    - coworker: "空头研究员"
    - task: "我（裁决官）现在把多头的二次回应交给你，请二次反驳：
            [多头第2步回应的关键论据]
            请明确：(a)反驳是否成立 (b)如果多头数据论据强你是否撤回部分质疑
            (c)给出最终的风险评级（高/中/低）"

    【第4步 - 综合裁决】
    综合4轮信息，对每只基金给出最终裁决：
    - 裁决结果：通过 / 否决 / 降级
    - 裁决理由（必须引用量化数据 + 多空辩论关键回合）
    - 置信度：高 / 中 / 低
    - 如果降级：建议仓位和止损调整
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    【裁决原则（不变）】
    - 数据先于叙事
    - 硬约束已否决的基金：不讨论，直接排除
    - 量化总分 ≥ 7.0：多头需明显数据硬伤才能否决
    - 量化总分 6.5-7.0：多空对等权衡
    - 量化总分 < 6.5（通过硬约束）：空头合理论据即可否决
    - 叙事与数据矛盾时，采信数据方

    【输出格式】
    请用 markdown 输出，包含三部分：
    1. ## 多轮辩论交锋记录（按基金分组，每只基金列出第2轮、第3轮的关键问答）
    2. ## 最终裁决表（markdown 表格：基金代码 | 名称 | 裁决 | 量化分 | 理由摘要 | 置信度）
    3. ## 详细裁决说明（每只基金一段，引用辩论关键回合）""",
    expected_output="多轮辩论裁决报告（含交锋记录+裁决表+详细说明），对每只基金给出通过/否决/降级裁决",
    agent=debate_judge,
    context=[task_constraint_veto, task_bull_debate, task_bear_debate, task_technical_analysis],
)

# --- Layer 4 Tasks ---

task_risk_control = Task(
    description="""从激进、中性、保守三个风险偏好维度，评估辩论裁决后的投资方案。

    只对辩论裁决中"通过"或"降级"的基金进行风控评估。

    对每只基金分别给出：

    激进型投资者（可承受20%+回撤）：
    - 建议总仓位 / 单只基金最大仓位
    - 止损线
    - 杠杆建议（是否适合加杠杆）

    中性型投资者（可承受10-20%回撤）：
    - 建议总仓位 / 单只基金最大仓位
    - 止损线
    - 是否需要搭配债券基金

    保守型投资者（仅能承受<10%回撤）：
    - 建议总仓位 / 单只基金最大仓位
    - 止损线
    - 是否建议放弃波段操作

    对"降级"的基金：所有风险偏好下的仓位应比"通过"的低10-20%。
    
    同时给出整体方案的最大可能亏损和风险等级。""",
    expected_output="风控评估报告，含三种风险偏好下的仓位/止损/杠杆建议",
    agent=risk_controller,
    context=[task_debate_judge],
)

task_final_decision = Task(
    description="""综合所有层级分析结果，输出最终的波段投资方案。

    你必须输出以下内容：

    ## 推荐基金（3-5只）
    每只基金包含：
    - 基金名称 + 代码
    - 推荐原因（一句话总结，引用量化评分）
    - 入场区间（净值范围）
    - 目标收益（%）
    - 止损线（%）
    - 建议持有期
    - 风险等级
    - 辩论裁决结果（通过/降级）
    - 量化评分总分

    ## 仓位分配
    - 各基金建议权重
    - 现金留存比例

    ## 风险偏好适配
    - 激进/中性/保守三种方案

    ## 关键观察指标
    - 未来3-6个月需要持续跟踪的信号
    - 触发调仓的条件

    ## 免责声明
    - 本分析基于历史数据和AI推理，不构成投资建议
    - 基金投资有风险，过往业绩不代表未来表现

    诚实原则：
    1. 辩论裁决中被否决的基金，绝不强行推荐
    2. 量化评分偏低的基金，必须标注风险
    3. 如果当前市场环境不适合波段操作，直接建议观望
    4. 每只推荐基金必须记录到反馈日志（fund_code, fund_name, score, reason）""",
    expected_output="最终波段投资方案，含推荐基金清单、量化评分、仓位分配、风险适配和观察指标",
    agent=portfolio_manager,
    context=[task_screen_funds, task_macro_analysis, task_constraint_veto,
             task_technical_analysis, task_fundamental_analysis, task_sentiment_analysis,
             task_bull_debate, task_bear_debate, task_debate_judge, task_risk_control],
)

# ============================================================
# 9. 组建 Crew
# ============================================================

def _build_crew(fast_mode=False):
    """每次执行创建新的 Crew 实例，避免并发 kickoff 冲突

    Args:
        fast_mode: 快速模式 — 仅启用 5 个核心 Agent（screener/bull/bear/judge/portfolio_manager）
                   跳过宏观/技术/基本面/情绪/风控/否决官，节省约 50% token，缩短至 ~3-5 分钟
                   适合演示场景
    """
    if fast_mode:
        # 精简 5-Agent 流水线：筛选 → 多空辩论（judge 可委派）→ 决策
        # 重新构建依赖关系：bull/bear/judge 直接依赖 fund_screener
        fast_task_bull = Task(
            description=task_bull_debate.description,
            expected_output=task_bull_debate.expected_output,
            agent=bullish_researcher,
            context=[task_screen_funds],  # 仅依赖筛选层
        )
        fast_task_bear = Task(
            description=task_bear_debate.description,
            expected_output=task_bear_debate.expected_output,
            agent=bearish_researcher,
            context=[task_screen_funds],
        )
        fast_task_judge = Task(
            description=task_debate_judge.description,
            expected_output=task_debate_judge.expected_output,
            agent=debate_judge,
            context=[task_screen_funds, fast_task_bull, fast_task_bear],
        )
        fast_task_decision = Task(
            description=task_final_decision.description,
            expected_output=task_final_decision.expected_output,
            agent=portfolio_manager,
            context=[task_screen_funds, fast_task_judge],
        )
        return Crew(
            agents=[fund_screener, bullish_researcher, bearish_researcher,
                    debate_judge, portfolio_manager],
            tasks=[task_screen_funds, fast_task_bull, fast_task_bear,
                   fast_task_judge, fast_task_decision],
            process=Process.sequential,
            verbose=True,
            memory=False,
        )

    # 完整 11-Agent 流水线
    return Crew(
        agents=[
            fund_screener, macro_analyst,
            constraint_veto_agent,
            technical_analyst, fundamental_analyst, sentiment_analyst,
            bullish_researcher, bearish_researcher, debate_judge,
            risk_controller, portfolio_manager,
        ],
        tasks=[
            task_screen_funds, task_macro_analysis,
            task_constraint_veto,
            task_technical_analysis, task_fundamental_analysis, task_sentiment_analysis,
            task_bull_debate, task_bear_debate, task_debate_judge,
            task_risk_control, task_final_decision,
        ],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

# ============================================================
# 10. 可复用的执行函数（供 web_app.py 调用）
# ============================================================

def _log_recommendations_from_result(result_text: str):
    """
    从 Agent 最终输出文本中解析推荐基金，自动记录到反馈日志
    支持多种输出格式：基金代码(6位数字) + 基金名称 + 评分 + 推荐理由
    """
    import re
    from fund_data import fetch_fund_nav_history, calc_technical_indicators, quantitative_score

    logged_count = 0

    # 策略1：匹配"代码:XXXXXX"或"基金代码XXXXXX"或单独的6位数字+基金名
    # 常见格式：000051/华夏沪深300ETF联接A (7.55分)
    patterns = [
        r'(\d{6})[\/\s]+([^\(（\n]{2,30})[\s]*[\(（]\s*([\d.]+)\s*分',  # 000051/华夏沪深300 (7.55分)
        r'(\d{6})\s+([^\(（\n]{2,30})[\s]*[\(（]\s*评分[:：]?\s*([\d.]+)',  # 000051 华夏沪深300 (评分7.55)
        r'(\d{6})[\/\s]+([^\(（\n]{2,30})',  # 000051/华夏沪深300 (无评分)
    ]

    found_funds = []
    for pattern in patterns:
        matches = re.findall(pattern, result_text)
        for m in matches:
            code = m[0]
            name = m[1].strip()
            score = float(m[2]) if len(m) > 2 and m[2] else 0
            found_funds.append({"code": code, "name": name, "score": score})

    # 去重
    seen = set()
    unique_funds = []
    for f in found_funds:
        if f["code"] not in seen:
            seen.add(f["code"])
            unique_funds.append(f)

    if not unique_funds:
        print("  ℹ️ 未从输出中解析到推荐基金（格式不匹配），跳过自动记录", flush=True)
        return

    for fund in unique_funds:
        code = fund["code"]
        name = fund["name"]
        score = fund["score"]

        # 尝试获取各维度得分明细
        score_detail = {}
        try:
            nav_data = fetch_fund_nav_history(code, days=120)
            if nav_data is not None and len(nav_data) >= 30:
                indicators = calc_technical_indicators(nav_data)
                qs = quantitative_score(indicators)
                score_detail = qs.get("scores", {})
                if score == 0:
                    score = qs.get("weighted_total", 0)
        except:
            pass

        try:
            log_recommendation(
                fund_code=code,
                fund_name=name,
                score=score,
                reason=f"Agent v2 全流程推荐 (Layer 1→L1.5→L2→L3→L4)",
                target_return=10.0,
                stop_loss=-8.0,
                layer_passed="L1→L1.5→L2→L3→L4",
                score_detail=score_detail,
            )
            logged_count += 1
            print(f"  ✅ 已记录: {name}({code}) 评分{score}", flush=True)
        except Exception as e:
            print(f"  ❌ 记录失败 {code}: {e}", flush=True)

    print(f"  📊 共记录 {logged_count} 只推荐基金", flush=True)


def _parse_recommendation(text: str) -> dict:
    """从 Agent 结果文本中解析结构化投资建议
    
    尽力解析，解析失败时返回 safe defaults，不影响功能。
    """
    import re
    rec = {
        "action": "hold",
        "fund_name": "",
        "fund_code": "",
        "score": "",
        "reason": "",
        "target_return": "",
        "stop_loss": "",
        "position": "",
        "strategy": "",
        "raw_summary": text[:2000],
    }

    # 1. 判断推荐/不推荐/观望
    lower = text.lower()
    if "不建议买入" in text or "建议不买入" in text or "建议卖出" in text:
        rec["action"] = "avoid"
    elif "不推荐" in text or "无推荐" in text:
        rec["action"] = "avoid"
    elif "唯一推荐" in text or "🥇" in text or "建议买入" in text or "建议配置" in text:
        rec["action"] = "buy"
    elif "推荐" in text:
        rec["action"] = "buy"
    elif "观望" in text or "hold" in lower:
        rec["action"] = "hold"

    # 2. 提取基金名称和代码
    # 模式: "### 🥇 唯一推荐：163407 兴全沪深300增强(LOF)A"
    # 模式: "**基金代码** | 163407"
    m = re.search(r'(?:唯一推荐[：:]\s*|推荐基金[：:]\s*|🥇\s*)(\d{6})\s+(.+?)(?:\n|$)', text)
    if m:
        rec["fund_code"] = m.group(1).strip()
        rec["fund_name"] = m.group(2).strip()
    else:
        m = re.search(r'\*\*基金代码\*\*\s*\|\s*(\d{6})', text)
        if m:
            rec["fund_code"] = m.group(1).strip()
        m = re.search(r'\*\*基金名称\*\*\s*\|\s*(.+?)(?:\n|\|)', text)
        if m:
            rec["fund_name"] = m.group(1).strip()

    # 3. 提取评分
    m = re.search(r'量化评分[：:]\s*\*\*(\d+\.?\d*)\s*/\s*10\*\*', text)
    if not m:
        m = re.search(r'评分[：:]\s*\*\*(\d+\.?\d*)\s*/\s*10\*\*', text)
    if not m:
        m = re.search(r'(\d+\.?\d+)\s*/\s*10', text)
    if m:
        rec["score"] = m.group(1)

    # 4. 提取推荐原因（一句话）
    m = re.search(r'推荐原因[（(]一句话[)）][:：]?\s*\*\*(.+?)\*\*', text)
    if m:
        rec["reason"] = m.group(1).strip()
    else:
        # 找 "一句话" 后面的内容
        m = re.search(r'一句话[:：]?\s*\*\*(.+?)\*\*', text)
        if m:
            rec["reason"] = m.group(1).strip()
        else:
            # 更宽松：找 "推荐原因" 后面紧跟的加粗文本
            m = re.search(r'推荐原因.*\n\s*\*\*(.+?)\*\*', text)
            if m:
                rec["reason"] = m.group(1).strip()

    # 5. 提取目标收益和止损（兼容表格格式 **目标收益** | **+5%**）
    # 先尝试表格格式
    m = re.search(r'\*\*目标收益\*\*\s*\|\s*\*\*(.+?)\*\*', text)
    if m:
        rec["target_return"] = m.group(1).strip()
    else:
        m = re.search(r'目标收益[：:]\s*\*\*(.+?)\*\*', text)
        if m:
            rec["target_return"] = m.group(1).strip()

    m = re.search(r'\*\*止损线?\*\*\s*\|\s*\*\*(.+?)\*\*', text)
    if m:
        rec["stop_loss"] = m.group(1).strip()
    else:
        m = re.search(r'止损线?[：:]\s*\*\*(.+?)\*\*', text)
        if m:
            rec["stop_loss"] = m.group(1).strip()

    # 6. 提取仓位建议（兼容表格格式）
    m = re.search(r'\*\*建议仓位\*\*\s*\|\s*\*\*(\d+)%?\*\*', text)
    if m:
        rec["position"] = m.group(1) + "%"
    else:
        m = re.search(r'\*\*仓位\*\*\s*\|\s*\*\*(\d+)%?\*\*', text)
        if m:
            rec["position"] = m.group(1) + "%"
        else:
            m = re.search(r'仓位[：:]\s*\*\*(\d+)%?\*\*', text)
            if m:
                rec["position"] = m.group(1) + "%"
            else:
                m = re.search(r'建议仓位[：:]\s*(?:\|)?\s*(\d+)%?', text)
                if m:
                    rec["position"] = m.group(1) + "%"

    # 7. 提取策略
    m = re.search(r'\*\*首选入场方式\*\*\s*\|\s*\*\*(.+?)\*\*', text)
    if m:
        rec["strategy"] = m.group(1).strip()
    else:
        m = re.search(r'策略[：:]\s*\*\*(.+?)\*\*', text)
        if m:
            rec["strategy"] = m.group(1).strip()
        else:
            m = re.search(r'首选入场方式[：:]\s*(?:\|)?\s*(.+?)(?:\n|\|)', text)
            if m:
                rec["strategy"] = m.group(1).strip()

    # ==================== 组合方案兜底解析 ====================
    # 当 portfolio_manager 输出"基金组合方案"（多基金）时，上面的"唯一推荐"模式匹配不到
    # 兜底策略：找文档底部的"反馈日志记录"表格，取第一只基金
    if not rec["fund_name"] or not rec["fund_code"]:
        # 模式: "| 012414 | 永赢智能领先混合A | 7.25 | xxx |"
        # 反馈日志格式：基金代码 | 基金名称 | 评分 | 推荐理由
        feedback_match = re.search(
            r'反馈日志记录[\s\S]+?\|\s*(\d{6})\s*\|\s*([^\|]+?)\s*\|\s*(\d+\.?\d*)\s*\|\s*([^\|\n]+?)\s*(?:\||\n)',
            text
        )
        if feedback_match:
            rec["fund_code"] = feedback_match.group(1).strip()
            rec["fund_name"] = feedback_match.group(2).strip()
            if not rec["score"]:
                rec["score"] = feedback_match.group(3).strip()
            if not rec["reason"]:
                rec["reason"] = feedback_match.group(4).strip()[:120]
            rec["action"] = "buy"  # 有反馈日志说明有推荐
        else:
            # 二次兜底：找任意"| 6位代码 | 名称 | 数字评分 |"模式
            any_fund = re.search(
                r'\|\s*(\d{6})\s*\|\s*([^\|]+?)\s*\|\s*(\d+\.?\d*)\s*[\|/]',
                text
            )
            if any_fund:
                rec["fund_code"] = any_fund.group(1).strip()
                rec["fund_name"] = any_fund.group(2).strip()
                if not rec["score"]:
                    rec["score"] = any_fund.group(3).strip()
                rec["action"] = "buy"

    # 推理摘要兜底：如果还没找到 reason，截取"裁决"或"投资建议"段落首句
    if not rec["reason"]:
        m = re.search(r'(?:核心结论|投资建议|最终建议|裁决理由)[：:]\s*(.+?)(?:\n|。)', text)
        if m:
            rec["reason"] = m.group(1).strip()[:120]

    # 总仓位兜底：找"总仓位 65%"或"合计仓位"
    if not rec["position"]:
        m = re.search(r'(?:总仓位|合计仓位|组合仓位)[：:]?\s*(?:\*\*)?(\d+)%?', text)
        if m:
            rec["position"] = m.group(1) + "%（组合）"

    # action 兜底：如果走到这里 action 还是 hold，但找到了 fund_name，就改成 buy
    if rec["action"] == "hold" and rec["fund_name"]:
        rec["action"] = "buy"

    # 推理过程摘要：摘取"多轮辩论交锋记录"或"最终裁决"段落（前 800 字）作为可视化用
    debate_summary = ""
    m = re.search(r'(?:多轮辩论交锋记录|最终裁决|辩论裁决)[\s\S]{0,1500}', text)
    if m:
        debate_summary = m.group(0)[:800]
    rec["debate_summary"] = debate_summary

    return rec


def run_agent(fund_codes=None, progress_callback=None, fast_mode=False):
    """运行完整 Agent 流程，返回结果文本

    Args:
        fund_codes: 可选，指定基金代码（逗号分隔，如 "000051,000071"）
                    指定时跳过全市场筛选，所有 Agent 围绕指定基金做分析和辩论
        progress_callback: 可选，进度回调函数，签名: cb(type, agent_id, agent_name, layer, message, output_preview="")
        fast_mode: 快速模式（demo），仅 5 个核心 Agent，~3-5 分钟，token 减半
    """
    def _cb(event_type, agent_id, agent_name, layer, message, output_preview=""):
        if progress_callback:
            try:
                progress_callback(event_type, agent_id, agent_name, layer, message, output_preview)
            except Exception:
                pass

    # 保存原始 Task 描述，确保多次调用不会互相污染
    _orig_task_desc = task_screen_funds.description
    _orig_task_output = task_screen_funds.expected_output
    _orig_screener_goal = fund_screener.goal

    try:
        # 如果指定了基金代码，动态修改 Task 描述
        if fund_codes:
            codes_list = [c.strip() for c in fund_codes.split(",") if c.strip()]
            codes_str = "、".join(codes_list)
            print(f"  🎯 指定基金模式: {codes_str}", flush=True)

            # 修改筛选师 Task：不再是"从全市场筛选"，而是"获取指定基金的详细数据"
            task_screen_funds.description = f"""对以下指定基金进行深度数据采集：{codes_str}

        请调用 fund_data.fetch_fund_details() 逐一获取每只基金的详细信息，包括：
        - 基金名称、类型、成立日期
        - 近1月/3月/6月/1年/3年收益率
        - 当前净值、累计净值
        - 基金经理、管理规模
        - 持仓前10大重仓股

        同时调用 fund_data.get_top_funds_report() 获取同类基金对比数据，
        以便后续分析时可以与同类基金比较。

        输出格式：每只基金的详细数据卡片 + 同类基金对比摘要。"""

            task_screen_funds.expected_output = f"指定基金 {codes_str} 的详细数据报告 + 同类基金对比"

            # 修改筛选师 Agent 的 goal
            fund_screener.goal = f"获取指定基金 {codes_str} 的完整数据和同类对比信息"

        print("=" * 70, flush=True)
        print("  Fund Trade Agent v2 — 基金波段投资多智能体分析系统", flush=True)
        print("  架构灵感: TradingAgents (GitHub 71.4k Star)", flush=True)
        print("  数据源: AKShare / 天天基金网 (覆盖支付宝全部基金)", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        print("  v2 核心改进:", flush=True)
        print("    ✅ 辩论层并行独立（消除框架效应）", flush=True)
        print("    ✅ 硬约束否决层（RSI>70/MA空头/波动率>35% 一票否决）", flush=True)
        print("    ✅ 量化评分卡（6维度加权替代LLM主观打分）", flush=True)
        print("    ✅ 反馈环（记录→追踪→复盘→经验注入）", flush=True)
        print(flush=True)
        print("  Agent 团队 (11 Agents, 5 Layers):", flush=True)
        print(flush=True)
        print("  Layer 1 — Data Screening:", flush=True)
        print("    1. 基金数据筛选师", flush=True)
        print("    2. 宏观经济分析师", flush=True)
        print(flush=True)
        print("  Layer 1.5 — Hard Constraint Veto [NEW]:", flush=True)
        print("    3. 硬约束否决官", flush=True)
        print(flush=True)
        print("  Layer 2 — Multi-dimensional Analysis:", flush=True)
        print("    4. 技术分析专家", flush=True)
        print("    5. 基本面分析专家", flush=True)
        print("    6. 市场情绪分析师", flush=True)
        print(flush=True)
        print("  Layer 3 — Parallel Debate + Judge [REDESIGNED]:", flush=True)
        print("    7. 多头研究员 (独立)", flush=True)
        print("    8. 空头研究员 (独立)", flush=True)
        print("    9. 辩论裁决官 [NEW]", flush=True)
        print(flush=True)
        print("  Layer 4 — Decision & Risk Control:", flush=True)
        print("    10. 风控专家", flush=True)
        print("    11. 投资组合经理 (含经验注入)", flush=True)
        print(flush=True)

        # 发送预加载进度事件 — 让用户在 kickoff 前有视觉反馈
        _cb("progress", "system", "系统初始化", 0, "🚀 正在初始化 11 个 Agent...")
        _cb("progress", "system", "系统初始化", 0, "📚 正在加载历史经验...")
        # 显示历史经验
        summary = get_recommendation_summary()
        if summary["total"] > 0:
            _cb("progress", "system", "系统初始化", 0,
                f"📊 历史推荐: {summary['total']}条, 已追踪: {summary['tracked']}条, 胜率: {summary.get('win_rate', 'N/A')}%")
        else:
            _cb("progress", "system", "系统初始化", 0, "📊 历史推荐: 暂无（首次运行）")

        # 自动追踪历史推荐结果
        if summary["total"] > 0 and summary["tracked"] < summary["total"]:
            _cb("progress", "system", "系统初始化", 0,
                f"🔄 自动追踪历史推荐结果（{summary['total'] - summary['tracked']}条待追踪）...")
            review_result = auto_track_and_review()
            tracked = review_result["tracked_count"]
            if tracked > 0:
                opt = review_result["optimization"]
                if opt.get("optimized"):
                    _cb("progress", "system", "系统初始化", 0,
                        f"🔧 权重已优化: {opt.get('reason', '')}")
                else:
                    _cb("progress", "system", "系统初始化", 0,
                        f"ℹ️ 权重暂不调整: {opt.get('reason', '无需调整')}")

        _cb("progress", "system", "系统初始化", 0, "🎯 正在准备分析任务...")
        _cb("progress", "system", "系统初始化", 0, "⚡ 即将启动 CrewAI 多智能体协作...")

        print("=" * 70, flush=True)
        print("  Fund Trade Agent v2 — 基金波段投资多智能体分析系统", flush=True)
        print("  架构灵感: TradingAgents (GitHub 71.4k Star)", flush=True)
        print("  数据源: AKShare / 天天基金网 (覆盖支付宝全部基金)", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        print("  v2 核心改进:", flush=True)
        print("    ✅ 辩论层并行独立（消除框架效应）", flush=True)
        print("    ✅ 硬约束否决层（RSI>70/MA空头/波动率>35% 一票否决）", flush=True)
        print("    ✅ 量化评分卡（6维度加权替代LLM主观打分）", flush=True)
        print("    ✅ 反馈环（记录→追踪→复盘→经验注入）", flush=True)
        print(flush=True)
        print("  Agent 团队 (11 Agents, 5 Layers):", flush=True)
        print(flush=True)
        print("  Layer 1 — Data Screening:", flush=True)
        print("    1. 基金数据筛选师", flush=True)
        print("    2. 宏观经济分析师", flush=True)
        print(flush=True)
        print("  Layer 1.5 — Hard Constraint Veto [NEW]:", flush=True)
        print("    3. 硬约束否决官", flush=True)
        print(flush=True)
        print("  Layer 2 — Multi-dimensional Analysis:", flush=True)
        print("    4. 技术分析专家", flush=True)
        print("    5. 基本面分析专家", flush=True)
        print("    6. 市场情绪分析师", flush=True)
        print(flush=True)
        print("  Layer 3 — Parallel Debate + Judge [REDESIGNED]:", flush=True)
        print("    7. 多头研究员 (独立)", flush=True)
        print("    8. 空头研究员 (独立)", flush=True)
        print("    9. 辩论裁决官 [NEW]", flush=True)
        print(flush=True)
        print("  Layer 4 — Decision & Risk Control:", flush=True)
        print("    10. 风控专家", flush=True)
        print("    11. 投资组合经理 (含经验注入)", flush=True)
        print(flush=True)

        print("-" * 70, flush=True)
        print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("-" * 70, flush=True)
        print(flush=True)

        # 发送 Layer 1 开始事件
        _cb("layer_start", "fund_screener", "基金数据筛选师", 1, "📊 Layer 1 数据采集启动 — 基金筛选师 + 宏观分析师并行工作")
        _cb("agent_start", "fund_screener", "基金数据筛选师", 1, "🔄 基金数据筛选师 开始分析...")
        _cb("agent_start", "macro_analyst", "宏观经济分析师", 1, "🔄 宏观经济分析师 开始分析...")

        print("  正在启动 Crew kickoff...", flush=True)
        crew = _build_crew(fast_mode=fast_mode)
        if fast_mode:
            _cb("progress", "system", "系统初始化", 0, "⚡ 快速模式：5 Agent 精简流水线（筛选→多空辩论→裁决→决策）")
        result = crew.kickoff()

        # Crew 完成后，提取每个 Agent 的输出
        _cb("layer_start", "system", "系统", 0, "📋 分析完成，提取各 Agent 输出...")

        # 映射 task 到 agent
        if fast_mode:
            task_agent_map = [
                ("fund_screener", "基金数据筛选师", 1),
                ("bullish_researcher", "多头研究员", 3),
                ("bearish_researcher", "空头研究员", 3),
                ("debate_judge", "辩论裁决官", 3),
                ("portfolio_manager", "投资组合经理", 4),
            ]
        else:
            task_agent_map = [
                ("fund_screener", "基金数据筛选师", 1),
                ("macro_analyst", "宏观经济分析师", 1),
                ("constraint_veto", "硬约束否决官", 1.5),
                ("technical_analyst", "技术分析专家", 2),
                ("fundamental_analyst", "基本面分析专家", 2),
                ("sentiment_analyst", "市场情绪分析师", 2),
                ("bullish_researcher", "多头研究员", 3),
                ("bearish_researcher", "空头研究员", 3),
                ("debate_judge", "辩论裁决官", 3),
                ("risk_controller", "风控专家", 4),
                ("portfolio_manager", "投资组合经理", 4),
            ]

        for i, (aid, aname, layer) in enumerate(task_agent_map):
            if i < len(crew.tasks):
                task_output = crew.tasks[i].output
                if task_output:
                    output_str = str(task_output)[:2000]
                    _cb("agent_done", aid, aname, layer, f"✅ {aname} 分析完成", output_str)
                    # 辩论层特殊处理
                    if aid == "bullish_researcher":
                        _cb("debate_bull", aid, aname, layer, f"🐂 多头论点:\n{output_str[:800]}")
                    elif aid == "bearish_researcher":
                        _cb("debate_bear", aid, aname, layer, f"🐻 空头论点:\n{output_str[:800]}")
                    elif aid == "debate_judge":
                        _cb("judge", aid, aname, layer, f"⚖️ 裁决结论:\n{output_str[:800]}")

        # 解析最终 recommendation，推送结构化结果
        result_str = str(result)
        rec = _parse_recommendation(result_str)
        rec_json = json.dumps(rec, ensure_ascii=False)
        _cb("recommendation", "system", "投资组合经理", 4,
            f"🎯 {'推荐' if rec.get('action') == 'buy' else ('观望' if rec.get('action') == 'hold' else '不推荐')} — {rec.get('fund_name', '未知')}",
            rec_json)

        print()
        print("=" * 70)
        print("  最终波段投资方案 (v2)")
        print("=" * 70)
        print(result)

        # 保存结果
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, "fund_trade_report_v2.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 基金波段投资分析报告 v2\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## v2 改进要点\n\n")
            f.write("- ✅ 辩论层并行独立（消除框架效应）\n")
            f.write("- ✅ 硬约束否决层（RSI>70/MA空头/波动率>35% 一票否决）\n")
            f.write("- ✅ 量化评分卡（6维度加权替代LLM主观打分）\n")
            f.write("- ✅ 反馈环（记录→追踪→复盘→经验注入）\n\n")
            f.write("---\n\n")
            f.write(str(result))
        print(f"\n报告已保存至: {output_file}")

        # 自动记录推荐到反馈日志
        print("\n📝 正在记录推荐到反馈日志...", flush=True)
        _log_recommendations_from_result(str(result))

        return str(result)
    finally:
        # 恢复原始 Task 描述，防止多次调用时互相污染
        task_screen_funds.description = _orig_task_desc
        task_screen_funds.expected_output = _orig_task_output
        fund_screener.goal = _orig_screener_goal


# ============================================================
# 11. 主入口
# ============================================================

if __name__ == "__main__":
    run_agent()
