"""
基金数据采集工具模块
使用 AKShare 从天天基金网（东方财富）获取中国开放式基金数据
覆盖支付宝/天天基金等平台可购买的全部基金（含 QDII 海外基金）

数据源说明：
- fund_open_fund_rank_em: 基金排行（含各周期收益率）
- fund_open_fund_info_em: 单只基金历史净值
- fund_portfolio_hold_em: 基金持仓
- fund_individual_basic_info_xq: 基金基本信息
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def fetch_fund_ranking(fund_type: str = "全部", top_n: int = 200) -> pd.DataFrame:
    """
    获取基金排行数据
    fund_type: 全部/股票型/混合型/债券型/指数型/QDII/FOF/另类投资
    top_n: 返回前 N 只
    """
    df = ak.fund_open_fund_rank_em(symbol=fund_type)
    # 清理数据：去掉增长率中的空值
    numeric_cols = ["日增长率", "近1周", "近1月", "近3月", "近6月", "近1年", "近2年", "近3年", "今年来", "成立来"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.head(top_n)


def screen_funds(
    fund_types: list = None,
    min_3m_return: float = None,
    max_3m_return: float = None,
    min_1y_return: float = None,
    min_scale: float = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    多参数筛选基金
    fund_types: 基金类型列表，如 ["股票型", "混合型", "QDII"]
    min_3m_return: 近3月最低收益率(%)
    max_3m_return: 近3月最高收益率(%)，用于排除极端值
    min_1y_return: 近1年最低收益率(%)
    min_scale: 最小规模(亿)，通过基金简称或代码间接筛选
    top_n: 最终返回数量
    """
    if fund_types is None:
        fund_types = ["股票型", "混合型", "指数型"]

    all_funds = []
    for ft in fund_types:
        try:
            df = fetch_fund_ranking(fund_type=ft, top_n=500)
            df["基金类型"] = ft
            all_funds.append(df)
        except Exception as e:
            print(f"获取{ft}基金失败: {e}")

    if not all_funds:
        return pd.DataFrame()

    result = pd.concat(all_funds, ignore_index=True)

    # 过滤条件
    if min_3m_return is not None:
        result = result[result["近3月"] >= min_3m_return]
    if max_3m_return is not None:
        result = result[result["近3月"] <= max_3m_return]
    if min_1y_return is not None:
        result = result[result["近1年"] >= min_1y_return]

    # 按近3月收益率排序
    result = result.sort_values("近3月", ascending=False)

    return result.head(top_n)


def fetch_fund_nav_history(fund_code: str, days: int = 180) -> pd.DataFrame:
    """
    获取基金历史净值
    fund_code: 基金代码
    days: 回溯天数
    """
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        # 确保净值日期列是字符串格式
        df["净值日期"] = pd.to_datetime(df["净值日期"]).dt.strftime("%Y-%m-%d")
        df = df[df["净值日期"] >= cutoff]
        return df
    except Exception as e:
        print(f"获取{fund_code}净值失败: {e}")
        return pd.DataFrame()


def calc_technical_indicators(df_nav: pd.DataFrame) -> dict:
    """
    计算技术指标：MA、RSI、MACD、波动率、最大回撤
    """
    if df_nav.empty or len(df_nav) < 20:
        return {"error": "数据不足"}

    nav = df_nav["单位净值"].astype(float)

    # 移动平均线
    ma5 = nav.rolling(5).mean().iloc[-1] if len(nav) >= 5 else None
    ma10 = nav.rolling(10).mean().iloc[-1] if len(nav) >= 10 else None
    ma20 = nav.rolling(20).mean().iloc[-1] if len(nav) >= 20 else None
    ma60 = nav.rolling(60).mean().iloc[-1] if len(nav) >= 60 else None

    # RSI (14日)
    delta = nav.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1] if not rsi.empty else None

    # MACD
    ema12 = nav.ewm(span=12, adjust=False).mean()
    ema26 = nav.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    macd_current = macd_hist.iloc[-1] if not macd_hist.empty else None

    # 日收益率
    daily_return = nav.pct_change().dropna()

    # 波动率（年化）
    volatility = daily_return.std() * np.sqrt(252) * 100 if len(daily_return) > 1 else None

    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else None

    # 夏普比率（简化，假设无风险利率2%）
    if volatility and volatility > 0:
        avg_return = daily_return.mean() * 252 * 100
        sharpe = (avg_return - 2) / volatility
    else:
        sharpe = None

    # 当前趋势判断
    current_nav = nav.iloc[-1]
    trend = "unknown"
    if ma5 and ma10 and ma20:
        if current_nav > ma5 > ma10 > ma20:
            trend = "strong_up"
        elif current_nav > ma5 and ma5 > ma10:
            trend = "up"
        elif current_nav < ma5 < ma10 < ma20:
            trend = "strong_down"
        elif current_nav < ma5 and ma5 < ma10:
            trend = "down"
        else:
            trend = "sideways"

    return {
        "current_nav": round(current_nav, 4),
        "MA5": round(ma5, 4) if ma5 else None,
        "MA10": round(ma10, 4) if ma10 else None,
        "MA20": round(ma20, 4) if ma20 else None,
        "MA60": round(ma60, 4) if ma60 else None,
        "RSI14": round(rsi_current, 2) if rsi_current and not np.isnan(rsi_current) else None,
        "MACD": round(macd_current, 4) if macd_current and not np.isnan(macd_current) else None,
        "volatility_annual": round(volatility, 2) if volatility else None,
        "max_drawdown": round(max_drawdown, 2) if max_drawdown else None,
        "sharpe_ratio": round(sharpe, 2) if sharpe else None,
        "trend": trend,
        "data_points": len(nav),
    }


def fetch_fund_details(fund_code: str) -> dict:
    """
    获取基金基本信息
    """
    try:
        df = ak.fund_individual_basic_info_xq(symbol=fund_code)
        info = dict(zip(df["item"], df["value"]))
        return info
    except Exception as e:
        return {"error": str(e)}


def fetch_fund_holdings(fund_code: str, year: str = "2024") -> pd.DataFrame:
    """
    获取基金持仓
    """
    try:
        df = ak.fund_portfolio_hold_em(symbol=fund_code, date=year)
        return df
    except Exception as e:
        return pd.DataFrame()


def get_top_funds_report(
    fund_types: list = None,
    top_n: int = 30,
    min_3m: float = 5.0,
) -> str:
    """
    一键生成基金筛选报告（供 Agent 调用）
    返回格式化文本，包含筛选后的基金列表及关键指标
    """
    df = screen_funds(
        fund_types=fund_types,
        min_3m_return=min_3m,
        top_n=top_n,
    )

    if df.empty:
        return "未找到符合条件的基金"

    report_lines = []
    report_lines.append(f"基金筛选报告（共{len(df)}只，近3月收益率>{min_3m}%）\n")
    report_lines.append("=" * 80)

    for _, row in df.iterrows():
        line = (
            f"\n基金代码: {row['基金代码']}  |  基金名称: {row['基金简称']}  |  类型: {row.get('基金类型', 'N/A')}\n"
            f"  单位净值: {row['单位净值']}  |  累计净值: {row['累计净值']}\n"
            f"  近1周: {row.get('近1周', 'N/A')}%  |  近1月: {row.get('近1月', 'N/A')}%  |  "
            f"近3月: {row.get('近3月', 'N/A')}%  |  近6月: {row.get('近6月', 'N/A')}%\n"
            f"  近1年: {row.get('近1年', 'N/A')}%  |  今年来: {row.get('今年来', 'N/A')}%  |  "
            f"成立来: {row.get('成立来', 'N/A')}%\n"
            f"  手续费: {row.get('手续费', 'N/A')}\n"
        )
        report_lines.append(line)

    return "\n".join(report_lines)


def get_fund_technical_analysis(fund_code: str, fund_name: str = "", days: int = 180) -> str:
    """
    获取单只基金的技术分析报告（供 Agent 调用）
    """
    df_nav = fetch_fund_nav_history(fund_code, days=days)
    indicators = calc_technical_indicators(df_nav)

    if "error" in indicators:
        return f"基金 {fund_code}({fund_name}) 技术分析失败: {indicators['error']}"

    trend_map = {
        "strong_up": "强势上涨（多头排列）",
        "up": "偏多趋势",
        "sideways": "横盘震荡",
        "down": "偏空趋势",
        "strong_down": "强势下跌（空头排列）",
        "unknown": "趋势不明",
    }

    rsi_signal = "超买" if indicators.get("RSI14") and indicators["RSI14"] > 70 else \
                 "超卖" if indicators.get("RSI14") and indicators["RSI14"] < 30 else "正常"

    report = f"""基金 {fund_code}({fund_name}) 技术分析报告
{'=' * 50}
当前净值: {indicators['current_nav']}
趋势判断: {trend_map.get(indicators['trend'], indicators['trend'])}

移动平均线:
  MA5:  {indicators.get('MA5', 'N/A')}
  MA10: {indicators.get('MA10', 'N/A')}
  MA20: {indicators.get('MA20', 'N/A')}
  MA60: {indicators.get('MA60', 'N/A')}

技术指标:
  RSI(14): {indicators.get('RSI14', 'N/A')} ({rsi_signal})
  MACD:    {indicators.get('MACD', 'N/A')}

风险指标:
  年化波动率: {indicators.get('volatility_annual', 'N/A')}%
  最大回撤:   {indicators.get('max_drawdown', 'N/A')}%
  夏普比率:   {indicators.get('sharpe_ratio', 'N/A')}

数据点数: {indicators.get('data_points', 'N/A')}
"""
    return report


def check_hard_constraints(indicators: dict) -> dict:
    """
    硬约束一票否决检查
    三条红线，触犯任一即否决：
    - RSI(14) > 70 → 超买否决
    - MA 空头排列 → 趋势否决
    - 年化波动率 > 35% → 风险否决
    """
    vetoes = []

    rsi = indicators.get("RSI14")
    if rsi is not None and not (isinstance(rsi, float) and np.isnan(rsi)):
        if rsi > 70:
            vetoes.append(f"RSI={rsi:.1f}>70 超买否决")

    trend = indicators.get("trend")
    if trend in ("strong_down", "down"):
        vetoes.append(f"趋势={trend} 空头排列否决")

    vol = indicators.get("volatility_annual")
    if vol is not None and vol > 35:
        vetoes.append(f"波动率={vol:.1f}%>35% 风险否决")

    return {
        "passed": len(vetoes) == 0,
        "vetoes": vetoes,
    }


# 默认评分权重
_DEFAULT_WEIGHTS = {
    "trend": 0.30, "rsi": 0.20, "macd": 0.15,
    "volatility": 0.15, "sharpe": 0.10, "drawdown": 0.10,
}

_SCORING_WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring_weights.json")


def _load_scoring_weights() -> dict:
    """从 scoring_weights.json 读取权重（由 feedback.py 优化后写入）"""
    if os.path.exists(_SCORING_WEIGHTS_FILE):
        try:
            import json as _json
            with open(_SCORING_WEIGHTS_FILE, "r", encoding="utf-8") as f:
                data = _json.load(f)
                return {k: data.get(k, v) for k, v in _DEFAULT_WEIGHTS.items()}
        except:
            pass
    return dict(_DEFAULT_WEIGHTS)


def quantitative_score(indicators: dict) -> dict:
    """
    量化评分卡 — 替代 LLM 主观打分
    6 个可计算维度加权评分，总分 0-10

    维度与权重：
      趋势(30%)  RSI(20%)  MACD(15%)  波动率(15%)  夏普(10%)  回撤(10%)

    通过条件：加权总分 ≥ 6.5 且硬约束全部通过
    """
    if "error" in indicators:
        return {"total": 0, "error": indicators["error"], "recommend": False}

    scores = {}

    # 1. 趋势得分 (30%)
    trend = indicators.get("trend", "unknown")
    trend_map = {"strong_up": 10, "up": 7, "sideways": 5, "down": 3, "strong_down": 0, "unknown": 5}
    scores["trend"] = trend_map.get(trend, 5)

    # 2. RSI 得分 (20%)
    rsi = indicators.get("RSI14")
    if rsi is not None and not (isinstance(rsi, float) and np.isnan(rsi)):
        if 40 <= rsi <= 60:
            scores["rsi"] = 10
        elif 30 <= rsi <= 70:
            scores["rsi"] = 7
        elif rsi > 70:
            scores["rsi"] = 2
        else:
            scores["rsi"] = 5
    else:
        scores["rsi"] = 5

    # 3. MACD 得分 (15%)
    macd = indicators.get("MACD")
    if macd is not None and not (isinstance(macd, float) and np.isnan(macd)):
        scores["macd"] = 8 if macd > 0 else 3
    else:
        scores["macd"] = 5

    # 4. 波动率得分 (15%)
    vol = indicators.get("volatility_annual")
    if vol is not None:
        if vol < 15:
            scores["volatility"] = 8
        elif vol < 25:
            scores["volatility"] = 7
        elif vol < 35:
            scores["volatility"] = 5
        else:
            scores["volatility"] = 2
    else:
        scores["volatility"] = 5

    # 5. 夏普得分 (10%)
    sharpe = indicators.get("sharpe_ratio")
    if sharpe is not None:
        if sharpe > 1:
            scores["sharpe"] = 10
        elif sharpe > 0.5:
            scores["sharpe"] = 7
        elif sharpe > 0:
            scores["sharpe"] = 5
        else:
            scores["sharpe"] = 2
    else:
        scores["sharpe"] = 5

    # 6. 回撤得分 (10%)
    max_dd = indicators.get("max_drawdown")
    if max_dd is not None:
        if max_dd > -5:
            scores["drawdown"] = 10
        elif max_dd > -10:
            scores["drawdown"] = 8
        elif max_dd > -20:
            scores["drawdown"] = 5
        else:
            scores["drawdown"] = 2
    else:
        scores["drawdown"] = 5

    weights = _load_scoring_weights()
    total = sum(scores[k] * weights[k] for k in weights)

    constraint = check_hard_constraints(indicators)

    return {
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "weights": weights,
        "weighted_total": round(total, 1),
        "hard_constraints": constraint,
        "recommend": total >= 6.5 and constraint["passed"],
    }


def get_quantitative_score_report(fund_code: str, fund_name: str = "", days: int = 180) -> str:
    """获取基金量化评分报告（供 Agent / CrewAI Tool 调用）"""
    df_nav = fetch_fund_nav_history(fund_code, days=days)
    indicators = calc_technical_indicators(df_nav)

    if "error" in indicators:
        return f"基金 {fund_code}({fund_name}) 量化评分失败: {indicators['error']}"

    score = quantitative_score(indicators)
    constraint = score["hard_constraints"]

    return f"""基金 {fund_code}({fund_name}) 量化评分报告
{'=' * 50}
加权总分: {score['weighted_total']}/10
是否推荐: {'✅ 通过' if score['recommend'] else '❌ 不推荐'}

各维度评分:
  趋势(30%): {score['scores']['trend']}/10
  RSI(20%):   {score['scores']['rsi']}/10
  MACD(15%):  {score['scores']['macd']}/10
  波动率(15%): {score['scores']['volatility']}/10
  夏普(10%):  {score['scores']['sharpe']}/10
  回撤(10%):  {score['scores']['drawdown']}/10

硬约束检查:
  {'✅ 全部通过' if constraint['passed'] else '❌ ' + '; '.join(constraint['vetoes'])}

底层指标:
  RSI(14): {indicators.get('RSI14', 'N/A')}
  趋势: {indicators.get('trend', 'N/A')}
  波动率: {indicators.get('volatility_annual', 'N/A')}%
  夏普: {indicators.get('sharpe_ratio', 'N/A')}
  最大回撤: {indicators.get('max_drawdown', 'N/A')}%
"""


def get_hard_constraints_report(fund_code: str, fund_name: str = "", days: int = 180) -> str:
    """获取硬约束否决检查报告（供 Agent / CrewAI Tool 调用）"""
    df_nav = fetch_fund_nav_history(fund_code, days=days)
    indicators = calc_technical_indicators(df_nav)

    if "error" in indicators:
        return f"基金 {fund_code}({fund_name}) 数据不足，无法检查"

    constraint = check_hard_constraints(indicators)
    status = "✅ 通过" if constraint["passed"] else "❌ 否决"
    veto_text = "; ".join(constraint["vetoes"]) if constraint["vetoes"] else "无"

    return f"""基金 {fund_code}({fund_name}) 硬约束检查
状态: {status}
否决原因: {veto_text}
RSI(14): {indicators.get('RSI14', 'N/A')}
趋势: {indicators.get('trend', 'N/A')}
波动率: {indicators.get('volatility_annual', 'N/A')}%
"""


if __name__ == "__main__":
    # 快速测试
    print("1. 基金筛选报告...")
    report = get_top_funds_report(fund_types=["股票型", "混合型"], top_n=10, min_3m=10.0)
    print(report[:1000])

    print("\n2. 技术分析报告...")
    tech = get_fund_technical_analysis("016370", "信澳业绩驱动混合A")
    print(tech)

    print("\n3. 量化评分报告...")
    qs = get_quantitative_score_report("016370", "信澳业绩驱动混合A")
    print(qs)

    print("\n4. 硬约束检查...")
    hc = get_hard_constraints_report("161725", "招商中证白酒指数A")
    print(hc)
