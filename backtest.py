"""
回测模块 — 验证 Agent 推荐的有效性
核心逻辑：用历史数据模拟 Agent 推荐场景，计算实际收益并与基准对比

使用方式：
  python backtest.py                          # 默认回测（3个月前买入，持有60天）
  python backtest.py --days-ago 90 --hold 60  # 自定义参数
  python backtest.py --fund 003834            # 回测单只基金
"""

import argparse
import json
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd


def fetch_nav_up_to(fund_code: str, end_date: str, days: int = 365) -> pd.DataFrame:
    """
    获取基金截至 end_date 的历史净值
    用于模拟"在 end_date 当天做决策"的场景
    """
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        df["净值日期"] = pd.to_datetime(df["净值日期"])
        # 只取 end_date 之前的数据（模拟当时只能看到这些数据）
        df = df[df["净值日期"] <= end_date]
        start = (pd.Timestamp(end_date) - timedelta(days=days)).strftime("%Y-%m-%d")
        df = df[df["净值日期"] >= start]
        return df.sort_values("净值日期")
    except Exception as e:
        print(f"获取 {fund_code} 净值失败: {e}")
        return pd.DataFrame()


def calc_indicators_at_date(df_nav: pd.DataFrame) -> dict:
    """
    在给定数据窗口内计算技术指标（模拟当时的分析结果）
    """
    if df_nav.empty or len(df_nav) < 20:
        return {"error": "数据不足"}

    nav = df_nav["单位净值"].astype(float)

    ma5 = nav.rolling(5).mean().iloc[-1] if len(nav) >= 5 else None
    ma10 = nav.rolling(10).mean().iloc[-1] if len(nav) >= 10 else None
    ma20 = nav.rolling(20).mean().iloc[-1] if len(nav) >= 20 else None
    ma60 = nav.rolling(60).mean().iloc[-1] if len(nav) >= 60 else None

    # RSI
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

    # 趋势
    current_nav = nav.iloc[-1]
    if ma5 and ma10 and ma20:
        if current_nav > ma5 > ma10 > ma20:
            trend = "strong_up"
        elif current_nav > ma5 and ma5 > ma10:
            trend = "up"
        elif current_nav < ma5 < ma10 < ma20:
            trend = "strong_down"
        else:
            trend = "sideways"
    else:
        trend = "unknown"

    # 波动率
    daily_return = nav.pct_change().dropna()
    volatility = daily_return.std() * np.sqrt(252) * 100 if len(daily_return) > 1 else None

    # 夏普
    sharpe = None
    if volatility and volatility > 0:
        avg_return = daily_return.mean() * 252 * 100
        sharpe = (avg_return - 2) / volatility

    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else None

    return {
        "current_nav": round(current_nav, 4),
        "MA5": round(ma5, 4) if ma5 else None,
        "MA10": round(ma10, 4) if ma10 else None,
        "MA20": round(ma20, 4) if ma20 else None,
        "MA60": round(ma60, 4) if ma60 else None,
        "RSI14": round(rsi_current, 2) if rsi_current and not np.isnan(rsi_current) else None,
        "MACD": round(macd_current, 4) if macd_current and not np.isnan(macd_current) else None,
        "trend": trend,
        "volatility": round(volatility, 2) if volatility else None,
        "max_drawdown": round(max_drawdown, 2) if max_drawdown else None,
        "sharpe": round(sharpe, 2) if sharpe else None,
    }


def agent_like_score(indicators: dict) -> float:
    """
    模拟 Agent 的打分逻辑（技术面评分）
    基于 Agent prompt 中的判断标准：MA趋势、RSI区间、MACD方向、波动率、夏普
    满分 10 分
    """
    score = 5.0  # 基础分

    # 趋势加分（+2 / +1 / -1 / -2）
    trend = indicators.get("trend", "unknown")
    trend_scores = {"strong_up": 2.0, "up": 1.0, "sideways": 0, "down": -1.0, "strong_down": -2.0}
    score += trend_scores.get(trend, 0)

    # RSI 加分（正常区 +1，超买 -1，超卖 +0.5）
    rsi = indicators.get("RSI14")
    if rsi:
        if 30 <= rsi <= 70:
            score += 1.0
        elif rsi > 70:
            score -= 1.0
        elif rsi < 30:
            score += 0.5

    # MACD 加分（金叉/正向 +1，死叉/负向 -1）
    macd = indicators.get("MACD")
    if macd:
        if macd > 0:
            score += 1.0
        else:
            score -= 1.0

    # 波动率扣分（>25% 扣分）
    vol = indicators.get("volatility")
    if vol:
        if vol > 30:
            score -= 0.5
        elif vol < 15:
            score += 0.5

    # 夏普加分（>1 加分）
    sharpe = indicators.get("sharpe")
    if sharpe:
        if sharpe > 1:
            score += 1.0
        elif sharpe < 0.5:
            score -= 0.5

    return max(0, min(10, round(score, 1)))


def backtest_fund(
    fund_code: str,
    fund_name: str,
    days_ago: int = 90,
    hold_days: int = 60,
    stop_loss_pct: float = -8.0,
    take_profit_pct: float = 15.0,
) -> dict:
    """
    回测单只基金：
    1. 取 days_ago 前的数据，计算 Agent 当时会看到的指标
    2. 模拟 Agent 打分，判断是否推荐
    3. 按 Agent 建议买入，持有 hold_days 天
    4. 计算实际收益（含止损/止盈触发）

    返回完整的回测结果
    """
    decision_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    exit_date = (datetime.now() - timedelta(days=days_ago - hold_days)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. 获取决策日的数据（模拟 Agent 当时的信息）
    df_decision = fetch_nav_up_to(fund_code, decision_date, days=180)
    if df_decision.empty or len(df_decision) < 20:
        return {"error": f"{fund_code} 数据不足"}

    indicators = calc_indicators_at_date(df_decision)
    if "error" in indicators:
        return {"error": f"{fund_code} 指标计算失败"}

    # 2. Agent 打分
    score = agent_like_score(indicators)

    # 3. 获取持有期的实际走势
    df_full = fetch_nav_up_to(fund_code, today, days=days_ago + hold_days + 30)
    if df_full.empty:
        return {"error": f"{fund_code} 持有期数据不足"}

    buy_nav = indicators["current_nav"]  # 决策日净值

    # 4. 模拟持有（含止损止盈）
    df_hold = df_full[df_full["净值日期"] > decision_date].head(hold_days + 10)

    actual_return = None
    actual_exit_date = None
    exit_reason = "到期"
    max_return = 0
    min_return = 0

    for _, row in df_hold.iterrows():
        ret = (float(row["单位净值"]) - buy_nav) / buy_nav * 100
        max_return = max(max_return, ret)
        min_return = min(min_return, ret)

        # 止损检查
        if ret <= stop_loss_pct:
            actual_return = round(ret, 2)
            actual_exit_date = row["净值日期"].strftime("%Y-%m-%d")
            exit_reason = f"止损({stop_loss_pct}%)"
            break

        # 止盈检查
        if ret >= take_profit_pct:
            actual_return = round(ret, 2)
            actual_exit_date = row["净值日期"].strftime("%Y-%m-%d")
            exit_reason = f"止盈({take_profit_pct}%)"
            break

    # 如果没触发止损止盈，取持有期末收益
    if actual_return is None and len(df_hold) > 0:
        last = df_hold.iloc[min(hold_days - 1, len(df_hold) - 1)]
        actual_return = round((float(last["单位净值"]) - buy_nav) / buy_nav * 100, 2)
        actual_exit_date = last["净值日期"].strftime("%Y-%m-%d")

    # 5. 获取同期沪深300收益作为基准
    benchmark_return = None
    try:
        hs300_df = ak.index_zh_a_hist(symbol="000300", period="daily",
                                       start_date=decision_date.replace("-", ""),
                                       end_date=min(actual_exit_date or exit_date, today).replace("-", ""))
        if not hs300_df.empty and len(hs300_df) >= 2:
            bench_start = float(hs300_df.iloc[0]["收盘"])
            bench_end = float(hs300_df.iloc[-1]["收盘"])
            benchmark_return = round((bench_end - bench_start) / bench_start * 100, 2)
    except Exception:
        benchmark_return = None

    return {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "decision_date": decision_date,
        "exit_date": actual_exit_date or exit_date,
        "buy_nav": buy_nav,
        "agent_score": score,
        "would_recommend": score >= 6.5,  # Agent 阈值：6.5 分以上才推荐
        "actual_return": actual_return,
        "max_return": round(max_return, 2),
        "min_return": round(min_return, 2),
        "exit_reason": exit_reason,
        "benchmark_return": benchmark_return,
        "alpha": round(actual_return - benchmark_return, 2) if actual_return is not None and benchmark_return is not None else None,
        "indicators": indicators,
    }


def backtest_portfolio(
    funds: list = None,
    days_ago: int = 90,
    hold_days: int = 60,
) -> dict:
    """
    回测一篮子基金组合
    funds: [{"code": "003834", "name": "华夏能源革新"}, ...]
    """
    if funds is None:
        # Agent 推荐的4只基金
        funds = [
            {"code": "003834", "name": "华夏能源革新股票A"},
            {"code": "110011", "name": "易方达中小盘混合"},
            {"code": "001071", "name": "华安媒体互联网混合A"},
            {"code": "161725", "name": "招商中证白酒指数(LOF)A"},
        ]

    results = []
    for f in funds:
        print(f"  回测 {f['name']} ({f['code']})...", flush=True)
        r = backtest_fund(f["code"], f["name"], days_ago, hold_days)
        results.append(r)

    # 组合统计
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "全部回测失败", "details": results}

    recommended = [r for r in valid if r["would_recommend"]]
    not_recommended = [r for r in valid if not r["would_recommend"]]

    returns_rec = [r["actual_return"] for r in recommended if r["actual_return"] is not None]
    returns_all = [r["actual_return"] for r in valid if r["actual_return"] is not None]

    alphas = [r["alpha"] for r in valid if r["alpha"] is not None]

    return {
        "backtest_period": f"{days_ago}天前买入，持有{hold_days}天",
        "total_funds": len(funds),
        "recommended_count": len(recommended),
        "not_recommended_count": len(not_recommended),
        "avg_return_recommended": round(np.mean(returns_rec), 2) if returns_rec else None,
        "avg_return_all": round(np.mean(returns_all), 2) if returns_all else None,
        "win_rate": round(sum(1 for r in returns_rec if r > 0) / len(returns_rec) * 100, 1) if returns_rec else None,
        "avg_alpha": round(np.mean(alphas), 2) if alphas else None,
        "details": results,
    }


def print_backtest_report(result: dict):
    """打印可读的回测报告"""
    if "error" in result and "details" not in result:
        print(f"❌ 回测失败: {result['error']}")
        return

    print("\n" + "=" * 70)
    print("  📊 回测报告 — 验证 Agent 推荐有效性")
    print("=" * 70)
    print(f"  回测期间: {result.get('backtest_period', 'N/A')}")
    print(f"  基金总数: {result.get('total_funds', 'N/A')}")
    print()

    # 逐只基金
    print("-" * 70)
    print(f"  {'基金':<20} {'Agent评分':>8} {'推荐?':>6} {'实际收益':>8} {'基准':>8} {'Alpha':>8} {'退出原因':>10}")
    print("-" * 70)

    for r in result.get("details", []):
        if "error" in r:
            print(f"  {r.get('fund_code', '?'):<20} ❌ {r['error']}")
            continue
        rec = "✅" if r["would_recommend"] else "❌"
        ret = f"{r['actual_return']:+.2f}%" if r["actual_return"] is not None else "N/A"
        bench = f"{r['benchmark_return']:+.2f}%" if r["benchmark_return"] is not None else "N/A"
        alpha = f"{r['alpha']:+.2f}%" if r["alpha"] is not None else "N/A"
        print(f"  {r['fund_name'][:18]:<20} {r['agent_score']:>8.1f} {rec:>6} {ret:>8} {bench:>8} {alpha:>8} {r['exit_reason']:>10}")

    # 汇总
    print("-" * 70)
    print()
    print("  📈 汇总统计:")
    if result.get("avg_return_recommended") is not None:
        print(f"    推荐基金平均收益: {result['avg_return_recommended']:+.2f}%")
    if result.get("avg_return_all") is not None:
        print(f"    全部基金平均收益: {result['avg_return_all']:+.2f}%")
    if result.get("win_rate") is not None:
        print(f"    推荐基金胜率: {result['win_rate']:.1f}%")
    if result.get("avg_alpha") is not None:
        print(f"    平均 Alpha(vs 沪深300): {result['avg_alpha']:+.2f}%")
        if result["avg_alpha"] > 0:
            print(f"    ✅ Agent 推荐跑赢基准 {result['avg_alpha']:.2f}%")
        else:
            print(f"    ❌ Agent 推荐跑输基准 {abs(result['avg_alpha']):.2f}%")
    print()
    print("=" * 70)

    # 判断
    if result.get("avg_alpha") is not None:
        if result["avg_alpha"] > 3:
            print("  🏆 结论：Agent 推荐显著跑赢基准，有效性较强")
        elif result["avg_alpha"] > 0:
            print("  ✅ 结论：Agent 推荐跑赢基准，但优势不明显，需更多样本验证")
        elif result["avg_alpha"] > -3:
            print("  ⚠️ 结论：Agent 推荐略跑输基准，可能需要优化评分逻辑")
        else:
            print("  ❌ 结论：Agent 推荐显著跑输基准，评分体系需要重新设计")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基金 Trade Agent 回测")
    parser.add_argument("--days-ago", type=int, default=90, help="多少天前买入（默认90）")
    parser.add_argument("--hold", type=int, default=60, help="持有天数（默认60）")
    parser.add_argument("--fund", type=str, default=None, help="回测单只基金代码")
    args = parser.parse_args()

    if args.fund:
        print(f"回测单只基金: {args.fund}")
        r = backtest_fund(args.fund, args.fund, days_ago=args.days_ago, hold_days=args.hold)
        print_backtest_report({"details": [r]})
    else:
        print(f"回测 Agent 推荐组合（{args.days_ago}天前买入，持有{args.hold}天）...")
        result = backtest_portfolio(days_ago=args.days_ago, hold_days=args.hold)
        print_backtest_report(result)

        # 保存 JSON 结果
        output_file = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"详细结果已保存: {output_file}")
