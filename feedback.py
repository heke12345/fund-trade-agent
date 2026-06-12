"""
反馈环模块 v2 — 让 Agent 从历史推荐的真实收益中学习
五大组件：Recommendation Log → Outcome Tracker → Review Engine → Score Optimizer → Experience Injector

v2 核心升级：
  ✅ auto_track_and_review(): 一键追踪→复盘→调参
  ✅ 深度因果分析：盈亏基金各维度得分差异 → 权重自适应
  ✅ optimize_scoring_weights(): 基于历史数据自动调参，写入 scoring_weights.json
  ✅ generate_optimization_report(): 人可读优化报告（追踪表+权重调整+原因）
  ✅ 主动通知：优化逻辑和原因

使用方式：
  from feedback import (
      log_recommendation, track_outcome, review_experience, inject_experience,
      auto_track_and_review, optimize_scoring_weights, generate_optimization_report,
  )
"""

import json
import os
from datetime import datetime, timedelta
from copy import deepcopy

RECOMMENDATION_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendation_log.json")
SCORING_WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring_weights.json")

# 默认评分权重（与 fund_data.py 保持一致）
DEFAULT_WEIGHTS = {
    "trend": 0.30,
    "rsi": 0.20,
    "macd": 0.15,
    "volatility": 0.15,
    "sharpe": 0.10,
    "drawdown": 0.10,
}


# ============================================================
# 1. Recommendation Log — 记录推荐
# ============================================================

def log_recommendation(
    fund_code: str,
    fund_name: str,
    score: float,
    reason: str,
    target_return: float,
    stop_loss: float,
    layer_passed: str = "",
    score_detail: dict = None,
) -> dict:
    """
    记录一次推荐到日志
    layer_passed: 经过了哪些层（如 "L1→L1.5→L2→L3→L4"）
    score_detail: 各维度得分明细，如 {"trend": 10, "rsi": 7, ...}
    """
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "fund_code": fund_code,
        "fund_name": fund_name,
        "quant_score": score,
        "reason": reason,
        "target_return": target_return,
        "stop_loss": stop_loss,
        "layer_passed": layer_passed,
        "score_detail": score_detail or {},
        "outcome": None,  # 待追踪
    }
    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)
    return entry


# ============================================================
# 2. Outcome Tracker — 追踪结果
# ============================================================

def track_outcome(fund_code: str, recommend_date: str, days: int = 60) -> dict:
    """
    追踪推荐结果：计算从推荐日到持有期结束的实际收益
    如果持有期尚未结束，取最新净值计算持有中收益
    """
    try:
        import akshare as ak
        import pandas as pd

        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        df["净值日期"] = pd.to_datetime(df["净值日期"])

        rec_date = pd.Timestamp(recommend_date)
        exit_date = rec_date + timedelta(days=days)

        buy_rows = df[df["净值日期"] >= rec_date]
        if buy_rows.empty:
            return {"error": f"无推荐日后的数据: {recommend_date}"}
        buy_row = buy_rows.iloc[0]

        sell_rows = df[df["净值日期"] >= exit_date]
        if sell_rows.empty:
            sell_row = df.iloc[-1]  # 取最新数据
            holding = True
        else:
            sell_row = sell_rows.iloc[0]
            holding = False

        buy_nav = float(buy_row["单位净值"])
        sell_nav = float(sell_row["单位净值"])
        actual_return = round((sell_nav - buy_nav) / buy_nav * 100, 2)

        # 计算持有天数
        holding_days = (sell_row["净值日期"] - buy_row["净值日期"]).days

        # 计算年化收益
        annualized = round(actual_return / max(holding_days, 1) * 365, 2) if holding_days > 0 else 0

        return {
            "buy_date": buy_row["净值日期"].strftime("%Y-%m-%d"),
            "sell_date": sell_row["净值日期"].strftime("%Y-%m-%d"),
            "buy_nav": buy_nav,
            "sell_nav": sell_nav,
            "actual_return": actual_return,
            "annualized_return": annualized,
            "holding_days": holding_days,
            "holding": holding,  # 是否仍在持有中
            "profitable": actual_return > 0,
        }
    except Exception as e:
        return {"error": str(e)}


def update_outcomes(days: int = 60) -> int:
    """
    批量更新所有未追踪推荐的 outcome
    返回更新条数
    """
    logs = _load_logs()
    updated = 0
    for entry in logs:
        if entry.get("outcome") is None and entry.get("date"):
            outcome = track_outcome(entry["fund_code"], entry["date"], days)
            if "error" not in outcome:
                entry["outcome"] = outcome
                updated += 1
    if updated > 0:
        _save_logs(logs)
    return updated


def get_performance_table() -> list:
    """
    获取所有推荐基金的实时盈亏表
    自动刷新 outcome=null 的记录
    """
    logs = _load_logs()
    table = []
    for entry in logs:
        # 如果 outcome 为 null，尝试实时追踪
        if entry.get("outcome") is None and entry.get("date"):
            outcome = track_outcome(entry["fund_code"], entry["date"])
            if "error" not in outcome:
                entry["outcome"] = outcome
                _save_logs(logs)

        row = {
            "date": entry.get("date", ""),
            "fund_code": entry.get("fund_code", ""),
            "fund_name": entry.get("fund_name", ""),
            "quant_score": entry.get("quant_score", 0),
            "target_return": entry.get("target_return", 0),
            "stop_loss": entry.get("stop_loss", 0),
        }
        if entry.get("outcome") and "error" not in entry["outcome"]:
            o = entry["outcome"]
            row.update({
                "buy_date": o.get("buy_date", ""),
                "buy_nav": o.get("buy_nav", 0),
                "current_nav": o.get("sell_nav", 0),
                "actual_return": o.get("actual_return", 0),
                "annualized_return": o.get("annualized_return", 0),
                "holding_days": o.get("holding_days", 0),
                "holding": o.get("holding", False),
                "profitable": o.get("profitable", False),
            })
        else:
            row.update({"error": entry["outcome"].get("error", "未知错误") if entry.get("outcome") else "未追踪"})
        table.append(row)
    return table


# ============================================================
# 3. Review Engine — 深度复盘（v2 升级）
# ============================================================

def review_experience() -> str:
    """
    深度复盘历史推荐，生成经验教训文本
    v2: 增加因果分析 — 哪些维度对盈亏预测力最强/最弱
    """
    logs = _load_logs()
    reviewed = [l for l in logs if l.get("outcome") is not None and "error" not in l.get("outcome", {})]

    if not reviewed:
        return ""

    wins = [r for r in reviewed if r["outcome"] and r["outcome"].get("profitable")]
    losses = [r for r in reviewed if r["outcome"] and not r["outcome"].get("profitable")]

    lines = [
        f"【历史推荐复盘】共{len(reviewed)}次推荐，{len(wins)}次盈利，{len(losses)}次亏损",
        f"胜率：{len(wins)/len(reviewed)*100:.0f}%",
    ]

    if wins:
        avg_win = sum(r["outcome"]["actual_return"] for r in wins) / len(wins)
        lines.append(f"平均盈利：{avg_win:+.2f}%")
        win_names = [f"{r['fund_name']}({r['quant_score']}分,实际{r['outcome']['actual_return']:+.2f}%)" for r in wins]
        lines.append(f"盈利推荐：{', '.join(win_names)}")

    if losses:
        avg_loss = sum(r["outcome"]["actual_return"] for r in losses) / len(losses)
        lines.append(f"平均亏损：{avg_loss:+.2f}%")
        loss_names = [f"{r['fund_name']}({r['quant_score']}分,实际{r['outcome']['actual_return']:+.2f}%)" for r in losses]
        lines.append(f"亏损推荐：{', '.join(loss_names)}")

    # 因果分析：各维度区分度
    dims = ["trend", "rsi", "macd", "volatility", "sharpe", "drawdown"]
    if len(reviewed) >= 3:
        lines.append("\n【维度区分度分析】")
        for dim in dims:
            win_scores = [r.get("score_detail", {}).get(dim, 0) for r in wins if r.get("score_detail", {}).get(dim) is not None]
            loss_scores = [r.get("score_detail", {}).get(dim, 0) for r in losses if r.get("score_detail", {}).get(dim) is not None]
            if win_scores and loss_scores:
                win_avg = sum(win_scores) / len(win_scores)
                loss_avg = sum(loss_scores) / len(loss_scores)
                discriminative = win_avg - loss_avg
                if discriminative > 1:
                    lines.append(f"  ✅ {dim}: 盈利组均分{win_avg:.1f} vs 亏损组{loss_avg:.1f} (区分度+{discriminative:.1f}，预测力强)")
                elif discriminative < -1:
                    lines.append(f"  ⚠️ {dim}: 盈利组均分{win_avg:.1f} vs 亏损组{loss_avg:.1f} (区分度{discriminative:.1f}，可能需要调整评分规则)")
                else:
                    lines.append(f"  ➡️ {dim}: 盈利组均分{win_avg:.1f} vs 亏损组{loss_avg:.1f} (区分度{discriminative:.1f}，无显著差异)")

    # 评分阈值验证
    if len(reviewed) >= 5:
        high_score = [r for r in reviewed if r.get("quant_score", 0) >= 7]
        low_score = [r for r in reviewed if r.get("quant_score", 0) < 7]
        if high_score and low_score:
            high_wr = sum(1 for r in high_score if r["outcome"] and r["outcome"].get("profitable")) / len(high_score)
            low_wr = sum(1 for r in low_score if r["outcome"] and r["outcome"].get("profitable")) / len(low_score)
            lines.append(f"\n评分阈值验证：高评分(≥7)胜率{high_wr*100:.0f}% vs 低评分(<7)胜率{low_wr*100:.0f}%")

    # 关键教训
    if losses:
        lines.append("\n关键教训：")
        for r in losses[:3]:
            reason = r.get("reason", "未知")[:80]
            ret = r["outcome"]["actual_return"]
            lines.append(f"  ❌ {r['fund_name']}({r['quant_score']}分): {reason} → 实际{ret:+.2f}%")

    if wins:
        lines.append("\n成功经验：")
        for r in wins[:3]:
            reason = r.get("reason", "未知")[:80]
            ret = r["outcome"]["actual_return"]
            lines.append(f"  ✅ {r['fund_name']}({r['quant_score']}分): {reason} → 实际{ret:+.2f}%")

    return "\n".join(lines)


# ============================================================
# 4. Score Optimizer — 权重自适应（v2 新增）
# ============================================================

def load_scoring_weights() -> dict:
    """从 scoring_weights.json 读取权重，不存在则返回默认值"""
    if os.path.exists(SCORING_WEIGHTS_FILE):
        try:
            with open(SCORING_WEIGHTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 返回权重部分
                return {k: data[k] for k in DEFAULT_WEIGHTS if k in data}
        except (json.JSONDecodeError, IOError):
            pass
    return deepcopy(DEFAULT_WEIGHTS)


def _save_scoring_weights(weights: dict, change_reason: str = ""):
    """保存权重到 scoring_weights.json，附带更新历史"""
    data = {}
    if os.path.exists(SCORING_WEIGHTS_FILE):
        try:
            with open(SCORING_WEIGHTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}

    old_weights = {k: data.get(k, v) for k, v in DEFAULT_WEIGHTS.items()}
    data.update(weights)
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    if "update_history" not in data:
        data["update_history"] = []

    # 记录变更
    changes = {}
    for k in DEFAULT_WEIGHTS:
        old_v = old_weights.get(k, DEFAULT_WEIGHTS[k])
        new_v = weights.get(k, DEFAULT_WEIGHTS[k])
        if abs(old_v - new_v) > 0.001:
            changes[k] = {"from": round(old_v, 2), "to": round(new_v, 2)}

    if changes:
        data["update_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "changes": changes,
            "reason": change_reason,
        })

    with open(SCORING_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def optimize_scoring_weights() -> dict:
    """
    基于历史数据自动调参
    核心算法：对比盈利组和亏损组在各维度的得分差异，调整权重
    
    返回:
      {
        "optimized": True/False,
        "old_weights": {...},
        "new_weights": {...},
        "changes": {...},
        "reason": "...",
        "discriminative_power": {...},
      }
    """
    logs = _load_logs()
    reviewed = [l for l in logs if l.get("outcome") is not None and "error" not in l.get("outcome", {})]

    if len(reviewed) < 3:
        return {
            "optimized": False,
            "reason": f"历史推荐不足（需≥3，当前{len(reviewed)}），无法优化权重",
            "old_weights": load_scoring_weights(),
            "new_weights": load_scoring_weights(),
            "changes": {},
            "discriminative_power": {},
        }

    wins = [r for r in reviewed if r["outcome"] and r["outcome"].get("profitable")]
    losses = [r for r in reviewed if r["outcome"] and not r["outcome"].get("profitable")]

    if not wins or not losses:
        return {
            "optimized": False,
            "reason": f"需要至少1次盈利和1次亏损（当前盈利{len(wins)}次，亏损{len(losses)}次）",
            "old_weights": load_scoring_weights(),
            "new_weights": load_scoring_weights(),
            "changes": {},
            "discriminative_power": {},
        }

    # 计算各维度的区分度
    dims = ["trend", "rsi", "macd", "volatility", "sharpe", "drawdown"]
    discriminative_power = {}
    for dim in dims:
        win_scores = [r.get("score_detail", {}).get(dim, 0) for r in wins if r.get("score_detail", {}).get(dim) is not None]
        loss_scores = [r.get("score_detail", {}).get(dim, 0) for r in losses if r.get("score_detail", {}).get(dim) is not None]
        if win_scores and loss_scores:
            win_avg = sum(win_scores) / len(win_scores)
            loss_avg = sum(loss_scores) / len(loss_scores)
            discriminative_power[dim] = round(win_avg - loss_avg, 2)
        else:
            discriminative_power[dim] = 0

    # 基于区分度调整权重
    old_weights = load_scoring_weights()
    new_weights = deepcopy(old_weights)

    # 将区分度转为权重调整量（正区分度→加权重，负→降权重）
    MAX_ADJUSTMENT = 0.05  # 每次最多调整5个百分点，防震荡
    MIN_WEIGHT = 0.05      # 最低5%保底

    for dim in dims:
        dp = discriminative_power.get(dim, 0)
        # 区分度映射为权重调整：dp>0 说明该维度能有效区分盈亏，加权重
        # dp<0 说明该维度失灵，降权重
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, dp * 0.01))  # 缩放系数0.01
        new_weights[dim] = round(max(MIN_WEIGHT, old_weights[dim] + adjustment), 3)

    # 归一化使总和=1
    total = sum(new_weights.values())
    for dim in dims:
        new_weights[dim] = round(new_weights[dim] / total, 3)

    # 二次归一化微调（处理浮点误差）
    diff = 1.0 - sum(new_weights.values())
    max_dim = max(new_weights, key=new_weights.get)
    new_weights[max_dim] = round(new_weights[max_dim] + diff, 3)

    # 检查是否有实质变化
    changes = {}
    for dim in dims:
        old_v = old_weights.get(dim, DEFAULT_WEIGHTS[dim])
        new_v = new_weights.get(dim, DEFAULT_WEIGHTS[dim])
        if abs(old_v - new_v) > 0.001:
            changes[dim] = {
                "from": old_v,
                "to": new_v,
                "delta": round(new_v - old_v, 3),
            }

    if not changes:
        return {
            "optimized": False,
            "reason": "权重无需调整，当前权重已是最优",
            "old_weights": old_weights,
            "new_weights": new_weights,
            "changes": {},
            "discriminative_power": discriminative_power,
        }

    # 生成变更原因
    increase_dims = [f"{d}(+{v['delta']:+.1%}, 区分度{discriminative_power.get(d,0):+.1f})" for d, v in changes.items() if v["delta"] > 0]
    decrease_dims = [f"{d}({v['delta']:+.1%}, 区分度{discriminative_power.get(d,0):+.1f})" for d, v in changes.items() if v["delta"] < 0]
    
    reason_parts = [f"基于{len(reviewed)}次推荐（{len(wins)}盈{len(losses)}亏）分析："]
    if increase_dims:
        reason_parts.append(f"加权重：{', '.join(increase_dims)}（预测力强）")
    if decrease_dims:
        reason_parts.append(f"降权重：{', '.join(decrease_dims)}（预测力弱/失灵）")

    # 保存新权重
    _save_scoring_weights(new_weights, change_reason="; ".join(reason_parts))

    return {
        "optimized": True,
        "reason": "; ".join(reason_parts),
        "old_weights": old_weights,
        "new_weights": new_weights,
        "changes": changes,
        "discriminative_power": discriminative_power,
    }


# ============================================================
# 5. 一键追踪+复盘+调参
# ============================================================

def auto_track_and_review() -> dict:
    """
    一键执行：追踪→复盘→调参
    返回完整的优化报告
    """
    # 1. 追踪所有未更新的推荐
    tracked_count = update_outcomes()

    # 2. 复盘
    review_text = review_experience()

    # 3. 权重优化
    optimization = optimize_scoring_weights()

    # 4. 生成优化报告
    report = generate_optimization_report(tracked_count, review_text, optimization)

    return {
        "tracked_count": tracked_count,
        "review_text": review_text,
        "optimization": optimization,
        "report": report,
    }


# ============================================================
# 6. Optimization Report — 人可读优化报告
# ============================================================

def generate_optimization_report(tracked_count: int = 0, review_text: str = "", optimization: dict = None) -> str:
    """
    生成人可读的优化报告
    """
    logs = _load_logs()
    reviewed = [l for l in logs if l.get("outcome") is not None and "error" not in l.get("outcome", {})]

    lines = []
    lines.append("=" * 60)
    lines.append("  📊 反馈环优化报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    # 追踪结果
    lines.append(f"\n📌 本次追踪更新: {tracked_count} 条")
    lines.append(f"📌 历史推荐总计: {len(logs)} 条（已追踪 {len(reviewed)} 条）")

    if not reviewed:
        lines.append("\n⚠️ 暂无追踪数据，无法生成优化报告")
        return "\n".join(lines)

    # 盈亏汇总表
    wins = [r for r in reviewed if r["outcome"] and r["outcome"].get("profitable")]
    losses = [r for r in reviewed if r["outcome"] and not r["outcome"].get("profitable")]

    lines.append(f"\n{'─' * 60}")
    lines.append("  💰 推荐基金盈亏表")
    lines.append(f"{'─' * 60}")
    lines.append(f"{'基金名称':<20} {'评分':>4} {'买入':>8} {'现价':>8} {'收益':>8} {'状态':>6}")
    lines.append(f"{'─' * 60}")

    for r in reviewed:
        name = r.get("fund_name", "")[:18]
        score = r.get("quant_score", 0)
        o = r.get("outcome", {})
        buy = o.get("buy_nav", 0)
        current = o.get("sell_nav", 0)
        ret = o.get("actual_return", 0)
        holding = "持有中" if o.get("holding") else ("✅盈利" if o.get("profitable") else "❌亏损")
        lines.append(f"{name:<20} {score:>4.1f} {buy:>8.4f} {current:>8.4f} {ret:>+7.2f}% {holding:>6}")

    lines.append(f"{'─' * 60}")
    win_rate = len(wins) / len(reviewed) * 100 if reviewed else 0
    lines.append(f"胜率: {win_rate:.0f}% ({len(wins)}盈/{len(losses)}亏)")

    if wins:
        avg_win = sum(r["outcome"]["actual_return"] for r in wins) / len(wins)
        lines.append(f"平均盈利: {avg_win:+.2f}%")
    if losses:
        avg_loss = sum(r["outcome"]["actual_return"] for r in losses) / len(losses)
        lines.append(f"平均亏损: {avg_loss:+.2f}%")

    # 优化动作
    if optimization and optimization.get("optimized"):
        lines.append(f"\n{'─' * 60}")
        lines.append("  🔧 权重优化调整")
        lines.append(f"{'─' * 60}")
        lines.append(f"原因: {optimization.get('reason', '')}")
        lines.append("")
        lines.append(f"{'维度':<12} {'原权重':>8} {'新权重':>8} {'变化':>8} {'区分度':>8}")
        lines.append(f"{'─' * 50}")

        changes = optimization.get("changes", {})
        dp = optimization.get("discriminative_power", {})
        for dim in ["trend", "rsi", "macd", "volatility", "sharpe", "drawdown"]:
            old_v = optimization.get("old_weights", {}).get(dim, DEFAULT_WEIGHTS[dim])
            new_v = optimization.get("new_weights", {}).get(dim, DEFAULT_WEIGHTS[dim])
            change = changes.get(dim, {})
            delta_str = f"{change['delta']:+.1%}" if change else "—"
            dp_val = dp.get(dim, 0)
            lines.append(f"{dim:<12} {old_v:>7.1%} {new_v:>7.1%} {delta_str:>8} {dp_val:>+7.2f}")

        lines.append(f"{'─' * 50}")
        lines.append("💡 调整幅度限制: 每次 ≤5%，防止震荡")
    elif optimization and not optimization.get("optimized"):
        lines.append(f"\n⚠️ 权重暂不调整: {optimization.get('reason', '无数据')}")

    # 复盘摘要
    if review_text:
        lines.append(f"\n{'─' * 60}")
        lines.append("  📝 复盘摘要")
        lines.append(f"{'─' * 60}")
        lines.append(review_text)

    return "\n".join(lines)


# ============================================================
# 7. Experience Injector — 注入经验到 Agent prompt
# ============================================================

def inject_experience() -> str:
    """
    生成可注入 Agent prompt 的经验文本
    如果没有历史经验则返回空字符串
    """
    exp = review_experience()
    if not exp:
        return ""

    # 附加权重调整信息
    current_weights = load_scoring_weights()
    weights_changed = current_weights != DEFAULT_WEIGHTS
    weights_info = ""
    if weights_changed:
        weights_info = f"\n\n【当前评分权重（已优化）】\n"
        for k, v in current_weights.items():
            default_v = DEFAULT_WEIGHTS.get(k, 0)
            if abs(v - default_v) > 0.001:
                weights_info += f"  {k}: {v:.0%}（默认{default_v:.0%}，已调整{v-default_v:+.0%}）\n"
            else:
                weights_info += f"  {k}: {v:.0%}\n"

    return f"""

{exp}
{weights_info}
请在推荐时参考以上历史经验，避免重复犯错。特别注意亏损推荐的共性特征，对类似情况提高警惕。评分权重已根据历史胜率自动优化，请按照新权重评分。
"""


# ============================================================
# 辅助函数
# ============================================================

def _load_logs() -> list:
    if not os.path.exists(RECOMMENDATION_LOG):
        return []
    try:
        with open(RECOMMENDATION_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_logs(logs: list):
    with open(RECOMMENDATION_LOG, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def get_recommendation_summary() -> dict:
    """获取推荐日志摘要"""
    logs = _load_logs()
    tracked = [l for l in logs if l.get("outcome") is not None]
    wins = [l for l in tracked if l["outcome"] and l["outcome"].get("profitable")]
    return {
        "total": len(logs),
        "tracked": len(tracked),
        "wins": len(wins),
        "losses": len(tracked) - len(wins),
        "win_rate": round(len(wins)/len(tracked)*100, 1) if tracked else None,
    }


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  反馈环模块 v2 测试", flush=True)
    print("=" * 60, flush=True)

    # 摘要
    summary = get_recommendation_summary()
    print(f"\n推荐摘要: {summary}", flush=True)

    # 追踪+复盘+调参
    print("\n执行 auto_track_and_review()...", flush=True)
    result = auto_track_and_review()

    print(f"\n追踪更新: {result['tracked_count']} 条", flush=True)
    print(f"\n优化结果: optimized={result['optimization'].get('optimized')}", flush=True)
    print(f"原因: {result['optimization'].get('reason', 'N/A')}", flush=True)

    # 完整报告
    print("\n" + result["report"], flush=True)
