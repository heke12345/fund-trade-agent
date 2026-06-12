"""
基金 Trade Agent Web UI v3 — 真正的实时可视化
启动方式: python web_app.py
访问地址: http://localhost:5001
"""

import os
import sys
import json
import re
import threading
import time
import traceback
from datetime import datetime
from queue import Queue, Empty

# LiteLLM 防阻塞（必须在 import crewai 之前）
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_MODE"] = "PRODUCTION"

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="web_static", static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# 全局状态
# ============================================================
agent_state = {
    "running": False,
    "progress": "",
    "start_time": None,
    "result": None,
    "error": None,
    "current_layer": 0,
    "current_agent": "",
    "agent_statuses": {},  # agent_id -> "idle"|"running"|"done"|"vetoed"
    "agent_outputs": {},   # agent_id -> output_text (摘要)
}

# 启动锁 — 防止并发启动导致 CrewAI Executor 冲突
agent_start_lock = threading.Lock()

# ============================================================
# 事件追踪系统 v2 — 基于 SSE 实时推送
# ============================================================
# 用 Queue 替代 list，每个 SSE 客户端一个 queue
sse_clients = []
sse_lock = threading.Lock()
event_counter = 0

AGENT_LAYERS = [
    {"id": 1, "name": "筛选层", "subtitle": "Data Screening", "color": "#3b82f6", "agents": [
        {"id": "fund_screener", "name": "基金数据筛选师", "icon": "🔍"},
        {"id": "macro_analyst", "name": "宏观经济分析师", "icon": "🌏"},
    ]},
    {"id": 1.5, "name": "否决层", "subtitle": "Hard Constraint Veto", "color": "#ef4444", "agents": [
        {"id": "constraint_veto", "name": "硬约束否决官", "icon": "⛔"},
    ]},
    {"id": 2, "name": "分析层", "subtitle": "Multi-dimensional Analysis", "color": "#10b981", "agents": [
        {"id": "technical_analyst", "name": "技术分析专家", "icon": "📊"},
        {"id": "fundamental_analyst", "name": "基本面分析专家", "icon": "📋"},
        {"id": "sentiment_analyst", "name": "市场情绪分析师", "icon": "🎭"},
    ]},
    {"id": 3, "name": "辩论层", "subtitle": "Parallel Debate + Judge", "color": "#8b5cf6", "agents": [
        {"id": "bullish_researcher", "name": "多头研究员", "icon": "🐂"},
        {"id": "bearish_researcher", "name": "空头研究员", "icon": "🐻"},
        {"id": "debate_judge", "name": "辩论裁决官", "icon": "⚖️"},
    ]},
    {"id": 4, "name": "决策层", "subtitle": "Decision & Risk Control", "color": "#f59e0b", "agents": [
        {"id": "risk_controller", "name": "风控专家", "icon": "🛡️"},
        {"id": "portfolio_manager", "name": "投资组合经理", "icon": "💼"},
    ]},
]

# Agent 名称 → agent_id 映射（CrewAI 输出用中文名）
NAME_TO_ID = {}
for layer in AGENT_LAYERS:
    for a in layer["agents"]:
        NAME_TO_ID[a["name"]] = {"id": a["id"], "layer": layer["id"]}

# Agent ID → Layer 映射
ID_TO_LAYER = {}
for layer in AGENT_LAYERS:
    for a in layer["agents"]:
        ID_TO_LAYER[a["id"]] = layer["id"]


# ============================================================
# 安全 stdout — 防止后台线程 print 导致 BrokenPipeError
# ============================================================
class SafeStdout:
    """包装 stdout，静默处理 BrokenPipeError"""
    def __init__(self, original):
        self.original = original

    def write(self, text):
        try:
            self.original.write(text)
        except (BrokenPipeError, OSError, IOError):
            pass

    def flush(self):
        try:
            self.original.flush()
        except (BrokenPipeError, OSError, IOError):
            pass

    def fileno(self):
        try:
            return self.original.fileno()
        except (BrokenPipeError, OSError, IOError):
            return -1

    def isatty(self):
        try:
            return self.original.isatty()
        except (BrokenPipeError, OSError, IOError):
            return False


# ============================================================
# 事件缓存 — 让新连接的 SSE 客户端也能收到最近事件
# ============================================================
MAX_CACHED_EVENTS = 30
recent_events_cache = []
recent_events_lock = threading.Lock()


def _cache_event(event):
    with recent_events_lock:
        recent_events_cache.append(event)
        if len(recent_events_cache) > MAX_CACHED_EVENTS:
            recent_events_cache.pop(0)


def broadcast_event(event_type, agent_id, agent_name, layer, message="", output_preview=""):
    global event_counter
    event_counter += 1
    event = {
        "id": event_counter,
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "layer": layer,
        "message": message,
        "output_preview": (output_preview or "")[:500],
    }

    # 更新全局状态
    if event_type == "agent_start":
        agent_state["agent_statuses"][agent_id] = "running"
        agent_state["current_agent"] = agent_name
        agent_state["current_layer"] = layer
    elif event_type == "agent_done":
        agent_state["agent_statuses"][agent_id] = "done"
        agent_state["agent_outputs"][agent_id] = (output_preview or "")[:300]
    elif event_type == "veto":
        agent_state["agent_statuses"][agent_id] = "vetoed"

    # 缓存事件
    _cache_event(event)

    # 推送给所有 SSE 客户端
    with sse_lock:
        dead = []
        for i, q in enumerate(sse_clients):
            try:
                q.put_nowait(event)
            except:
                dead.append(i)
        for i in reversed(dead):
            sse_clients.pop(i)

    return event


# ============================================================
# Stdout 拦截器 — 从 CrewAI 的输出中解析 Agent 进度
# ============================================================
class CrewAIStreamInterceptor:
    """拦截 sys.stdout，实时解析 CrewAI 的结构化输出，推送 SSE 事件"""

    def __init__(self, original_stdout):
        self.original = original_stdout
        self.buffer = ""
        self.current_agent_id = None
        self.current_agent_name = None
        self.current_layer = None
        self.agent_output_buffer = ""
        self.capturing_output = False
        # 状态机：刚看到 Agent Started 标题，等待 Agent: XXX 行
        self.expecting_agent_name = False
        self._last_agent_started_pos = 0
        # v4: 定期推送中间输出
        self._last_progress_push = 0
        self._last_debate_push = {"bull": 0, "bear": 0, "judge": 0}
        # v4: 辩论内容缓存
        self._debate_buffer = {"bull": "", "bear": "", "judge": ""}

    def write(self, text):
        # 始终写入原始 stdout（保持日志）
        self.original.write(text)
        self.original.flush()

        if not text or not text.strip():
            return

        self.buffer += text
        now = time.time()

        # ── 状态机 Step 1: 检测 Agent Started 标题 ──
        if not self.expecting_agent_name and 'Agent Started' in self.buffer:
            pos = self.buffer.rfind('Agent Started')
            if pos > self._last_agent_started_pos:
                self.expecting_agent_name = True
                self._last_agent_started_pos = pos
                self.agent_output_buffer = ""

        # ── 状态机 Step 2: 在 expecting 状态下找 Agent: XXX ──
        if self.expecting_agent_name:
            match = re.search(r'Agent:\s*([^\n╭╰╯╮│┃┆┊┏┓┗┛┣┫┳┻╋─━┄┅┈┉\s][^\n╭╰╯╮│┃┆┊┏┓┗┛┣┫┳┻╋─━┄┅┈┉]*)', self.buffer)
            if match:
                agent_name = match.group(1).strip()
                matched_id = None
                matched_layer = None
                matched_name = None

                if agent_name in NAME_TO_ID:
                    matched_id = NAME_TO_ID[agent_name]["id"]
                    matched_layer = NAME_TO_ID[agent_name]["layer"]
                    matched_name = agent_name
                else:
                    for known_name, info in NAME_TO_ID.items():
                        if known_name in agent_name or agent_name in known_name:
                            matched_id = info["id"]
                            matched_layer = info["layer"]
                            matched_name = known_name
                            break

                if matched_id:
                    self.current_agent_id = matched_id
                    self.current_agent_name = matched_name
                    self.current_layer = matched_layer
                    self.capturing_output = True
                    self.expecting_agent_name = False
                    # v4: 推送 data_flow 事件（上一个 Agent → 当前 Agent）
                    broadcast_event("data_flow", matched_id, matched_name, matched_layer,
                                   message=f"📤 数据流入 → {matched_name}")
                    broadcast_event("agent_start", matched_id, matched_name, matched_layer,
                                   message=f"🔄 {matched_name} 开始分析...")
                    agent_state["progress"] = f"{matched_name} 正在分析..."
                else:
                    self.expecting_agent_name = False

        # ── 检测 Agent Final Answer / Task Completed ──
        if self.current_agent_id and (self.capturing_output or self.agent_output_buffer):
            fa_patterns = ['Agent Final Answer', 'Final Answer:', 'Task Completed', 'Task Completion']
            completed = any(p in self.buffer for p in fa_patterns)
            if completed:
                output_preview = self.agent_output_buffer[:1200] if self.agent_output_buffer else ""
                # v4: 推送层间数据流动事件
                broadcast_event("agent_done", self.current_agent_id, self.current_agent_name,
                               self.current_layer, message=f"✅ {self.current_agent_name} 分析完成",
                               output_preview=output_preview)
                broadcast_event("data_flow_out", self.current_agent_id, self.current_agent_name,
                               self.current_layer, message=f"📤 {self.current_agent_name} 输出数据")
                agent_state["progress"] = f"{self.current_agent_name} ✅ 完成"
                self.capturing_output = False
                self.agent_output_buffer = ""
                self.current_agent_id = None
                self.current_agent_name = None

        # ── 捕获 Agent 输出内容 ──
        if self.capturing_output and self.current_agent_id:
            clean = re.sub(r'[╭╰╯╮│┃┆┊┏┓┗┛┣┫┳┻╋─━┄┅┈┉]', '', text)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean and len(clean) > 5 and not clean.startswith('Agent:') and 'Task Started' not in clean:
                self.agent_output_buffer += clean + "\n"

            # v4: 每 2 秒推送一次中间输出
            if now - self._last_progress_push >= 2.0:
                self._last_progress_push = now
                preview = self.agent_output_buffer[-600:] if self.agent_output_buffer else ""
                if preview and self.current_agent_id:
                    broadcast_event("agent_output", self.current_agent_id, self.current_agent_name,
                                   self.current_layer, message=f"💭 {self.current_agent_name} 生成中...",
                                   output_preview=preview)

        # ── 解析否决关键词 ──
        if self.current_agent_id == "constraint_veto" and self.capturing_output:
            if any(kw in text for kw in ['否决', '一票否决', '不通过', 'VETO', 'veto', '❌']):
                broadcast_event("veto", "constraint_veto", "硬约束否决官", 1.5,
                               message="⛔ 硬约束否决！基金未通过技术指标检查")
                agent_state["agent_statuses"]["constraint_veto"] = "vetoed"

        # ── 辩论层事件 v4：推送实质内容 ──
        if self.current_agent_id == "bullish_researcher" and self.capturing_output:
            self._debate_buffer["bull"] = self.agent_output_buffer
            if now - self._last_debate_push["bull"] >= 3.0:
                self._last_debate_push["bull"] = now
                content = self._extract_debate_points(self.agent_output_buffer, "bull")
                broadcast_event("debate_bull", "bullish_researcher", "多头研究员", 3,
                               message=content or "🐂 多头正在构建买入论点...")
        elif self.current_agent_id == "bearish_researcher" and self.capturing_output:
            self._debate_buffer["bear"] = self.agent_output_buffer
            if now - self._last_debate_push["bear"] >= 3.0:
                self._last_debate_push["bear"] = now
                content = self._extract_debate_points(self.agent_output_buffer, "bear")
                broadcast_event("debate_bear", "bearish_researcher", "空头研究员", 3,
                               message=content or "🐻 空头正在识别风险因素...")
        elif self.current_agent_id == "debate_judge" and self.capturing_output:
            self._debate_buffer["judge"] = self.agent_output_buffer
            if now - self._last_debate_push["judge"] >= 3.0:
                self._last_debate_push["judge"] = now
                content = self._extract_debate_points(self.agent_output_buffer, "judge")
                broadcast_event("judge", "debate_judge", "辩论裁决官", 3,
                               message=content or "⚖️ 裁决官正在评估多空双方论点...")

        # ── 限制 buffer 大小 ──
        if len(self.buffer) > 8000:
            self.buffer = self.buffer[-3000:]
            self._last_agent_started_pos = max(0, self._last_agent_started_pos - 5000)

    def _extract_debate_points(self, text, side):
        """从 Agent 输出中提取辩论要点"""
        if not text or len(text) < 20:
            return ""
        # 提取带编号的论点（1. 2. 3. 或 一、二、三、）
        points = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            # 跳过 CrewAI 框线残留
            if re.match(r'^[╭╰╯╮│┃┆┊┏┓┗┛┣┫┳┻╋─━┄┅┈┉\s]+$', line):
                continue
            # 匹配编号论点
            if re.match(r'^(\d+[\.\)、]|[一二三四五六七八九十]+[、）])', line):
                points.append(line)
            # 匹配关键词开头的句子
            elif side == "bull" and any(kw in line for kw in ['上涨', '买入', '看好', '利好', '增长', '突破', '反弹', '支撑', '机会']):
                points.append(line)
            elif side == "bear" and any(kw in line for kw in ['下跌', '风险', '看空', '利空', '回调', '压力', '警示', '危险', '损失']):
                points.append(line)
            elif side == "judge" and any(kw in line for kw in ['综合', '评估', '裁决', '权衡', '结论', '建议', '判断']):
                points.append(line)

        if not points:
            # 没有结构化论点，取最后 200 字
            return text[-200:].strip()
        return '\n'.join(points[:5])

    def flush(self):
        self.original.flush()


# ============================================================
# 快速分析功能
# ============================================================
def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def quick_screen(top_n=10):
    """快速筛选基金"""
    from fund_data import screen_funds, quantitative_score, check_hard_constraints
    import pandas as pd

    df = screen_funds(top_n=top_n)
    results = []
    for _, row in df.iterrows():
        code = str(row.get("基金代码", "")).zfill(6)
        name = str(row.get("基金简称", ""))
        fund_type = str(row.get("基金类型", ""))
        try:
            from fund_data import fetch_fund_nav_history, calc_technical_indicators
            nav_data = fetch_fund_nav_history(code, days=120)
            if nav_data is None or len(nav_data) < 30:
                continue
            indicators = calc_technical_indicators(nav_data)
            constraints = check_hard_constraints(indicators)
            score_result = quantitative_score(indicators)
            if isinstance(score_result, dict):
                quant_score = _safe_float(score_result.get("weighted_total", 0))
                hard_passed = score_result.get("hard_constraints", {}).get("passed", False)
                veto_reasons = score_result.get("hard_constraints", {}).get("vetoes", [])
            else:
                quant_score = _safe_float(score_result)
                hard_passed = constraints.get("passed", False) if isinstance(constraints, dict) else False
                veto_reasons = constraints.get("vetoes", []) if isinstance(constraints, dict) else []
            rsi_val = _safe_float(indicators.get("rsi_6"), 0)
            vol_val = _safe_float(indicators.get("volatility"), 0)
            results.append({
                "code": code, "name": name, "type": fund_type,
                "return_1m": _safe_float(row.get("近1月"), 0) if pd.notna(row.get("近1月")) else 0,
                "return_3m": _safe_float(row.get("近3月"), 0) if pd.notna(row.get("近3月")) else 0,
                "return_6m": _safe_float(row.get("近6月"), 0) if pd.notna(row.get("近6月")) else 0,
                "return_ytd": _safe_float(row.get("今年来"), 0) if pd.notna(row.get("今年来")) else 0,
                "rsi": round(rsi_val, 1) if rsi_val else 0,
                "trend": indicators.get("trend", "unknown") or "unknown",
                "volatility": round(vol_val * 100, 1) if vol_val else 0,
                "macd_signal": indicators.get("macd_signal", "unknown") or "unknown",
                "quant_score": round(quant_score, 2),
                "constraints_pass": hard_passed,
                "constraints_reasons": veto_reasons if isinstance(veto_reasons, list) else [str(veto_reasons)],
            })
        except Exception as e:
            results.append({"code": code, "name": name, "type": fund_type, "error": str(e)})
    passed = [r for r in results if r.get("constraints_pass", False) and r.get("quant_score", 0) >= 6.5]
    passed.sort(key=lambda x: x.get("quant_score", 0), reverse=True)
    vetoed = [r for r in results if not r.get("constraints_pass", False) and "error" not in r]
    return {"passed": passed, "vetoed": vetoed}


def quick_analyze(fund_code, days=120):
    """快速分析单只基金"""
    from fund_data import (
        fetch_fund_nav_history, calc_technical_indicators,
        quantitative_score, check_hard_constraints,
        fetch_fund_details, fetch_fund_holdings,
    )
    import pandas as pd

    code = str(fund_code).zfill(6)
    nav_data = fetch_fund_nav_history(code, days=days)
    if nav_data is None or len(nav_data) < 30:
        return {"error": f"无法获取基金 {code} 的净值数据"}

    indicators = calc_technical_indicators(nav_data)
    constraints = check_hard_constraints(indicators)
    score_result = quantitative_score(indicators)

    if isinstance(score_result, dict):
        quant_score = _safe_float(score_result.get("weighted_total", 0))
        hard_passed = score_result.get("hard_constraints", {}).get("passed", False)
        veto_reasons = score_result.get("hard_constraints", {}).get("vetoes", [])
        score_detail = score_result.get("scores", {})
    else:
        quant_score = _safe_float(score_result)
        hard_passed = constraints.get("passed", False) if isinstance(constraints, dict) else False
        veto_reasons = constraints.get("vetoes", []) if isinstance(constraints, dict) else []
        score_detail = {}

    details = None
    try:
        d = fetch_fund_details(code)
        if isinstance(d, dict):
            details = {}
            for k, v in d.items():
                try:
                    if isinstance(v, type(pd.NA)) or (v is not None and pd.isna(v)):
                        details[k] = None
                    else:
                        details[k] = v
                except:
                    details[k] = v if v is not None else None
    except:
        pass

    holdings = None
    try:
        h = fetch_fund_holdings(code)
        if h is not None and hasattr(h, 'to_dict'):
            holdings = h.to_dict(orient='records')
    except:
        pass

    recent_nav = []
    for _, row in nav_data.tail(5).iterrows():
        recent_nav.append({
            "date": str(row.get("净值日期", "")),
            "nav": _safe_float(row.get("单位净值", 0)),
            "change": _safe_float(row.get("日增长率", 0)),
        })

    return {
        "code": code, "details": details,
        "indicators": {
            "rsi_6": round(_safe_float(indicators.get("rsi_6")), 1),
            "rsi_14": round(_safe_float(indicators.get("rsi_14")), 1),
            "trend": indicators.get("trend", "unknown") or "unknown",
            "macd_signal": indicators.get("macd_signal", "unknown") or "unknown",
            "volatility": round(_safe_float(indicators.get("volatility")) * 100, 1),
            "sharpe": round(_safe_float(indicators.get("sharpe_ratio")), 2),
            "max_drawdown": round(_safe_float(indicators.get("max_drawdown")) * 100, 1),
            "ma5": round(_safe_float(indicators.get("ma5")), 4),
            "ma20": round(_safe_float(indicators.get("ma20")), 4),
            "ma60": round(_safe_float(indicators.get("ma60")), 4),
            "current_nav": round(_safe_float(indicators.get("current_nav")), 4),
        },
        "quant_score": round(quant_score, 2),
        "score_detail": score_detail,
        "constraints": {
            "pass": hard_passed,
            "reasons": veto_reasons if isinstance(veto_reasons, list) else [str(veto_reasons)],
        },
        "recent_nav": recent_nav, "holdings": holdings,
    }


# ============================================================
# Agent 执行 v4 — 显式回调 + SSE 推送（替代 stdout 拦截）
# ============================================================
def run_full_agent(fund_codes=None, fast_mode=False):
    """在后台线程运行完整 CrewAI Agent（通过显式回调推送进度）"""
    global agent_state

    # 给前端 1.5 秒时间建立 SSE 连接，避免早期事件丢失
    time.sleep(1.5)

    # 清空旧的事件缓存，确保新运行不会收到上一轮的事件
    with recent_events_lock:
        recent_events_cache.clear()

    # 重置 Agent 状态（running/start_time 已在 api_agent_run 中设置）
    agent_state["result"] = None
    agent_state["error"] = None
    agent_state["current_layer"] = 0
    agent_state["current_agent"] = ""
    agent_state["agent_statuses"] = {a["id"]: "idle" for layer in AGENT_LAYERS for a in layer["agents"]}
    agent_state["agent_outputs"] = {}

    # 用安全 stdout 包装，防止后台线程 print 触发 BrokenPipeError
    old_stdout = sys.stdout
    sys.stdout = SafeStdout(old_stdout)

    mode = "指定基金分析" if fund_codes else "全市场扫描"
    target = f"目标: {fund_codes}" if fund_codes else "目标: 全市场Top50"

    broadcast_event("system_start", "system", "系统", 0,
                   message=f"🚀 {mode}启动 — {target}")

    # ── 心跳线程：每 3 秒推送运行状态，让前端知道系统还活着 ──
    _heartbeat_stop = threading.Event()
    def heartbeat_loop():
        while not _heartbeat_stop.is_set():
            _heartbeat_stop.wait(3)
            if _heartbeat_stop.is_set():
                break
            try:
                start = datetime.fromisoformat(agent_state.get("start_time", ""))
                elapsed = int((datetime.now() - start).total_seconds())
            except:
                elapsed = 0
            current = agent_state.get("current_agent", "")
            broadcast_event("heartbeat", "system", "系统", 0,
                           message=f"⏱️ 已运行 {elapsed}s — 当前: {current or '初始化...'}",
                           output_preview=json.dumps({"elapsed": elapsed, "current_agent": current}))
    ht = threading.Thread(target=heartbeat_loop, daemon=True)
    ht.start()

    def progress_callback(event_type, agent_id, agent_name, layer, message, output_preview=""):
        """由 main.py 调用的进度回调"""
        # 更新状态
        agent_state["current_agent"] = agent_name
        agent_state["current_layer"] = layer
        agent_state["progress"] = message

        if event_type == "agent_start":
            agent_state["agent_statuses"][agent_id] = "running"
            broadcast_event("agent_start", agent_id, agent_name, layer, message=message)
        elif event_type == "agent_done":
            agent_state["agent_statuses"][agent_id] = "done"
            if output_preview:
                agent_state["agent_outputs"][agent_id] = output_preview
            broadcast_event("agent_done", agent_id, agent_name, layer,
                           message=message, output_preview=output_preview)
            # 层间数据流动画
            broadcast_event("data_flow_out", agent_id, agent_name, layer,
                           message=f"📤 {agent_name} 输出数据")
        elif event_type == "layer_start":
            broadcast_event("layer_start", agent_id, agent_name, layer, message=message)
        elif event_type == "debate_bull":
            broadcast_event("debate_bull", "bullish_researcher", "多头研究员", 3, message=message)
        elif event_type == "debate_bear":
            broadcast_event("debate_bear", "bearish_researcher", "空头研究员", 3, message=message)
        elif event_type == "judge":
            broadcast_event("judge", "debate_judge", "辩论裁决官", 3, message=message)
        elif event_type == "veto":
            agent_state["agent_statuses"]["constraint_veto"] = "vetoed"
            broadcast_event("veto", "constraint_veto", "硬约束否决官", 1.5, message=message)
        elif event_type == "progress":
            # 预加载进度事件 — 直接透传到前端
            broadcast_event("progress", agent_id, agent_name, layer, message=message)
        elif event_type == "recommendation":
            # 结构化投资建议 — 存储并推送
            agent_state["recommendation"] = output_preview
            broadcast_event("recommendation", agent_id, agent_name, layer,
                           message=message, output_preview=output_preview)

    try:
        import main as agent_main

        if fund_codes:
            agent_state["progress"] = f"指定基金模式 ({fund_codes}) — Agent 围绕目标基金分析中..."
        elif fast_mode:
            agent_state["progress"] = "快速模式：5 Agent 精简流水线，预计 3-5 分钟..."
        else:
            agent_state["progress"] = "全市场扫描执行中（预计 5-8 分钟）..."

        # ── 用 concurrent.futures 给 crew.kickoff 加超时 ──
        import concurrent.futures
        TIMEOUT_SECONDS = 600  # 10 分钟总超时

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                agent_main.run_agent,
                fund_codes=fund_codes,
                progress_callback=progress_callback,
                fast_mode=fast_mode,
            )
            try:
                result = future.result(timeout=TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                broadcast_event("error", "system", "系统", 0,
                               message=f"❌ 分析超时（{TIMEOUT_SECONDS}s），部分 Agent 可能未响应。请尝试快速模式或指定单只基金。")
                agent_state["error"] = "Timeout"
                agent_state["progress"] = f"分析超时（{TIMEOUT_SECONDS}s）"
                # 超时后不等待线程，直接退出
                return

        # 完成
        broadcast_event("complete", "system", "系统", 0,
                       message="✅ 全部分析完成",
                       output_preview=str(result)[:2000])
        agent_state["result"] = str(result)
        agent_state["progress"] = "分析完成！"

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        broadcast_event("error", "system", "系统", 0,
                       message=f"❌ 执行失败: {str(e)}")
        agent_state["error"] = error_msg
        agent_state["progress"] = f"执行失败: {str(e)}"
    finally:
        _heartbeat_stop.set()  # 停止心跳线程
        agent_state["running"] = False
        # 恢复原始 stdout
        sys.stdout = old_stdout


# ============================================================
# API 路由
# ============================================================
@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")


@app.route("/api/screen", methods=["POST"])
def api_screen():
    data = request.json or {}
    top_n = data.get("top_n", 10)
    try:
        result = quick_screen(top_n=top_n)
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    fund_code = data.get("fund_code", "")
    days = data.get("days", 120)
    if not fund_code:
        return jsonify({"ok": False, "error": "请提供基金代码"})
    try:
        result = quick_analyze(fund_code, days=days)
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/agent/reset", methods=["POST"])
def api_agent_reset():
    """强制重置 Agent 状态（解决卡死问题）"""
    global agent_state
    agent_state["running"] = False
    agent_state["result"] = None
    agent_state["error"] = None
    agent_state["current_layer"] = 0
    agent_state["current_agent"] = ""
    agent_state["agent_statuses"] = {a["id"]: "idle" for layer in AGENT_LAYERS for a in layer["agents"]}
    agent_state["agent_outputs"] = {}
    agent_state["progress"] = "已重置"
    # 清空事件缓存
    with recent_events_lock:
        recent_events_cache.clear()
    broadcast_event("system", "system", "系统", 0, message="🔄 Agent 状态已强制重置")
    return jsonify({"ok": True, "message": "Agent 状态已重置"})


@app.route("/api/agent/run", methods=["POST"])
def api_agent_run():
    with agent_start_lock:
        # 如果 running 超过 10 分钟，视为卡死，允许强制启动
        if agent_state["running"]:
            try:
                start = datetime.fromisoformat(agent_state["start_time"])
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed < 600:  # 10分钟内，真在跑
                    return jsonify({"ok": False, "error": "Agent 正在运行中，请等待完成（或点强制重置）"})
            except:
                pass
            # 超过10分钟或解析失败，视为僵尸状态，自动重置
            agent_state["running"] = False

        # 立即标记为运行中 + 记录启动时间，防止并发
        agent_state["running"] = True
        agent_state["start_time"] = datetime.now().isoformat()
        agent_state["error"] = None
        agent_state["result"] = None

    data = request.json or {}
    fund_codes = data.get("fund_codes", None)
    fast_mode = bool(data.get("fast_mode", False))
    thread = threading.Thread(target=run_full_agent, args=(fund_codes, fast_mode), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Agent 已启动", "fast_mode": fast_mode})


@app.route("/api/agent/status", methods=["GET"])
def api_agent_status():
    return jsonify(agent_state)


@app.route("/api/agent/events/stream", methods=["GET"])
def api_agent_events_sse():
    """SSE 实时事件流 — 前端用 EventSource 连接"""
    q = Queue(maxsize=200)

    # 先发送缓存的最近事件，让新客户端不错过早期事件
    with recent_events_lock:
        cached = list(recent_events_cache)
    for evt in cached:
        try:
            q.put_nowait(evt)
        except:
            break

    with sse_lock:
        sse_clients.append(q)

    def generate():
        try:
            while True:
                try:
                    # 短超时：客户端断开后线程能快速退出
                    event = q.get(timeout=5)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except Empty:
                    # 发送心跳保持连接
                    yield f": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                try:
                    sse_clients.remove(q)
                except ValueError:
                    pass

    return Response(generate(), mimetype="text/event-stream",
                   headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/agent/layers", methods=["GET"])
def api_agent_layers():
    return jsonify({"ok": True, "layers": AGENT_LAYERS})


@app.route("/api/reports", methods=["GET"])
def api_reports():
    reports = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(os.listdir(base_dir)):
        if f.startswith("fund_trade_report") and f.endswith(".md"):
            filepath = os.path.join(base_dir, f)
            mtime = os.path.getmtime(filepath)
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
            reports.append({
                "filename": f,
                "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                "size": len(content),
                "preview": content[:200],
            })
    return jsonify({"ok": True, "data": reports})


@app.route("/api/report/<name>", methods=["GET"])
def api_report_detail(name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, name)
    if not os.path.exists(filepath):
        return jsonify({"ok": False, "error": "报告不存在"})
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"ok": True, "data": {"filename": name, "content": content}})


@app.route("/api/feedback", methods=["GET"])
def api_feedback():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "recommendation_log.json")
    if not os.path.exists(log_path):
        return jsonify({"ok": True, "data": []})
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"ok": True, "data": data})


@app.route("/api/feedback/track", methods=["POST"])
def api_feedback_track():
    try:
        from feedback import auto_track_and_review
        result = auto_track_and_review()
        return jsonify({"ok": True, "tracked_count": result["tracked_count"],
                       "optimization": result["optimization"], "report": result["report"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/feedback/optimization", methods=["GET"])
def api_feedback_optimization():
    try:
        from feedback import auto_track_and_review, load_scoring_weights
        result = auto_track_and_review()
        weights = load_scoring_weights()
        return jsonify({"ok": True, "optimization": result["optimization"],
                       "current_weights": weights, "report": result["report"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/feedback/performance", methods=["GET"])
def api_feedback_performance():
    try:
        from feedback import get_performance_table
        table = get_performance_table()
        return jsonify({"ok": True, "data": table})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_static")
    os.makedirs(static_dir, exist_ok=True)
    port = int(os.environ.get("PORT", 5001))
    print("=" * 50, flush=True)
    print("  Fund Trade Agent Web UI v3", flush=True)
    print(f"  访问地址: http://0.0.0.0:{port}", flush=True)
    print("=" * 50, flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
