"""Streamlit 演示看板 —— 初赛「可运行链接」交付的核心载体。

三个 Tab：
1. **单日决策**：跑一次完整的多 Agent 决策链路 —— 录制演示视频的主舞台。
2. **回测**：在历史区间跑多 Agent 团队，输出净值曲线与绩效指标。
3. **数据探索**：K线 + 技术指标 + 资金流 + 新闻 + 公司基本面。

启动：
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# 让本脚本无需安装即可运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from aifund import indicators
from aifund.agents import AgentTeam
from aifund.backtest import BacktestEngine, Portfolio
from aifund.data import DataPipeline, sources
from aifund.data.models import MarketSnapshot
from aifund.output import build_markdown_report, orders_to_competition_payload
from config.settings import settings

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="智投未来 · 多智能体 A股投研",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 视觉系统：Bloomberg Terminal 风（深炭灰 + 古金色 + 等宽数字）
# ---------------------------------------------------------------------------

# 品牌色（与 plotly 图表共享）
COLOR_BG = "#0a0d14"
COLOR_CARD = "#141821"
COLOR_BORDER = "#1f2433"
COLOR_GOLD = "#d4af37"
COLOR_GOLD_LIGHT = "#f5d76e"
COLOR_TEXT = "#e8e8e8"
COLOR_MUTED = "#8a93a8"
COLOR_UP = "#ff3b30"  # 中国市场习惯：涨红
COLOR_DOWN = "#34c759"  # 跌绿
COLOR_CYAN = "#5ac8fa"

st.markdown(
    f"""
    <style>
    /* ---- 字体导入 ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    /* ---- 全局 ---- */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        letter-spacing: 0.2px;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* ---- 数字一律等宽，更专业 ---- */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: {COLOR_GOLD_LIGHT} !important;
        letter-spacing: -0.5px;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {COLOR_MUTED} !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }}

    /* ---- Metric 卡片：深色 + 金色左边线 ---- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {COLOR_CARD} 0%, #0f131c 100%);
        border: 1px solid {COLOR_BORDER};
        border-left: 3px solid {COLOR_GOLD};
        padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        border-radius: 4px;
        transition: border-color 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        border-left-color: {COLOR_GOLD_LIGHT};
    }}

    /* ---- 标题 ---- */
    h1, h2, h3 {{
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }}
    h1 {{
        color: {COLOR_TEXT};
        font-size: 1.8rem !important;
    }}
    h2 {{
        color: {COLOR_TEXT};
        font-size: 1.25rem !important;
        border-bottom: 1px solid {COLOR_BORDER};
        padding-bottom: 0.4rem;
        margin-top: 1.8rem !important;
    }}
    h3 {{
        color: {COLOR_GOLD_LIGHT} !important;
        font-size: 1.0rem !important;
    }}

    /* ---- 按钮：金色渐变 + 黑色字 ---- */
    button[kind="primary"] {{
        background: linear-gradient(135deg, {COLOR_GOLD} 0%, #b8941f 100%) !important;
        color: #0a0d14 !important;
        font-weight: 700 !important;
        border: 1px solid {COLOR_GOLD} !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.15s ease;
    }}
    button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {COLOR_GOLD_LIGHT} 0%, #d4af37 100%) !important;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.4);
    }}

    /* ---- 侧边栏 ---- */
    [data-testid="stSidebar"] {{
        background: #07090f !important;
        border-right: 1px solid {COLOR_BORDER};
    }}
    [data-testid="stSidebar"] .stTitle, [data-testid="stSidebar"] h1 {{
        color: {COLOR_GOLD_LIGHT} !important;
        letter-spacing: 2px;
        font-size: 1.1rem !important;
    }}

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {COLOR_MUTED} !important;
        font-weight: 500;
        letter-spacing: 0.5px;
        padding: 0.6rem 1.4rem !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLOR_GOLD_LIGHT} !important;
        border-bottom: 2px solid {COLOR_GOLD} !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {COLOR_TEXT} !important;
    }}

    /* ---- DataFrame 表格 ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
    }}

    /* ---- 输入框 ---- */
    input, textarea, .stTextInput input, .stNumberInput input, .stDateInput input {{
        background: {COLOR_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ---- Expander ---- */
    details {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 4px;
        margin: 0.4rem 0;
    }}
    details summary {{
        color: {COLOR_GOLD_LIGHT} !important;
        font-weight: 500;
    }}

    /* ---- 代码块 ---- */
    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
        background: {COLOR_BG} !important;
    }}

    /* ---- 自定义类 ---- */
    .small-muted {{ color: {COLOR_MUTED}; font-size: 0.8rem; letter-spacing: 0.5px; }}
    .hero-banner {{
        background: linear-gradient(135deg, {COLOR_CARD} 0%, #0a0d14 60%, #0a0d14 100%);
        border: 1px solid {COLOR_BORDER};
        border-left: 4px solid {COLOR_GOLD};
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
    }}
    .hero-banner::after {{
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 200px; height: 100%;
        background: radial-gradient(circle at 100% 50%, rgba(212, 175, 55, 0.08), transparent 70%);
        pointer-events: none;
    }}
    .brand-name {{
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: 6px;
        color: {COLOR_GOLD_LIGHT};
        margin: 0;
        line-height: 1;
    }}
    .brand-en {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: {COLOR_MUTED};
        letter-spacing: 3px;
        margin: 0.4rem 0 0 0;
        text-transform: uppercase;
    }}
    .brand-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: {COLOR_GOLD};
        letter-spacing: 1.5px;
        text-align: right;
        margin: 0;
    }}
    .brand-sub {{
        font-size: 0.72rem;
        color: {COLOR_MUTED};
        letter-spacing: 1.2px;
        text-align: right;
        margin: 0.3rem 0 0 0;
    }}
    .info-pill {{
        display: inline-block;
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        padding: 0.15rem 0.7rem;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: {COLOR_MUTED};
        margin-right: 0.5rem;
    }}
    .info-pill .v {{ color: {COLOR_GOLD_LIGHT}; font-weight: 600; }}

    /* 涨跌颜色（红涨绿跌，中国习惯） */
    .up {{ color: {COLOR_UP}; font-weight: 600; }}
    .down {{ color: {COLOR_DOWN}; font-weight: 600; }}
    .neutral {{ color: {COLOR_MUTED}; }}

    /* ---- 自定义表格（分析师矩阵 / 订单清单）---- */
    table.terminal-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        margin: 0.2rem 0 0.8rem 0;
    }}
    table.terminal-table thead th {{
        color: {COLOR_MUTED};
        text-align: left;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid {COLOR_BORDER};
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1.2px;
        font-family: 'Inter', sans-serif;
    }}
    table.terminal-table tbody td {{
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        color: {COLOR_TEXT};
        vertical-align: top;
    }}
    table.terminal-table tbody tr:last-child td {{
        border-bottom: none;
    }}
    table.terminal-table tbody tr:hover td {{
        background: rgba(212, 175, 55, 0.04);
    }}
    table.terminal-table td.mono, table.terminal-table td.num {{
        font-family: 'JetBrains Mono', monospace;
    }}
    table.terminal-table td.num {{
        text-align: right;
    }}
    table.terminal-table td.reason {{
        color: #b8bdc9;
        font-size: 0.82rem;
        max-width: 480px;
    }}
    table.terminal-table td.agent {{
        color: {COLOR_GOLD_LIGHT};
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 缓存的资源构造器
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="初始化数据管道…")
def get_pipeline(lookback_days: int = 120) -> DataPipeline:
    return DataPipeline(lookback_days=lookback_days)


@st.cache_resource(show_spinner="初始化 Agent 团队…")
def get_team() -> AgentTeam:
    return AgentTeam(log_dir=settings.paths.runs / "dashboard")


@st.cache_data(show_spinner="拉取行情…")
def fetch_price(symbol: str, start: date, end: date, asset_type: str = "stock") -> pd.DataFrame:
    return sources.get_price_history(symbol, start, end, asset_type=asset_type)


@st.cache_data(show_spinner="拉取新闻 / 资金流 / 基本面…")
def fetch_snapshot_cached(symbols: tuple[str, ...], as_of: date) -> MarketSnapshot:
    pipeline = get_pipeline()
    return pipeline.snapshot(list(symbols), as_of)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 智投未来")
    st.caption("多智能体 A股投研系统 · 驼灵大赛参赛作品")
    st.markdown("---")

    api_ok = bool(settings.llm.deepseek_api_key)
    if api_ok:
        st.success(f"✓ DeepSeek 已配置 ({settings.llm.deepseek_api_key[:6]}…)")
        st.caption(f"分析师：`{settings.llm.deepseek_fast_model}`")
        st.caption(f"基金经理：`{settings.llm.deepseek_deep_model}`")
    else:
        st.error("✗ DeepSeek API key 未配置")
        st.caption("请在项目根目录的 `.env` 填入 `DEEPSEEK_API_KEY`")

    st.markdown("---")
    st.caption(f"初始资金：{settings.backtest.initial_capital:,.0f} 元")
    st.caption(f"单票上限：{settings.backtest.max_position_per_stock:.0%}")
    st.caption(f"总仓上限：{settings.backtest.max_total_position:.0%}")


# ---------------------------------------------------------------------------
# 顶部 Hero Banner
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero-banner">
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
          <h1 class="brand-name">智 投 未 来</h1>
          <p class="brand-en">ZHI·TOU · MULTI-AGENT A-SHARE INVESTMENT RESEARCH</p>
          <div style="margin-top: 0.9rem;">
            <span class="info-pill">SYSTEM <span class="v">ONLINE</span></span>
            <span class="info-pill">ANALYSTS <span class="v">×4 · V3</span></span>
            <span class="info-pill">MANAGER <span class="v">R1</span></span>
            <span class="info-pill">CAPITAL <span class="v">¥{settings.backtest.initial_capital:,.0f}</span></span>
          </div>
        </div>
        <div style="min-width: 200px;">
          <p class="brand-tag">驼灵智能体大赛 / 2026</p>
          <p class="brand-sub">金融投资赛道 · 首都经济贸易大学</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_decide, tab_backtest, tab_explore = st.tabs(["🎯 单日决策", "📊 历史回测", "🔍 数据探索"])


# ---------------------------------------------------------------------------
# 通用绘图：K线 + 均线 + MACD + 成交量
# ---------------------------------------------------------------------------


def _html_escape(s: str) -> str:
    """轻量 HTML 转义，避免分析师理由里的特殊字符破坏布局。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_opinion_matrix(opinions) -> str:
    """把分析师意见列表渲染为金融终端风的 HTML 表。"""
    rows: list[str] = [
        '<table class="terminal-table"><thead><tr>',
        "<th>分析师</th><th>方向</th><th>评分</th><th>置信度</th><th>核心理由</th>",
        "</tr></thead><tbody>",
    ]
    for op in opinions:
        action_cls = {"看多": "up", "看空": "down"}.get(op.action, "neutral")
        action_arrow = {"看多": "▲ 看多", "看空": "▼ 看空"}.get(op.action, "● 中性")
        score_cls = "up" if op.score > 0 else ("down" if op.score < 0 else "neutral")
        rows.append(
            f'<tr>'
            f'<td class="agent">{_html_escape(op.agent_name)}</td>'
            f'<td class="{action_cls}"><b>{action_arrow}</b></td>'
            f'<td class="num {score_cls}">{op.score:+.2f}</td>'
            f'<td class="num">{op.confidence:.2f}</td>'
            f'<td class="reason">{_html_escape(op.reasoning)}</td>'
            f'</tr>'
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def render_orders_table(orders) -> str:
    """把订单列表渲染为金融终端风的 HTML 表。"""
    rows: list[str] = [
        '<table class="terminal-table"><thead><tr>',
        "<th>方向</th><th>代码</th><th>名称</th><th>股数</th><th>理由</th>",
        "</tr></thead><tbody>",
    ]
    for o in orders:
        side_cls = "up" if o.side == "BUY" else "down"
        side_text = "▲ 买入" if o.side == "BUY" else "▼ 卖出"
        rows.append(
            f'<tr>'
            f'<td class="{side_cls}"><b>{side_text}</b></td>'
            f'<td class="mono">{_html_escape(o.symbol)}</td>'
            f'<td>{_html_escape(o.name or "—")}</td>'
            f'<td class="num">{o.shares:,}</td>'
            f'<td class="reason">{_html_escape(o.reason or "")}</td>'
            f'</tr>'
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _apply_terminal_theme(fig: go.Figure, height: int = 620) -> go.Figure:
    """统一施加金融终端风：深炭灰底 + 浅金标题 + 极淡网格。"""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT, size=12),
        title=dict(font=dict(color=COLOR_GOLD_LIGHT)),
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR_MUTED, size=11),
        ),
        hoverlabel=dict(bgcolor=COLOR_CARD, bordercolor=COLOR_GOLD, font_size=12,
                        font_family="JetBrains Mono"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_MUTED, size=10, family="JetBrains Mono"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_MUTED, size=10, family="JetBrains Mono"),
    )
    return fig


def render_price_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    """绘制 3 子图：K线 + 均线 / MACD / 成交量（金融终端配色）。"""
    enriched = indicators.compute(df) if not df.empty else df

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.20, 0.22],
        vertical_spacing=0.025,
        subplot_titles=(title, "MACD", "VOLUME"),
    )

    if enriched.empty:
        return _apply_terminal_theme(fig)

    x = enriched["date"]

    # K 线（涨红 / 跌绿 —— 中国市场习惯）
    fig.add_trace(go.Candlestick(
        x=x, open=enriched["open"], high=enriched["high"],
        low=enriched["low"], close=enriched["close"],
        name="K", increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        line=dict(width=1),
    ), row=1, col=1)

    # 均线：金色系 + 蓝青色，避免与 K 线红绿打架
    ma_colors = {
        "ma5":  ("#f5d76e", 1.4),  # 浅金
        "ma10": ("#d4af37", 1.2),  # 古金
        "ma20": (COLOR_CYAN, 1.0),  # 青
        "ma60": ("#8a93a8", 0.9),  # 灰
    }
    for col, (color, width) in ma_colors.items():
        if col in enriched.columns and enriched[col].notna().any():
            fig.add_trace(go.Scatter(
                x=x, y=enriched[col], name=col.upper(),
                line=dict(width=width, color=color), opacity=0.9,
            ), row=1, col=1)

    # MACD：柱状图涨红跌绿，DIF/DEA 金色/青色
    if "macd_hist" in enriched.columns:
        colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in enriched["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=x, y=enriched["macd_hist"], name="HIST",
            marker_color=colors, marker_line_width=0, opacity=0.85,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=enriched["macd_dif"], name="DIF",
                                 line=dict(color=COLOR_GOLD_LIGHT, width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=enriched["macd_dea"], name="DEA",
                                 line=dict(color=COLOR_CYAN, width=1.0)), row=2, col=1)

    # 成交量：按当日涨跌着色（K 线同色逻辑）
    if "volume" in enriched.columns and enriched["volume"].notna().any():
        up_mask = enriched["close"] >= enriched["open"]
        vol_colors = [COLOR_UP if up else COLOR_DOWN for up in up_mask]
        fig.add_trace(go.Bar(
            x=x, y=enriched["volume"], name="VOL",
            marker_color=vol_colors, marker_line_width=0, opacity=0.75,
        ), row=3, col=1)

    fig = _apply_terminal_theme(fig, height=640)
    # 给子图标题上金色
    for ann in fig.layout.annotations:
        ann.font.color = COLOR_GOLD_LIGHT
        ann.font.size = 11
        ann.font.family = "JetBrains Mono"
    return fig


# ---------------------------------------------------------------------------
# Tab 1 · 单日决策
# ---------------------------------------------------------------------------


with tab_decide:
    st.subheader("🎯 跑一次完整的多 Agent 决策链路")
    st.caption("展示数据 → 4 位分析师 → 基金经理 → 标准 JSON 的端到端流程。")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        decide_date = st.date_input("决策基准日", value=date(2026, 5, 13),
                                    min_value=date(2010, 1, 4), max_value=date.today())
    with c2:
        decide_symbols = st.text_input("候选标的代码（逗号分隔）",
                                       value="600519,601318",
                                       help="A股 6 位代码，逗号分隔；最多 5 只")
    with c3:
        decide_capital = st.number_input("初始资金", min_value=10000.0,
                                         value=float(settings.backtest.initial_capital),
                                         step=50_000.0)

    run_decide = st.button("🚀 启动决策", type="primary", disabled=not api_ok,
                           use_container_width=False)

    if run_decide:
        symbols = tuple(s.strip() for s in decide_symbols.replace("，", ",").split(",") if s.strip())[:5]
        if not symbols:
            st.error("请输入至少一个标的代码")
        else:
            with st.spinner(f"拉取 {len(symbols)} 只标的的市场快照…"):
                snapshot = fetch_snapshot_cached(symbols, decide_date)

            tradable = snapshot.tradable_symbols
            if not tradable:
                st.error(f"{decide_date} 候选标的均无有效行情。")
            else:
                # --- 候选标的卡片 ---
                cols = st.columns(len(tradable))
                for col, sym in zip(cols, tradable):
                    sd = snapshot.get(sym)
                    if sd is None:
                        continue
                    with col:
                        st.metric(
                            label=f"{sd.name} ({sym})",
                            value=f"{sd.last_close:.2f}" if sd.last_close else "-",
                            help=f"行业：{sd.industry or '未知'}",
                        )

                # --- 调用 Agent 团队 ---
                with st.spinner("4 位分析师并行思考中… 基金经理 R1 综合权衡…"):
                    t0 = time.perf_counter()
                    portfolio = Portfolio(initial_capital=decide_capital)
                    team = get_team()
                    decision = team.decide(snapshot, portfolio, save_log=True)
                    elapsed = time.perf_counter() - t0
                st.success(f"决策完成 · 总耗时 {elapsed:.1f}s · "
                           f"基金经理 {decision.metadata.get('tokens_completion', 0)} 输出 token")

                # --- 分析师矩阵 ---
                st.markdown("### 🔍 分析师矩阵")
                for sym, ops in decision.analyst_opinions.items():
                    sd = snapshot.get(sym)
                    name = sd.name if sd else sym
                    with st.expander(f"**{sym} · {name}**（{len(ops)} 位分析师）", expanded=True):
                        if ops:
                            st.markdown(render_opinion_matrix(ops), unsafe_allow_html=True)
                        # 关键事实 / 风险
                        facts, risks = [], []
                        for op in ops:
                            facts.extend(f"`[{op.agent_name}]` {f}" for f in op.key_facts)
                            risks.extend(f"`[{op.agent_name}]` {r}" for r in op.risks)
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**关键事实**")
                            if facts:
                                for f in facts[:8]:
                                    st.markdown(f"- {f}")
                            else:
                                st.caption("（无）")
                        with cc2:
                            st.markdown("**风险提示**")
                            if risks:
                                for r in risks[:5]:
                                    st.markdown(f"- {r}")
                            else:
                                st.caption("（无）")

                # --- 基金经理决策 ---
                st.markdown("### 🧠 基金经理决策")
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,{COLOR_CARD} 0%,#0f131c 100%);'
                    f'border:1px solid {COLOR_BORDER}; border-left:4px solid {COLOR_GOLD};'
                    f'padding:1rem 1.2rem; border-radius:4px; margin:0.4rem 0 0.8rem 0;">'
                    f'<div style="font-family:JetBrains Mono,monospace; font-size:0.7rem; '
                    f'letter-spacing:1.2px; color:{COLOR_GOLD}; margin-bottom:0.4rem; '
                    f'text-transform:uppercase;">PORTFOLIO MANAGER · DEEPSEEK-R1</div>'
                    f'<div style="color:{COLOR_TEXT}; font-size:0.95rem; line-height:1.6;">'
                    f'{_html_escape(decision.rationale or "（无总体逻辑）")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if decision.orders:
                    st.markdown(render_orders_table(decision.orders), unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="padding:0.8rem 1rem; background:{COLOR_CARD}; '
                        f'border:1px solid {COLOR_BORDER}; border-left:3px solid {COLOR_GOLD}; '
                        f'border-radius:4px; color:{COLOR_MUTED}; font-style:italic;">'
                        f'今日无交易订单 · <b style="color:{COLOR_GOLD_LIGHT}">按兵不动</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if decision.rejected:
                    with st.expander("🚫 主动放弃的标的"):
                        for rj in decision.rejected:
                            st.markdown(f"- **{rj.get('symbol')}**：{rj.get('reason')}")

                if decision.reasoning_content:
                    with st.expander("💡 基金经理思维链（DeepSeek-R1 `reasoning_content`）"):
                        st.markdown("> 来自推理模型的内部思考过程，是「决策可审计」的直接证据。")
                        st.code(decision.reasoning_content[:5000], language="markdown")

                # --- 命题 JSON 输出 ---
                st.markdown("### 📤 命题 JSON 输出")
                payload = orders_to_competition_payload(decision.orders, mode="strict")
                cc1, cc2 = st.columns([2, 1])
                with cc1:
                    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
                with cc2:
                    st.caption("严格模式：仅包含买入指令")
                    st.caption(f"订单数：**{len(payload)}**")
                    if any(o.side == "SELL" for o in decision.orders):
                        ext = orders_to_competition_payload(decision.orders, mode="extended")
                        st.caption("扩展模式（负 volume = 卖出）：")
                        st.code(json.dumps(ext, ensure_ascii=False), language="json")

                # --- 下载完整报告 ---
                report = build_markdown_report(decision, snapshot, portfolio)
                st.download_button(
                    "📥 下载完整决策报告（Markdown）",
                    data=report.encode("utf-8"),
                    file_name=f"决策报告_{decide_date}.md",
                    mime="text/markdown",
                )


# ---------------------------------------------------------------------------
# Tab 2 · 回测
# ---------------------------------------------------------------------------


with tab_backtest:
    st.subheader("📊 多 Agent 驱动的历史区间回测")
    st.caption("用同一套 Agent 团队作为决策函数，在历史区间内逐日推进；展示净值曲线、回撤、绩效指标与交易明细。")
    st.warning("提示：每个交易日 ≈ 6 次 LLM 调用（5 LLM 分析师 + 1 R1 基金经理；LSTM 量化 Agent 零成本），60 日回测约花 3-6 元 RMB，建议先在短区间小规模验证。", icon="💰")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        bt_start = st.date_input("起始日期", value=date(2026, 4, 15),
                                 min_value=date(2010, 1, 4), max_value=date.today(),
                                 key="bt_start")
    with c2:
        bt_end = st.date_input("结束日期", value=date(2026, 4, 30),
                               min_value=date(2010, 1, 4), max_value=date.today(),
                               key="bt_end")
    with c3:
        bt_symbols = st.text_input("候选标的", value="600519,601318",
                                   key="bt_symbols", help="A股 6 位代码，逗号分隔")

    bt_capital = st.number_input("初始资金", min_value=10000.0,
                                 value=float(settings.backtest.initial_capital), step=50_000.0,
                                 key="bt_capital")
    run_bt = st.button("🏁 启动回测", type="primary", disabled=not api_ok, key="run_bt")

    if run_bt:
        syms = [s.strip() for s in bt_symbols.replace("，", ",").split(",") if s.strip()]
        if not syms:
            st.error("请输入至少一个标的代码")
        elif bt_end < bt_start:
            st.error("结束日期不能早于起始日期")
        else:
            pipeline = get_pipeline()
            team = AgentTeam(log_dir=settings.paths.runs / f"backtest_{bt_start}_{bt_end}")
            engine = BacktestEngine(
                pipeline=pipeline,
                candidate_symbols=syms,
                decide_fn=team.as_decide_fn(save_log=True),
                initial_capital=bt_capital,
            )
            with st.spinner(f"回测进行中… 区间 {bt_start} ~ {bt_end}（约 5 次 LLM × N 个交易日）"):
                t0 = time.perf_counter()
                result = engine.run(bt_start, bt_end, verbose=False)
                bt_elapsed = time.perf_counter() - t0
            st.success(f"回测完成 · 耗时 {bt_elapsed:.1f}s")

            # --- 绩效指标卡片 ---
            m = result.metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("累计收益率", f"{m.total_return * 100:+.2f}%")
            mc2.metric("年化收益率", f"{m.annualized_return * 100:+.2f}%")
            mc3.metric("最大回撤", f"{m.max_drawdown * 100:.2f}%")
            mc4.metric("夏普比率", f"{m.sharpe:.2f}")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Calmar", f"{m.calmar:.2f}")
            mc2.metric("胜率", f"{m.win_rate * 100:.1f}%")
            mc3.metric("交易笔数", f"{m.trade_count}")
            mc4.metric("换手率", f"{m.turnover:.2f}x")

            # --- 净值曲线（金融终端风）---
            eq_df = pd.DataFrame(result.portfolio.equity_curve, columns=["date", "equity"])
            eq_df["return_pct"] = (eq_df["equity"] / result.initial_capital - 1) * 100

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.06,
                subplot_titles=("EQUITY CURVE 权益曲线", "RETURN % 累计收益率"),
            )
            # 权益曲线：金色主线 + 微弱金色填充
            fig.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["equity"], name="权益",
                line=dict(width=2.4, color=COLOR_GOLD_LIGHT),
                fill="tozeroy", fillcolor="rgba(245, 215, 110, 0.06)",
                hovertemplate="<b>%{x}</b><br>权益 <b>¥%{y:,.0f}</b><extra></extra>",
            ), row=1, col=1)
            # 初始资金参考线
            fig.add_hline(
                y=result.initial_capital, line_dash="dash",
                line_color="rgba(255,255,255,0.25)", line_width=1,
                annotation_text=f"初始 ¥{result.initial_capital:,.0f}",
                annotation_position="top right",
                annotation=dict(font=dict(color=COLOR_MUTED, size=10, family="JetBrains Mono")),
                row=1, col=1,
            )
            # 收益率：按正负着色
            return_pos = eq_df["return_pct"].where(eq_df["return_pct"] >= 0)
            return_neg = eq_df["return_pct"].where(eq_df["return_pct"] < 0)
            fig.add_trace(go.Scatter(
                x=eq_df["date"], y=return_pos, name="盈利",
                line=dict(color=COLOR_UP, width=1.5),
                fill="tozeroy", fillcolor="rgba(255, 59, 48, 0.18)",
                hovertemplate="<b>%{x}</b><br><b>%{y:+.2f}%</b><extra></extra>",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=eq_df["date"], y=return_neg, name="亏损",
                line=dict(color=COLOR_DOWN, width=1.5),
                fill="tozeroy", fillcolor="rgba(52, 199, 89, 0.18)",
                hovertemplate="<b>%{x}</b><br><b>%{y:+.2f}%</b><extra></extra>",
            ), row=2, col=1)

            fig = _apply_terminal_theme(fig, height=560)
            fig.update_layout(showlegend=False)
            for ann in fig.layout.annotations:
                if "EQUITY" in (ann.text or "") or "RETURN" in (ann.text or ""):
                    ann.font.color = COLOR_GOLD_LIGHT
                    ann.font.size = 11
                    ann.font.family = "JetBrains Mono"
            st.plotly_chart(fig, use_container_width=True)

            # --- 交易记录 ---
            with st.expander("📜 交易明细", expanded=False):
                if result.portfolio.trades:
                    trade_rows = [{
                        "日期": t.date, "方向": t.side, "代码": t.symbol, "名称": t.name,
                        "股数": t.shares, "成交价": round(t.price, 4),
                        "成交额": round(t.amount, 2), "手续费": round(t.fee, 2),
                        "印花税": round(t.tax, 2),
                        "已实现盈亏": round(t.realized_pnl, 2) if t.side == "SELL" else "—",
                    } for t in result.portfolio.trades]
                    st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("_无交易_")

            # --- 期末持仓 ---
            with st.expander("💼 期末持仓", expanded=False):
                snap = result.portfolio.snapshot()
                if snap["positions"]:
                    st.dataframe(pd.DataFrame(snap["positions"]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("_空仓_")


# ---------------------------------------------------------------------------
# Tab 3 · 数据探索
# ---------------------------------------------------------------------------


with tab_explore:
    st.subheader("🔍 单只标的的多维度数据视图")
    st.caption("行情 K线 / 均线 / MACD / 成交量 + 技术指标摘要 + 资金流 + 近期新闻 + 公司基本面")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ex_symbol = st.text_input("标的代码", value="600519", key="ex_symbol")
    with c2:
        ex_end = st.date_input("截止日期", value=date(2026, 5, 13),
                               key="ex_end", max_value=date.today())
    with c3:
        ex_lookback = st.number_input("回看交易日数", min_value=30, max_value=500,
                                      value=120, step=10, key="ex_lookback")

    if ex_symbol.strip():
        sym = ex_symbol.strip()
        ex_start = ex_end - timedelta(days=int(ex_lookback * 1.7) + 30)

        with st.spinner("拉取行情…"):
            price = fetch_price(sym, ex_start, ex_end)
        if not price.empty:
            price = price[price["date"] <= ex_end].tail(ex_lookback).reset_index(drop=True)

        info = sources.get_stock_info(sym) if price is not None else {}
        name = str(info.get("股票简称") or sym)
        industry = str(info.get("行业") or "未知")

        st.markdown(f"### {sym} · {name}  <span class='small-muted'>（行业：{industry}）</span>",
                    unsafe_allow_html=True)

        if price.empty:
            st.warning("该标的在选定区间内无行情数据。")
        else:
            st.plotly_chart(render_price_chart(price, title=f"{name} 日线"),
                            use_container_width=True)

            ind = indicators.summarize(price)
            with st.expander("📐 技术指标摘要", expanded=True):
                st.json(ind)

            # 资金流 + 新闻
            cc1, cc2 = st.columns(2)
            with cc1:
                with st.spinner("拉取资金流…"):
                    flow = sources.get_fund_flow(sym)
                st.markdown("**📊 个股资金流（近 20 日）**")
                if not flow.empty:
                    show = flow.tail(20)[["date", "pct_change", "main_net", "main_net_pct"]].copy()
                    show.columns = ["日期", "涨跌幅%", "主力净流入(元)", "主力净占比%"]
                    st.dataframe(show, use_container_width=True, hide_index=True)
                else:
                    st.caption("_无资金流数据_")
            with cc2:
                with st.spinner("拉取新闻…"):
                    news = sources.get_news(sym, limit=10)
                st.markdown("**📰 近期新闻（最多 10 条）**")
                if not news.empty:
                    for _, row in news.iterrows():
                        title = row.get("title", "")
                        ts = str(row.get("publish_time", ""))[:16]
                        content = str(row.get("content", ""))[:160]
                        st.markdown(f"**{title}** <span class='small-muted'>· {ts}</span>",
                                    unsafe_allow_html=True)
                        st.caption(content + ("…" if len(str(row.get('content', ''))) > 160 else ""))
                        st.markdown("---")
                else:
                    st.caption("_无新闻_")

            # 公司基本面
            if info:
                with st.expander("🏛️ 公司基本面", expanded=False):
                    cols = st.columns(4)
                    for i, (k, v) in enumerate(info.items()):
                        cols[i % 4].caption(f"**{k}**")
                        cols[i % 4].write(v)
