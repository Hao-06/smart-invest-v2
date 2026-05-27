"""智投未来 · LiveTradingAgentV2 专属看板。

围绕 V2 真实输出（regime 概率 / 4 策略权重 / top 8 推荐 / 决策日志）的可视化看板，
与早期 `app/dashboard.py`（6 分析师架构）并存。

3 个 Tab：
  📊 业绩长卷 —— 连续回测净值曲线 + 6 区间汇总（数据 100% 来自 runs/v2_*.json，秒显）
  🔍 决策审计 —— 在连续回测 571 天里挑任一天，展示 regime → 权重 → 推荐三层
  🤖 当日推荐 —— 日期选择 → 跑 `agent.recommend()` → 实时显示决策

启动：
  streamlit run app/dashboard_v2.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date as date_type
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═════════════════════════ 视觉系统（金融终端风） ═════════════════════════

COLOR_BG = "#0a0d14"
COLOR_CARD = "#141821"
COLOR_BORDER = "#1f2433"
COLOR_GOLD = "#d4af37"
COLOR_GOLD_LIGHT = "#f5d76e"
COLOR_TEXT = "#e8e8e8"
COLOR_MUTED = "#8a93a8"
COLOR_UP = "#ff3b30"      # A 股涨红
COLOR_DOWN = "#34c759"    # A 股跌绿
COLOR_CYAN = "#5ac8fa"
COLOR_PURPLE = "#a78bfa"
COLOR_ORANGE = "#ff9500"

STRATEGY_COLORS = {
    "momentum": COLOR_UP,
    "etf_defense": COLOR_GOLD,
    "sector_rotation": COLOR_CYAN,
    "high_dividend": COLOR_PURPLE,
}
STRATEGY_NAMES = {
    "momentum": "动量",
    "etf_defense": "ETF 防御",
    "sector_rotation": "行业轮动",
    "high_dividend": "高股息",
}

REGIME_COLORS = {
    "trending_bull": COLOR_UP,
    "trending_bear": COLOR_DOWN,
    "range_bound": COLOR_GOLD,
    "breakout": COLOR_ORANGE,
    "volatile": COLOR_PURPLE,
}
REGIME_NAMES = {
    "trending_bull": "趋势牛",
    "trending_bear": "趋势熊",
    "range_bound": "震荡市",
    "breakout": "突破启动",
    "volatile": "高波动",
}
REGIME_ICONS = {
    "trending_bull": "📈",
    "trending_bear": "📉",
    "range_bound": "〰️",
    "breakout": "🚀",
    "volatile": "⚡",
}
REGIME_DESCRIPTIONS = {
    "trending_bull": "趋势向上，买盘主导。骑趋势、加配动量与行业轮动。",
    "trending_bear": "趋势向下，卖盘主导。防御优先，偏配高股息与 ETF。",
    "range_bound": "震荡磨耗，无明确方向。ETF 跟住大盘 beta、控仓为主。",
    "breakout": "突破启动，趋势加速向上。重配行业轮动捕捉主线。",
    "volatile": "高波动无序，风险显著升高。重防御、降低单一暴露。",
}

STRATEGY_ICONS = {
    "momentum": "📈",
    "etf_defense": "🛡️",
    "sector_rotation": "⚡",
    "high_dividend": "💰",
}
STRATEGY_DESCRIPTIONS = {
    "momentum": "横截面动量 + 大盘择时",
    "etf_defense": "沪深300 ETF 防御",
    "sector_rotation": "15 只行业 ETF 轮动",
    "high_dividend": "高股息蓝筹防御",
}


st.set_page_config(
    page_title="智投未来 · LiveTradingAgentV2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: {COLOR_BG};
}}
.main .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}

/* Metric 卡片：深色 + 金色左边线 */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
    font-size: 1.55rem !important; font-weight: 700 !important;
    color: {COLOR_GOLD_LIGHT} !important; letter-spacing: -0.5px;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important; color: {COLOR_MUTED} !important;
}}
[data-testid="stMetricLabel"] p {{
    color: {COLOR_MUTED} !important; font-size: 0.68rem !important;
    text-transform: uppercase; letter-spacing: 1.3px; font-weight: 500;
}}
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {COLOR_CARD} 0%, #0f131c 100%);
    border: 1px solid {COLOR_BORDER};
    border-left: 3px solid {COLOR_GOLD};
    padding: 0.7rem 0.9rem; border-radius: 4px;
    transition: border-color 0.15s ease;
}}
div[data-testid="stMetric"]:hover {{ border-left-color: {COLOR_GOLD_LIGHT}; }}

/* 标题 */
h1, h2, h3 {{ font-weight: 600 !important; }}
h2 {{
    color: {COLOR_GOLD_LIGHT}; font-size: 1.05rem !important;
    border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: 0.35rem;
    margin-top: 1.4rem !important; letter-spacing: 0.5px;
}}

/* Tab 风格 */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none; }}
.stTabs [data-baseweb="tab"] {{
    background: {COLOR_CARD}; padding: 0.55rem 1.4rem; border-radius: 4px;
    color: {COLOR_MUTED}; font-weight: 500; font-size: 0.95rem;
    border: 1px solid {COLOR_BORDER};
}}
.stTabs [aria-selected="true"] {{
    background: {COLOR_GOLD} !important; color: {COLOR_BG} !important;
    border-color: {COLOR_GOLD} !important; font-weight: 600;
}}

/* 表格 */
[data-testid="stDataFrame"] {{ background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; }}
[data-testid="stDataFrame"] thead tr th {{
    background: {COLOR_CARD} !important; color: {COLOR_GOLD_LIGHT} !important;
    font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 1px;
    font-size: 0.7rem !important;
}}
[data-testid="stDataFrame"] tbody tr td {{
    font-family: 'JetBrains Mono', monospace !important; color: {COLOR_TEXT};
}}

/* selectbox */
div[data-baseweb="select"] > div {{
    background: {COLOR_CARD} !important; border-color: {COLOR_BORDER} !important;
    color: {COLOR_TEXT} !important;
}}

/* date input */
.stDateInput input {{
    background: {COLOR_CARD} !important; color: {COLOR_GOLD_LIGHT} !important;
    border: 1px solid {COLOR_BORDER} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* 按钮 */
.stButton button {{
    background: {COLOR_GOLD}; color: {COLOR_BG}; border: none; font-weight: 600;
    padding: 0.5rem 1.4rem; border-radius: 3px;
    font-family: 'Inter', sans-serif; letter-spacing: 1px; text-transform: uppercase;
    font-size: 0.85rem;
}}
.stButton button:hover {{ background: {COLOR_GOLD_LIGHT}; color: {COLOR_BG}; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════ 数据加载 ═════════════════════════

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CONT_PATH = RUNS_DIR / "v2_continuous_2024_2026.json"
REGION_PATHS = {
    "2023 全年": RUNS_DIR / "v2_region_2023_全年.json",
    "2024 Q1": RUNS_DIR / "v2_region_2024_Q1.json",
    "2024 Q2-Q3": RUNS_DIR / "v2_region_2024_Q2-Q3.json",
    "2024 9.24": RUNS_DIR / "v2_region_2024_9.24.json",
    "2025 全年": RUNS_DIR / "v2_region_2025_全年.json",
    "2026 H1": RUNS_DIR / "v2_region_2026_H1.json",
}


@st.cache_data(show_spinner=False)
def load_report(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


# ═════════════════════════ 工具：plotly 主题 ═════════════════════════

def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT, size=12),
        height=height, margin=dict(l=50, r=30, t=30, b=40),
        xaxis=dict(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
                   showline=True, linecolor=COLOR_BORDER,
                   tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER,
                   showline=True, linecolor=COLOR_BORDER,
                   tickfont=dict(family="JetBrains Mono", size=10)),
        legend=dict(bgcolor="rgba(20,24,33,0.8)", bordercolor=COLOR_BORDER,
                    borderwidth=1, font=dict(color=COLOR_TEXT, size=11),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLOR_CARD, bordercolor=COLOR_GOLD,
                        font=dict(family="JetBrains Mono", color=COLOR_TEXT, size=12)),
    )
    return fig


def render_regime_stacked(probs: dict, primary: str, height: int = 60) -> go.Figure:
    """全宽水平堆叠条 —— 5 种 regime 占 100%，主 regime 高亮，其余调暗。"""
    items = sorted(probs.items(), key=lambda x: -x[1])
    fig = go.Figure()
    for k, v in items:
        is_primary = k == primary
        pct = v * 100
        fig.add_trace(go.Bar(
            x=[pct], y=[""], orientation="h",
            name=REGIME_NAMES.get(k, k),
            marker=dict(color=REGIME_COLORS.get(k, COLOR_GOLD),
                        line=dict(color=COLOR_BG, width=2)),
            opacity=1.0 if is_primary else 0.35,
            text=f"{pct:.0f}%" if v >= 0.05 else "",
            textposition="inside",
            textfont=dict(family="JetBrains Mono", color=COLOR_BG, size=13),
            insidetextanchor="middle",
            hovertemplate=f"<b>{REGIME_NAMES.get(k, k)}</b>: {pct:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height, bargap=0,
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
    )
    return fig


def render_weight_donut(weights: dict, height: int = 280) -> go.Figure:
    """4 策略权重环形图，中心显示总仓位。"""
    items = list(weights.items())
    labels = [STRATEGY_NAMES.get(k, k) for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = [STRATEGY_COLORS.get(k, COLOR_GOLD) for k, _ in items]
    total_pos = sum(values)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=COLOR_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12, color=COLOR_BG),
        hovertemplate="<b>%{label}</b><br>权重 %{value:.1f}%<extra></extra>",
        rotation=90, direction="clockwise", sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(text=f"<b>{total_pos:.0f}%</b>", x=0.5, y=0.56,
                 font=dict(size=30, color=COLOR_GOLD_LIGHT, family="JetBrains Mono"),
                 showarrow=False),
            dict(text="总仓位", x=0.5, y=0.42,
                 font=dict(size=11, color=COLOR_MUTED, family="Inter"),
                 showarrow=False),
        ],
        margin=dict(l=10, r=10, t=10, b=10), height=height,
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
    )
    return fig


def render_regime_hero(primary: str, conf: float) -> None:
    """大幅 regime 焦点卡 —— 图标 + 名称 + 描述 + 置信度。"""
    color = REGIME_COLORS.get(primary, COLOR_GOLD)
    icon = REGIME_ICONS.get(primary, "🎯")
    name = REGIME_NAMES.get(primary, primary)
    desc = REGIME_DESCRIPTIONS.get(primary, "")

    st.markdown(f"""
    <div style='padding:1.1rem 1.4rem;
                background:linear-gradient(135deg, {COLOR_CARD} 0%, #0f131c 100%);
                border:1px solid {COLOR_BORDER}; border-left:4px solid {color};
                border-radius:6px;'>
      <div style='display:flex; align-items:center; gap:1.2rem;'>
        <div style='font-size:3rem; line-height:1;'>{icon}</div>
        <div style='flex:1;'>
          <div style='color:{COLOR_MUTED}; font-size:0.65rem;
                      text-transform:uppercase; letter-spacing:1.8px;'>市场状态</div>
          <div style='color:{color}; font-size:1.75rem; font-weight:700;
                      line-height:1.1; margin-top:0.2rem; letter-spacing:1px;'>{name}</div>
          <div style='color:{COLOR_TEXT}; font-size:0.85rem;
                      line-height:1.5; margin-top:0.5rem;'>{desc}</div>
        </div>
        <div style='text-align:right; min-width:90px;'>
          <div style='color:{COLOR_GOLD_LIGHT}; font-family:JetBrains Mono;
                      font-size:2rem; font-weight:700; line-height:1;'>{conf:.0%}</div>
          <div style='color:{COLOR_MUTED}; font-size:0.65rem;
                      text-transform:uppercase; letter-spacing:1.5px;
                      margin-top:0.3rem;'>置信度</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_strategy_breakdown(weights: dict) -> None:
    """4 策略横向卡片列表 —— 排序、活跃/休眠区分。"""
    items = sorted(weights.items(), key=lambda x: -x[1])
    rows = []
    for k, v in items:
        active = v >= 0.10
        color = STRATEGY_COLORS.get(k, COLOR_GOLD)
        icon = STRATEGY_ICONS.get(k, "🎯")
        name = STRATEGY_NAMES.get(k, k)
        desc = STRATEGY_DESCRIPTIONS.get(k, "")
        opacity = "1.0" if active else "0.4"
        status = "" if active else f"<span style='color:{COLOR_MUTED};font-size:0.65rem;margin-left:0.4rem;'>· 休眠(权重<10%)</span>"
        rows.append(f"""
        <div style='display:flex; align-items:center; gap:0.9rem; padding:0.65rem 0.85rem;
                    background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                    border-left:3px solid {color}; border-radius:4px;
                    margin-bottom:0.45rem; opacity:{opacity};'>
          <div style='font-size:1.5rem; line-height:1;'>{icon}</div>
          <div style='flex:1;'>
            <div style='color:{COLOR_TEXT}; font-weight:600; font-size:0.92rem;'>{name}{status}</div>
            <div style='color:{COLOR_MUTED}; font-size:0.7rem; margin-top:0.1rem;'>{desc}</div>
          </div>
          <div style='color:{color}; font-family:JetBrains Mono;
                      font-size:1.35rem; font-weight:700;'>{v*100:.0f}<span style='font-size:0.7rem; color:{COLOR_MUTED}; margin-left:2px;'>%</span></div>
        </div>
        """)
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_rec_cards(recs: list) -> None:
    """推荐卡片栅格 —— 4 列 × N 行，每张卡 # / 名称 / 代码 / volume / ETF/个股 tag。"""
    if not recs:
        st.markdown(f"<div style='padding:1rem 1.2rem; background:{COLOR_CARD}; "
                    f"border:1px solid {COLOR_BORDER}; border-left:3px solid {COLOR_MUTED}; "
                    f"color:{COLOR_MUTED}; border-radius:4px;'>"
                    f"📭 该日 V2 无推荐 —— 当日无操作（市场信号不利或候选为空）。"
                    f"</div>", unsafe_allow_html=True)
        return

    cols_per_row = 4
    for row_start in range(0, len(recs), cols_per_row):
        row_recs = recs[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, r in enumerate(row_recs):
            rank = row_start + i + 1
            sym = r.get("symbol", "")
            name = r.get("symbol_name", sym) or sym
            vol = r.get("volume", 0)
            is_etf = len(sym) == 6 and sym[:2] in ("51", "15", "58", "56")
            accent = COLOR_GOLD if is_etf else COLOR_CYAN
            tag = "ETF" if is_etf else "个股"
            # 名字过长截断
            display_name = name if len(name) <= 10 else name[:9] + "…"
            with cols[i]:
                st.markdown(f"""
                <div style='padding:0.7rem 0.85rem; background:{COLOR_CARD};
                            border:1px solid {COLOR_BORDER}; border-top:3px solid {accent};
                            border-radius:4px; height:118px; display:flex; flex-direction:column;
                            justify-content:space-between;'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='color:{accent}; font-family:JetBrains Mono;
                                font-size:0.9rem; font-weight:700;'>#{rank}</div>
                    <div style='color:{accent}; font-size:0.58rem; text-transform:uppercase;
                                letter-spacing:1.3px; padding:0.12rem 0.45rem; background:{COLOR_BG};
                                border:1px solid {accent}; border-radius:3px;
                                font-weight:600;'>{tag}</div>
                  </div>
                  <div>
                    <div style='color:{COLOR_TEXT}; font-size:0.92rem; font-weight:600;
                                line-height:1.2; margin-bottom:0.18rem;'>{display_name}</div>
                    <div style='color:{COLOR_MUTED}; font-family:JetBrains Mono;
                                font-size:0.72rem;'>{sym}</div>
                  </div>
                  <div style='border-top:1px solid {COLOR_BORDER}; padding-top:0.35rem;
                              display:flex; justify-content:space-between; align-items:baseline;'>
                    <div style='color:{COLOR_MUTED}; font-size:0.6rem; text-transform:uppercase;
                                letter-spacing:1px;'>建议量</div>
                    <div style='color:{COLOR_GOLD_LIGHT}; font-family:JetBrains Mono;
                                font-size:0.95rem; font-weight:700;'>{vol:,}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


def render_decision_layers(dec: dict) -> None:
    """共享：渲染单日决策的三层可视化(决策审计 + 当日推荐 共用)。"""
    regime = dec.get("regime", {}) or {}
    probs = regime.get("probs", {}) or {}
    primary = regime.get("regime", "unknown")
    conf = regime.get("confidence", 0)

    # ── 第 1 层 · 市场状态识别 ──
    st.markdown("## 第 1 层 · 市场状态识别")
    if probs:
        render_regime_hero(primary, conf)
        st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)
        st.plotly_chart(render_regime_stacked(probs, primary),
                        use_container_width=True, config={"displayModeBar": False})
        # 图例条
        legend_parts = []
        for k, v in sorted(probs.items(), key=lambda x: -x[1]):
            color = REGIME_COLORS.get(k, COLOR_GOLD)
            is_primary = k == primary
            weight = "700" if is_primary else "500"
            opacity = "1.0" if is_primary else "0.55"
            legend_parts.append(
                f"<span style='display:inline-block; padding:0.18rem 0.55rem;"
                f"margin-right:0.35rem; margin-bottom:0.3rem; background:{COLOR_CARD};"
                f"border:1px solid {COLOR_BORDER}; border-left:3px solid {color};"
                f"border-radius:3px; font-size:0.73rem; opacity:{opacity};'>"
                f"<span style='color:{COLOR_TEXT}; font-weight:{weight};'>{REGIME_NAMES.get(k, k)}</span>"
                f" <span style='font-family:JetBrains Mono; color:{color}; font-weight:700;'>"
                f"{v*100:.0f}%</span></span>"
            )
        st.markdown(
            f"<div style='margin-top:0.5rem; line-height:1.8;'>{''.join(legend_parts)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("该日无 regime 数据")

    # ── 第 2 层 · 策略权重分配 ──
    weights = dec.get("weights", {}) or {}
    if weights:
        st.markdown("## 第 2 层 · 策略权重分配")
        cL, cR = st.columns([1, 1])
        with cL:
            st.plotly_chart(render_weight_donut(weights),
                            use_container_width=True, config={"displayModeBar": False})
        with cR:
            st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
            render_strategy_breakdown(weights)

    # ── 第 3 层 · 推荐清单 ──
    recs = dec.get("recs") or dec.get("recommendations") or []
    n = dec.get("n_recs", len(recs))
    st.markdown(f"## 第 3 层 · 推荐清单 ({n} 只)")
    render_rec_cards(recs)


# ═════════════════════════ 顶部标题 ═════════════════════════

st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.6rem;
            padding:0.5rem 0; border-bottom:1px solid {COLOR_BORDER};">
  <div style="font-size:2.2rem;">🤖</div>
  <div style="flex:1;">
    <div style="color:{COLOR_GOLD_LIGHT}; font-size:1.6rem; font-weight:700;
                letter-spacing:0.5px; line-height:1.1;">
      智投未来 · LiveTradingAgentV2
    </div>
    <div style="color:{COLOR_MUTED}; font-size:0.78rem; letter-spacing:1.5px;
                text-transform:uppercase; margin-top:0.15rem;">
      纯推荐型多智能体 A股投研系统 · Pure-Recommendation Multi-Agent
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:{COLOR_GOLD}; font-family:'JetBrains Mono'; font-size:0.7rem;
                letter-spacing:1px;">CUEB · 驼灵智能体大赛</div>
    <div style="color:{COLOR_MUTED}; font-family:'JetBrains Mono'; font-size:0.7rem;">
      金融投资赛道 · v2.0
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  业绩长卷", "🔍  决策审计", "🤖  当日推荐"])


# ═════════════════════════════════════════════════════════════════
# Tab 1 · 业绩长卷
# ═════════════════════════════════════════════════════════════════
with tab1:
    if not CONT_PATH.exists():
        st.error(f"找不到连续回测报告：`{CONT_PATH.relative_to(ROOT)}`\n\n"
                 f"请先跑：`python3 main.py validate --start 2024-01-02 --end 2026-05-18 "
                 f"--out runs/v2_continuous_2024_2026.json`")
    else:
        cont = load_report(str(CONT_PATH))
        de = cont.get("daily_equity", [])

        # ── KPI 条 ──
        n_days = len(de) if de else len(cont.get("decisions", []))
        years = max(n_days / 242, 0.01)
        ann_v2 = ((1 + cont["total_return_pct"] / 100) ** (1 / years) - 1) * 100
        ann_hs = ((1 + cont["hs300_return_pct"] / 100) ** (1 / years) - 1) * 100
        calmar = ann_v2 / abs(cont["max_drawdown_pct"]) if cont.get("max_drawdown_pct") else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("累计收益", f"{cont['total_return_pct']:+.2f}%",
                            f"沪深300 {cont['hs300_return_pct']:+.2f}%")
        with c2: st.metric("年化收益", f"{ann_v2:+.1f}%", f"沪深300 {ann_hs:+.1f}%")
        with c3: st.metric("超额", f"{cont['excess_pct']:+.2f} pp", "vs 沪深300")
        with c4: st.metric("最大回撤", f"{cont['max_drawdown_pct']:+.2f}%",
                            f"{years:.1f} 年峰谷")
        with c5: st.metric("Calmar", f"{calmar:.2f}", "年化/|回撤|")
        with c6: st.metric("期末资金", f"¥{cont['final']:,.0f}",
                            f"初始 ¥{cont['capital']:,.0f}")

        # ── 净值曲线 ──
        st.markdown("## 净值曲线 · V2 vs 沪深300")
        if de:
            dates = [d["date"] for d in de]
            v2_pct = [d["v2_pct"] for d in de]
            hs_pct = [d["hs300_pct"] if d.get("hs300_pct") is not None else 0 for d in de]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=v2_pct, name="LiveAgentV2",
                line=dict(color=COLOR_GOLD, width=2.5),
                hovertemplate="<b>V2</b>: %{y:+.2f}%<extra></extra>",
                fill="tozeroy", fillcolor="rgba(212,175,55,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=hs_pct, name="沪深300",
                line=dict(color=COLOR_CYAN, width=1.8, dash="dot"),
                hovertemplate="<b>HS300</b>: %{y:+.2f}%<extra></extra>",
            ))
            fig.update_yaxes(title="累计收益率", ticksuffix="%")
            fig.add_hline(y=0, line=dict(color=COLOR_MUTED, width=0.8))
            st.plotly_chart(style_fig(fig, 440), use_container_width=True,
                            config={"displayModeBar": False})

            # ── 回撤水下图 ──
            st.markdown("## 回撤水下图(跨整段连续峰谷)")
            equities = [d["v2_equity"] for d in de]
            peak = equities[0]; dd = []
            for eq in equities:
                peak = max(peak, eq); dd.append((eq / peak - 1) * 100)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dates, y=dd, fill="tozeroy", name="V2 回撤",
                line=dict(color=COLOR_DOWN, width=1.5),
                fillcolor="rgba(52,199,89,0.18)",
                hovertemplate="<b>回撤</b>: %{y:.2f}%<extra></extra>",
            ))
            fig_dd.update_yaxes(title="回撤", ticksuffix="%")
            fig_dd.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig_dd, 220), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.warning("⚠️  当前回测报告里没有 `daily_equity` 序列 —— 请重跑：\n"
                       "`python3 main.py validate --start 2024-01-02 --end 2026-05-18 "
                       "--out runs/v2_continuous_2024_2026.json`")

        # ── 6 区间分行情 ──
        st.markdown("## 6 区间分行情验证 · 跨 regime 鲁棒性")
        cols = st.columns(6)
        for (name, p), col in zip(REGION_PATHS.items(), cols):
            with col:
                if not p.exists():
                    st.markdown(f"<div style='padding:0.6rem; background:{COLOR_CARD}; "
                                f"border:1px solid {COLOR_BORDER}; border-radius:4px;'>"
                                f"<div style='color:{COLOR_MUTED};font-size:0.65rem;'>{name}</div>"
                                f"<div style='color:{COLOR_MUTED};font-size:0.75rem;margin-top:0.3rem;'>无数据</div>"
                                f"</div>", unsafe_allow_html=True)
                    continue
                r = load_report(str(p))
                excess = r["excess_pct"]
                color = COLOR_UP if excess > 0 else COLOR_DOWN
                icon = "✅" if excess > 0 else "❌"
                st.markdown(f"""
                <div style='padding:0.7rem 0.8rem; background:{COLOR_CARD};
                           border:1px solid {COLOR_BORDER}; border-left:3px solid {color};
                           border-radius:4px;'>
                  <div style='color:{COLOR_MUTED};font-size:0.65rem;
                              text-transform:uppercase;letter-spacing:1px;'>{name}</div>
                  <div style='color:{color};font-family:JetBrains Mono;
                              font-size:1.3rem;font-weight:700;
                              margin:0.25rem 0;'>{excess:+.2f}<span style="font-size:0.65rem;color:{COLOR_MUTED};margin-left:2px;">pp</span></div>
                  <div style='color:{COLOR_TEXT};font-family:JetBrains Mono;font-size:0.75rem;'>
                    V2 {r['total_return_pct']:+.2f}%</div>
                  <div style='color:{COLOR_MUTED};font-family:JetBrains Mono;font-size:0.7rem;'>
                    HS300 {r['hs300_return_pct']:+.2f}%</div>
                  <div style='margin-top:0.2rem;font-size:0.85rem;'>{icon}</div>
                </div>""", unsafe_allow_html=True)

        # ── 死亡测试对比 ──
        st.markdown("## 通过 v9 死亡测试 · 非过拟合的硬证据")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""
            <div style='padding:1rem; background:{COLOR_CARD};
                       border:1px solid {COLOR_BORDER}; border-left:3px solid {COLOR_DOWN};
                       border-radius:4px;'>
              <div style='color:{COLOR_MUTED};font-size:0.7rem;
                          text-transform:uppercase;letter-spacing:1.5px;'>第 1 代 · 固定参数 v9</div>
              <div style='color:{COLOR_TEXT};font-size:0.85rem;margin:0.4rem 0;'>
                4 区间综合分 <span style='color:{COLOR_GOLD_LIGHT};font-family:JetBrains Mono;'>+7.57</span> 看似优秀
              </div>
              <div style='color:{COLOR_DOWN};font-family:JetBrains Mono;
                          font-size:1.8rem;font-weight:700;margin:0.3rem 0;'>
                −17.54 pp <span style='font-size:1rem;'>❌</span>
              </div>
              <div style='color:{COLOR_MUTED};font-size:0.78rem;'>
                12 月连续回测 · 过拟合崩盘
              </div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""
            <div style='padding:1rem; background:{COLOR_CARD};
                       border:1px solid {COLOR_BORDER}; border-left:3px solid {COLOR_UP};
                       border-radius:4px;'>
              <div style='color:{COLOR_MUTED};font-size:0.7rem;
                          text-transform:uppercase;letter-spacing:1.5px;'>第 5 代 · LiveTradingAgentV2</div>
              <div style='color:{COLOR_TEXT};font-size:0.85rem;margin:0.4rem 0;'>
                2.4 年连续 · 571 个交易日不间断 · 同款检验
              </div>
              <div style='color:{COLOR_UP};font-family:JetBrains Mono;
                          font-size:1.8rem;font-weight:700;margin:0.3rem 0;'>
                {cont['excess_pct']:+.2f} pp <span style='font-size:1rem;'>✅</span>
              </div>
              <div style='color:{COLOR_MUTED};font-size:0.78rem;'>
                {cont['total_return_pct']:+.2f}% vs 沪深300 {cont['hs300_return_pct']:+.2f}%
              </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# Tab 2 · 决策审计
# ═════════════════════════════════════════════════════════════════
with tab2:
    if not CONT_PATH.exists():
        st.error(f"找不到连续回测报告：`{CONT_PATH.relative_to(ROOT)}`")
    else:
        cont = load_report(str(CONT_PATH))
        decisions = cont.get("decisions", [])
        if not decisions:
            st.warning("连续回测报告里没有 decisions —— 请检查 V2 回测脚本。")
        else:
            dates_list = [d["as_of"] for d in decisions]
            non_empty = [d for d in decisions if d.get("recs")]

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                sel = st.selectbox(
                    f"决策日(共 {len(dates_list)} 天)",
                    dates_list,
                    index=dates_list.index("2024-02-19") if "2024-02-19" in dates_list else len(dates_list) // 2,
                )
                dec = next(d for d in decisions if d["as_of"] == sel)
            with c2:
                if st.button("🎲 随机抽一天", use_container_width=True):
                    import random
                    sel = random.choice(non_empty)["as_of"] if non_empty else random.choice(dates_list)
                    st.rerun()
            with c3:
                st.markdown(f"<div style='padding:0.5rem 0; color:{COLOR_MUTED}; font-size:0.85rem;'>"
                            f"V2 在 <b style='color:{COLOR_GOLD_LIGHT};'>{sel}</b> 这一天的完整决策"
                            f"</div>", unsafe_allow_html=True)

            # ── 当日大盘上下文（HS300 ±30 交易日小图，黄虚线标当前日） ──
            de_all = cont.get("daily_equity", [])
            if de_all:
                date_to_idx = {d["date"]: i for i, d in enumerate(de_all)}
                sel_idx = date_to_idx.get(sel)
                if sel_idx is not None:
                    lo = max(0, sel_idx - 30)
                    hi = min(len(de_all), sel_idx + 31)
                    window = de_all[lo:hi]
                    win_dates = [d["date"] for d in window]
                    win_hs = [d["hs300_close"] for d in window if d.get("hs300_close")]
                    win_dates_h = [d["date"] for d in window if d.get("hs300_close")]
                    # 当日涨幅
                    today_close = de_all[sel_idx].get("hs300_close")
                    prev_close = de_all[sel_idx - 1].get("hs300_close") if sel_idx > 0 else None
                    if today_close and prev_close:
                        day_pct = (today_close / prev_close - 1) * 100
                        day_color = COLOR_UP if day_pct >= 0 else COLOR_DOWN
                        day_label = f"{day_pct:+.2f}%"
                    else:
                        day_color, day_label = COLOR_MUTED, "—"

                    cL, cR = st.columns([3, 1])
                    with cL:
                        fig_ctx = go.Figure()
                        fig_ctx.add_trace(go.Scatter(
                            x=win_dates_h, y=win_hs, mode="lines",
                            line=dict(color=COLOR_CYAN, width=2),
                            fill="tozeroy", fillcolor="rgba(90,200,250,0.07)",
                            hovertemplate="<b>沪深300</b> %{x}<br>收盘 %{y:.3f}<extra></extra>",
                            name="沪深300",
                        ))
                        fig_ctx.add_vline(x=sel,
                                          line=dict(color=COLOR_GOLD_LIGHT, width=2, dash="dash"))
                        if today_close:
                            fig_ctx.add_trace(go.Scatter(
                                x=[sel], y=[today_close], mode="markers",
                                marker=dict(color=COLOR_GOLD_LIGHT, size=10,
                                            line=dict(color=COLOR_BG, width=2)),
                                hovertemplate=f"<b>选中日 {sel}</b><br>收盘 {today_close:.3f}<extra></extra>",
                                showlegend=False,
                            ))
                        fig_ctx.update_yaxes(title="510300 收盘价")
                        fig_ctx.update_layout(showlegend=False)
                        fig_ctx.update_layout(title=dict(
                            text=f"沪深300 · 选中日 ±30 交易日上下文",
                            font=dict(color=COLOR_GOLD_LIGHT, size=13),
                            x=0.02, y=0.97))
                        st.plotly_chart(style_fig(fig_ctx, 220),
                                        use_container_width=True,
                                        config={"displayModeBar": False})
                    with cR:
                        close_str = f"{today_close:.3f}" if today_close else "—"
                        st.markdown(f"""
                        <div style='padding:1rem 1.1rem; background:{COLOR_CARD};
                                    border:1px solid {COLOR_BORDER};
                                    border-left:3px solid {day_color};
                                    border-radius:4px; margin-top:0.5rem; height:200px;
                                    display:flex; flex-direction:column; justify-content:center;'>
                          <div style='color:{COLOR_MUTED}; font-size:0.65rem;
                                      text-transform:uppercase; letter-spacing:1.5px;'>当日大盘</div>
                          <div style='color:{day_color}; font-family:JetBrains Mono;
                                      font-size:2rem; font-weight:700;
                                      margin-top:0.3rem; line-height:1;'>{day_label}</div>
                          <div style='color:{COLOR_TEXT}; font-size:0.78rem;
                                      margin-top:0.6rem;'>沪深300 当日涨跌</div>
                          <div style='color:{COLOR_MUTED}; font-family:JetBrains Mono;
                                      font-size:0.72rem; margin-top:0.3rem;'>
                            510300 收盘 {close_str}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            render_decision_layers(dec)

            # ── 整段 regime 概率堆叠面积图（替换原散点） ──
            st.markdown("## 整段 regime 概率分布(2.4 年)")
            timeline_dates = [d["as_of"] for d in decisions]
            regime_keys = list(REGIME_COLORS.keys())
            stacks = {rg: [] for rg in regime_keys}
            for d in decisions:
                ps = (d.get("regime") or {}).get("probs", {}) or {}
                for rg in regime_keys:
                    stacks[rg].append(ps.get(rg, 0) * 100)
            fig_area = go.Figure()
            for rg in regime_keys:
                color = REGIME_COLORS[rg]
                # rgba 半透明
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fillcolor = f"rgba({r},{g},{b},0.75)"
                fig_area.add_trace(go.Scatter(
                    x=timeline_dates, y=stacks[rg], mode="none",
                    stackgroup="one", name=REGIME_NAMES.get(rg, rg),
                    fillcolor=fillcolor,
                    hovertemplate=f"<b>{REGIME_NAMES.get(rg, rg)}</b>: %{{y:.0f}}%<extra></extra>",
                ))
            fig_area.add_vline(x=sel,
                               line=dict(color=COLOR_GOLD_LIGHT, width=2, dash="dash"))
            fig_area.update_yaxes(title="概率累加", ticksuffix="%", range=[0, 100])
            fig_area.update_layout(
                legend=dict(orientation="h", y=-0.18, x=0,
                            font=dict(size=11)),
                hovermode="x unified",
            )
            st.plotly_chart(style_fig(fig_area, 280),
                            use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════
# Tab 3 · 当日推荐
# ═════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div style='color:{COLOR_MUTED}; font-size:0.85rem; margin-bottom:0.8rem;'>"
                f"选定一个交易日 → 实时调用 <code style='color:{COLOR_GOLD_LIGHT};'>"
                f"LiveTradingAgentV2.recommend(date)</code> → 输出三层决策。"
                f"</div>", unsafe_allow_html=True)

    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
    with cc1:
        target = st.date_input("决策日", value=date_type(2025, 12, 26),
                               min_value=date_type(2024, 1, 2),
                               max_value=date_type(2026, 12, 31))
    with cc2:
        universe_size = st.number_input("PIT universe 大小", min_value=10, max_value=200, value=80, step=10)
    with cc3:
        use_llm = st.checkbox("启用 LLM 推理(R1)", value=False,
                              help="实盘场景；需 .env 配置 DEEPSEEK_API_KEY")
    with cc4:
        st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
        run = st.button("▶  生成推荐", use_container_width=True, type="primary")

    if run:
        from aifund.agents.live_agent_v2 import LiveTradingAgentV2
        from aifund.data.pipeline import DataPipeline
        from aifund.strategies import all_strategies
        from aifund.stockpool.factor_loader import FactorLoader

        # ── 资源缓存：FactorLoader (55MB CSV) 和 DataPipeline 整个会话只构造一次 ──
        @st.cache_resource(show_spinner=False)
        def _get_pipeline():
            p = DataPipeline()
            p.light_mode = True
            return p

        @st.cache_resource(show_spinner=False)
        def _get_factor_loader():
            return FactorLoader()

        t0 = time.time()
        with st.status("V2 推荐中…", expanded=True) as status:
            ts1 = time.time()
            status.write("📦 加载数据管道 + 因子库(会话内只构造一次)…")
            pipeline = _get_pipeline()
            fl = _get_factor_loader()
            status.write(f"   ✓ 用时 {time.time()-ts1:.2f}s")

            ts2 = time.time()
            status.write(f"🎯 PIT 选股池 ≤ {target}…")
            pit_uni = fl.pit_top_n(target, n=int(universe_size))
            individual_uni = pit_uni["symbol"].tolist()
            status.write(f"   ✓ {len(individual_uni)} 只,用时 {time.time()-ts2:.2f}s")

            ts3 = time.time()
            status.write("🧩 加载 4 策略 + 初始化 V2 智能体…")
            strats = all_strategies(pipeline, universe=individual_uni)
            agent = LiveTradingAgentV2(pipeline=pipeline, strategies=strats,
                                        offline=not use_llm)
            status.write(f"   ✓ 用时 {time.time()-ts3:.2f}s")

            ts4 = time.time()
            mode_label = "LLM 推理(R1)" if use_llm else "offline 规则判断"
            status.write(f"🤖 调用 agent.recommend({target}) · 模式 {mode_label}…")
            recs, decision = agent.recommend(target)
            status.write(f"   ✓ 推荐 {len(recs)} 只,用时 {time.time()-ts4:.2f}s")

            elapsed = time.time() - t0
            status.update(label=f"V2 推荐完成 · 总耗时 {elapsed:.2f}s",
                          state="complete", expanded=False)

        # 状态条
        if individual_uni:
            pit_day = pit_uni["日期"].iloc[0]
            uni_info = f"{len(individual_uni)} 只(PIT 因子 @ {pit_day})"
        else:
            uni_info = "空(早于因子数据)"
        mode_info = "LLM 推理" if use_llm else "offline 规则"
        st.markdown(f"""
        <div style='display:flex; gap:0.5rem; margin:0.5rem 0 1rem; flex-wrap:wrap;'>
          <div style='padding:0.3rem 0.7rem; background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                      border-radius:3px; font-family:JetBrains Mono; font-size:0.75rem;
                      color:{COLOR_GOLD_LIGHT};'>⏱ {elapsed:.1f}s</div>
          <div style='padding:0.3rem 0.7rem; background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                      border-radius:3px; font-family:JetBrains Mono; font-size:0.75rem;
                      color:{COLOR_TEXT};'>🧮 {mode_info}</div>
          <div style='padding:0.3rem 0.7rem; background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                      border-radius:3px; font-family:JetBrains Mono; font-size:0.75rem;
                      color:{COLOR_TEXT};'>🎯 universe {uni_info}</div>
        </div>
        """, unsafe_allow_html=True)

        # 渲染三层
        render_decision_layers({
            "regime": decision.regime,
            "weights": decision.weights,
            "recs": recs,
            "n_recs": len(recs),
        })

        # JSON 输出
        with st.expander("📋 完整 JSON(评委可拿这份复现)"):
            st.code(json.dumps({
                "as_of": str(target),
                "regime": decision.regime,
                "weights": decision.weights,
                "recommendations": recs,
            }, ensure_ascii=False, indent=2), language="json")
    else:
        st.info(f"👆 选择日期后点「生成推荐」—— offline 模式通常 **1-2 秒**返回"
                f"(首次会话载入数据稍慢,后续秒回)。")


# ═════════════════════════ 页脚 ═════════════════════════
st.markdown(f"""
<div style='margin-top:3rem; padding-top:1rem; border-top:1px solid {COLOR_BORDER};
           color:{COLOR_MUTED}; font-size:0.72rem; text-align:center;
           font-family:JetBrains Mono;'>
  ↳ data: runs/v2_*.json · 100% reproducible offline · CUEB Tuoling 2026
</div>
""", unsafe_allow_html=True)
