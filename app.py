"""Index Construction Dashboard — minimalist, bilingual (中文 / English)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.decomposition import PCA

# ---------------- i18n ----------------
TEXTS = {
    "zh": {
        "page_title": "Index Construction",
        "title": "Index Construction",
        "subtitle": "自訂成分股組合，建構你的指數。資料：Yahoo Finance",
        "settings": "設定",
        "language": "語言",
        "tickers": "股票代碼",
        "tickers_help": "用逗號分隔。例：AAPL, MSFT, 2330.TW",
        "weight": "權重方式",
        "weight_equal": "等權重",
        "weight_cap": "市值加權",
        "weight_float": "自由流通市值",
        "period": "期間",
        "preset": "快速選擇",
        "custom": "自訂",
        "presets": ["自訂", "1 個月", "3 個月", "6 個月", "年初至今", "1 年", "3 年", "5 年"],
        "start": "起始日",
        "end": "結束日",
        "base": "指數基準值",
        "run": "載入資料",
        "hint": "請在左側設定後點選「載入資料」",
        "empty_tickers": "請輸入至少一支股票代碼",
        "loading_prices": "抓取股價中…",
        "loading_shares": "讀取股本資訊中…",
        "missing": "查無資料的代碼",
        "no_data": "沒有可用的股價資料，請檢查代碼或日期",
        "kpi_value": "指數現值",
        "kpi_return": "區間報酬",
        "kpi_ann": "年化報酬",
        "kpi_vol": "年化波動",
        "kpi_dd": "最大回撤",
        "chart_index": "指數走勢",
        "chart_components": "成分股表現 (歸一化 = 100)",
        "summary": "成分股摘要",
        "weights": "權重分布",
        "shares": "股本與市值",
        "col_name": "名稱",
        "col_ccy": "幣別",
        "col_price": "最新價",
        "col_cap": "市值",
        "col_total": "總股數",
        "col_float": "自由流通股數",
        "col_floatpct": "Float %",
        "col_weight": "權重 %",
        "col_open": "起始價",
        "col_close": "最新價",
        "col_pnl": "報酬 %",
        "total_row": "合計",
        "m_cap": "指數總市值",
        "m_total": "總股數合計",
        "m_float": "自由流通股數合計",
        "download": "下載 CSV",
        "base_label": "基準",
        "pca_title": "因子分析 (PCA)",
        "pca_caption": "用主成分分析找出驅動你指數的隱藏因子。PC1 通常代表「整體市場」，PC2、PC3 則是次要主題（如產業偏好）。",
        "pca_variance": "各因子解釋變異量",
        "pca_loadings": "成分股在因子上的權重 (Loadings)",
        "pca_need_more": "PCA 需要至少 2 支股票與 30 天以上資料。",
        "pca_factor": "因子",
        "pca_explained": "解釋變異 %",
        "pca_interpret_pc1": "PC1 解釋了 {p:.0f}% 的變異 — 通常代表整體市場走勢（系統性風險）",
        "pca_interpret_pc2": "PC2 解釋了 {p:.0f}% — 通常代表產業 / 風格分歧",
        "pca_interpret_pc3": "PC3 解釋了 {p:.0f}% — 個別公司特有風險",
    },
    "en": {
        "page_title": "Index Construction",
        "title": "Index Construction",
        "subtitle": "Construct your custom equity index. Source: Yahoo Finance",
        "settings": "Settings",
        "language": "Language",
        "tickers": "Tickers",
        "tickers_help": "Comma-separated. e.g. AAPL, MSFT, 2330.TW",
        "weight": "Weighting",
        "weight_equal": "Equal weight",
        "weight_cap": "Market cap",
        "weight_float": "Free float",
        "period": "Period",
        "preset": "Quick select",
        "custom": "Custom",
        "presets": ["Custom", "1 Month", "3 Months", "6 Months", "YTD", "1 Year", "3 Years", "5 Years"],
        "start": "Start",
        "end": "End",
        "base": "Index base value",
        "run": "Load data",
        "hint": "Configure on the left, then click \"Load data\"",
        "empty_tickers": "Please enter at least one ticker",
        "loading_prices": "Fetching prices…",
        "loading_shares": "Fetching share info…",
        "missing": "Tickers with no data",
        "no_data": "No price data available — check tickers or date range",
        "kpi_value": "Index value",
        "kpi_return": "Period return",
        "kpi_ann": "Annualized",
        "kpi_vol": "Annualized vol",
        "kpi_dd": "Max drawdown",
        "chart_index": "Index performance",
        "chart_components": "Component performance (normalized = 100)",
        "summary": "Component summary",
        "weights": "Weight distribution",
        "shares": "Shares & market cap",
        "col_name": "Name",
        "col_ccy": "Ccy",
        "col_price": "Last",
        "col_cap": "Market cap",
        "col_total": "Total shares",
        "col_float": "Free float",
        "col_floatpct": "Float %",
        "col_weight": "Weight %",
        "col_open": "Open",
        "col_close": "Last",
        "col_pnl": "Return %",
        "total_row": "TOTAL",
        "m_cap": "Total market cap",
        "m_total": "Total shares",
        "m_float": "Total free float",
        "download": "Download CSV",
        "base_label": "Base",
        "pca_title": "Factor analysis (PCA)",
        "pca_caption": "Principal Component Analysis uncovers the hidden factors driving your index. PC1 typically captures broad market movement; PC2/PC3 reveal secondary themes like sector tilts.",
        "pca_variance": "Variance explained by each factor",
        "pca_loadings": "Component loadings on each factor",
        "pca_need_more": "PCA requires at least 2 tickers and 30+ days of data.",
        "pca_factor": "Factor",
        "pca_explained": "Explained %",
        "pca_interpret_pc1": "PC1 explains {p:.0f}% of variance — typically the broad market move (systematic risk)",
        "pca_interpret_pc2": "PC2 explains {p:.0f}% — typically a sector / style tilt",
        "pca_interpret_pc3": "PC3 explains {p:.0f}% — idiosyncratic / company-specific risk",
    },
}

# ---------------- Page config & minimalist styling ----------------
st.set_page_config(page_title="Index Construction", layout="wide", initial_sidebar_state="expanded")

# Warm Terracotta palette — minimalist, warm, refined
ACCENT = "#C75B3C"   # terracotta
ACCENT_DARK = "#9E4429"
INK = "#3D2B1F"      # warm dark brown (text)
MUTED = "#8A7968"    # warm taupe
LINE = "#EDE3D6"     # cream divider
BG = "#FBF6F0"       # cream background
CARD = "#FFFFFF"
PALETTE = ["#C75B3C", "#3D2B1F", "#D9A679", "#8A7968", "#6B4F3A", "#E8C39E", "#A0522D"]

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {INK}; }}
    h1, h2, h3, h4, p, label, span, div {{ color: {INK}; }}
    h1, h2, h3, h4 {{ font-weight: 500; letter-spacing: -0.01em; }}
    h1 {{ font-size: 1.9rem; color: {ACCENT}; }}
    h2 {{ font-size: 1.1rem; margin-top: 1.5rem; color: {INK}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
    [data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{ font-weight: 500; color: {ACCENT}; }}
    [data-testid="stMetricDelta"] {{ color: {MUTED}; }}
    .stButton>button {{
        background: {ACCENT}; color: #fff; border: none;
        border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500;
        transition: background 0.15s;
    }}
    .stButton>button:hover {{ background: {ACCENT_DARK}; color: #fff; }}
    section[data-testid="stSidebar"] {{ background: #F5EDE2; border-right: 1px solid {LINE}; }}
    section[data-testid="stSidebar"] * {{ color: {INK}; }}
    .stDataFrame {{ border: 1px solid {LINE}; border-radius: 8px; }}
    .stDownloadButton>button {{
        background: transparent; color: {ACCENT}; border: 1px solid {ACCENT};
        border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500;
    }}
    .stDownloadButton>button:hover {{ background: {ACCENT}; color: #fff; }}
    hr {{ border-color: {LINE}; }}
    .caption {{ color: {MUTED}; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }}
    input, textarea, [data-baseweb="select"] > div {{ background: {CARD} !important; border-color: {LINE} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    lang = st.radio("Language / 語言", ["English", "中文"], index=0, horizontal=True)
L = "en" if lang == "English" else "zh"
T = TEXTS[L]

with st.sidebar:
    st.markdown(f"### {T['settings']}")

    tickers_input = st.text_area(
        T["tickers"],
        value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
        help=T["tickers_help"],
        height=80,
    )

    weight_label_map = {
        T["weight_equal"]: "equal",
        T["weight_cap"]: "cap",
        T["weight_float"]: "float",
    }
    weight_choice = st.radio(T["weight"], list(weight_label_map.keys()), index=0)
    weight_mode = weight_label_map[weight_choice]

    st.markdown("---")
    st.markdown(f"**{T['period']}**")

    preset = st.selectbox(T["preset"], T["presets"], index=5)
    today = date.today()
    preset_idx = T["presets"].index(preset)
    deltas = [None, 30, 90, 180, "ytd", 365, 365 * 3, 365 * 5]
    d = deltas[preset_idx]
    if d == "ytd":
        start_default = date(today.year, 1, 1)
    elif d is None:
        start_default = today - timedelta(days=365)
    else:
        start_default = today - timedelta(days=d)

    c1, c2 = st.columns(2)
    start_date = c1.date_input(T["start"], value=start_default, max_value=today)
    end_date = c2.date_input(T["end"], value=today, max_value=today)

    base_value = st.number_input(T["base"], value=100, step=10)

    st.markdown("")
    run = st.button(T["run"], use_container_width=True)


# ---------------- Data ----------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if len(tickers) == 1:
        return pd.DataFrame({tickers[0]: data["Close"]})
    closes = {t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]}
    return pd.DataFrame(closes).dropna(how="all")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_share_info(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}
        total = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0
        float_ = info.get("floatShares") or 0
        cap = info.get("marketCap") or 0
        rows.append({
            "Ticker": t,
            "Name": info.get("shortName") or info.get("longName") or t,
            "Currency": info.get("currency", ""),
            "MarketCap": float(cap),
            "TotalShares": float(total),
            "FloatShares": float(float_),
            "FloatPct": (float(float_) / float(total) * 100) if total else 0.0,
        })
    return pd.DataFrame(rows).set_index("Ticker")


def build_index(prices: pd.DataFrame, weights: dict[str, float], base: float) -> pd.Series:
    normalized = prices.divide(prices.iloc[0])
    weighted = normalized.multiply(pd.Series(weights))
    return weighted.sum(axis=1) * base


def fmt_big(n: float) -> str:
    if not n:
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:,.2f}{unit}"
    return f"{n:,.0f}"


# ---------------- Main ----------------
st.markdown(f"# {T['title']}")
st.markdown(f"<div class='caption'>{T['subtitle']}</div>", unsafe_allow_html=True)

tickers = tuple(t.strip().upper() for t in tickers_input.split(",") if t.strip())

if not tickers:
    st.info(T["empty_tickers"])
    st.stop()

if not run and "loaded" not in st.session_state:
    st.info(T["hint"])
    st.stop()
st.session_state["loaded"] = True

with st.spinner(T["loading_prices"]):
    prices = fetch_prices(tickers, start_date, end_date)

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"{T['missing']}: {', '.join(missing)}")
prices = prices.dropna(axis=1, how="all").ffill().dropna()

if prices.empty or prices.shape[1] == 0:
    st.error(T["no_data"])
    st.stop()

with st.spinner(T["loading_shares"]):
    share_info = fetch_share_info(tuple(prices.columns))

# Weights
if weight_mode == "cap":
    caps = share_info["MarketCap"].to_dict()
    total = sum(caps.values()) or 1
    weights = {t: caps.get(t, 0) / total for t in prices.columns}
elif weight_mode == "float":
    floats = share_info["FloatShares"].to_dict()
    last = prices.iloc[-1]
    ff_cap = {t: floats.get(t, 0) * last[t] for t in prices.columns}
    total = sum(ff_cap.values()) or 1
    weights = {t: ff_cap[t] / total for t in prices.columns}
else:
    n = prices.shape[1]
    weights = {t: 1 / n for t in prices.columns}

index_series = build_index(prices, weights, base_value)

# KPIs
start_val = index_series.iloc[0]
end_val = index_series.iloc[-1]
total_return = (end_val / start_val - 1) * 100
days = (index_series.index[-1] - index_series.index[0]).days or 1
ann_return = ((end_val / start_val) ** (365 / days) - 1) * 100
daily_ret = index_series.pct_change().dropna()
volatility = daily_ret.std() * (252 ** 0.5) * 100
max_dd = ((index_series / index_series.cummax()) - 1).min() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(T["kpi_value"], f"{end_val:,.2f}", f"{end_val - base_value:+.2f}")
k2.metric(T["kpi_return"], f"{total_return:+.2f}%")
k3.metric(T["kpi_ann"], f"{ann_return:+.2f}%")
k4.metric(T["kpi_vol"], f"{volatility:.2f}%")
k5.metric(T["kpi_dd"], f"{max_dd:.2f}%")

# Helper for minimal plotly layout
def _minimal(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family="-apple-system, system-ui, sans-serif", size=12, color=INK),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=""),
        xaxis=dict(showgrid=False, showline=True, linecolor=LINE, ticks="outside", tickcolor=LINE),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False),
    )
    return fig


# Index chart
st.markdown(f"## {T['chart_index']}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=index_series.index, y=index_series.values,
    line=dict(width=2.5, color=ACCENT), name="Index",
    fill="tozeroy", fillcolor="rgba(199,91,60,0.08)",
))
fig.add_hline(y=base_value, line_dash="dot", line_color=MUTED,
              annotation_text=f"{T['base_label']} {base_value}", annotation_font_color=MUTED)
fig.update_yaxes(range=[min(index_series.min(), base_value) * 0.98, index_series.max() * 1.02])
st.plotly_chart(_minimal(fig, 400), use_container_width=True)

# Component chart
st.markdown(f"## {T['chart_components']}")
norm = prices.divide(prices.iloc[0]) * 100
fig2 = px.line(norm, x=norm.index, y=norm.columns, color_discrete_sequence=PALETTE)
fig2.update_traces(line=dict(width=1.5))
st.plotly_chart(_minimal(fig2, 380), use_container_width=True)

# Summary + pie
ca, cb = st.columns([1.5, 1])
with ca:
    st.markdown(f"## {T['summary']}")
    summary = pd.DataFrame({
        T["col_name"]: share_info["Name"].astype(str),
        T["col_weight"]: [round(weights[t] * 100, 2) for t in prices.columns],
        T["col_open"]: prices.iloc[0].round(2).values,
        T["col_close"]: prices.iloc[-1].round(2).values,
        T["col_pnl"]: ((prices.iloc[-1] / prices.iloc[0] - 1) * 100).round(2).values,
    }, index=prices.columns)
    st.dataframe(summary, use_container_width=True, height=300)

with cb:
    st.markdown(f"## {T['weights']}")
    pie = px.pie(
        values=list(weights.values()), names=list(weights.keys()),
        hole=0.55, color_discrete_sequence=PALETTE,
    )
    pie.update_traces(textposition="outside", textinfo="label+percent",
                      marker=dict(line=dict(color=BG, width=2)))
    st.plotly_chart(_minimal(pie, 320), use_container_width=True)

# Shares
st.markdown(f"## {T['shares']}")
shares_table = pd.DataFrame({
    T["col_name"]: share_info["Name"].astype(str),
    T["col_ccy"]: share_info["Currency"].astype(str),
    T["col_price"]: [f"{v:,.2f}" for v in prices.iloc[-1].values],
    T["col_cap"]: share_info["MarketCap"].map(fmt_big),
    T["col_total"]: share_info["TotalShares"].map(fmt_big),
    T["col_float"]: share_info["FloatShares"].map(fmt_big),
    T["col_floatpct"]: share_info["FloatPct"].round(2).astype(str),
}, index=share_info.index)

total_cap = share_info["MarketCap"].sum()
total_shares_sum = share_info["TotalShares"].sum()
total_float = share_info["FloatShares"].sum()
shares_table.loc[T["total_row"]] = [
    "—", "—", "—",
    fmt_big(total_cap), fmt_big(total_shares_sum), fmt_big(total_float),
    f"{(total_float / total_shares_sum * 100):.2f}" if total_shares_sum else "0",
]
st.dataframe(shares_table, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric(T["m_cap"], fmt_big(total_cap))
m2.metric(T["m_total"], fmt_big(total_shares_sum))
m3.metric(T["m_float"], fmt_big(total_float))

# ---------------- Factor Analysis (PCA) ----------------
st.markdown(f"## {T['pca_title']}")
st.markdown(f"<div class='caption'>{T['pca_caption']}</div>", unsafe_allow_html=True)

returns = prices.pct_change().dropna()
if returns.shape[1] < 2 or returns.shape[0] < 30:
    st.info(T["pca_need_more"])
else:
    n_comp = min(3, returns.shape[1])
    pca = PCA(n_components=n_comp).fit(returns.values)
    explained = pca.explained_variance_ratio_ * 100  # %
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )

    pca_left, pca_right = st.columns([1, 1.6])

    with pca_left:
        var_df = pd.DataFrame({
            T["pca_factor"]: [f"PC{i+1}" for i in range(n_comp)],
            T["pca_explained"]: explained.round(2),
        })
        fig_var = px.bar(
            var_df, x=T["pca_factor"], y=T["pca_explained"],
            text=T["pca_explained"], color_discrete_sequence=[ACCENT],
        )
        fig_var.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_var.update_yaxes(range=[0, max(explained) * 1.15])
        st.plotly_chart(_minimal(fig_var, 320), use_container_width=True)

    with pca_right:
        # Heatmap of loadings — diverging palette around 0
        fig_load = px.imshow(
            loadings.values,
            x=loadings.columns,
            y=loadings.index,
            color_continuous_scale=[[0, "#3D2B1F"], [0.5, BG], [1, ACCENT]],
            zmin=-max(abs(loadings.values.min()), abs(loadings.values.max())),
            zmax=max(abs(loadings.values.min()), abs(loadings.values.max())),
            text_auto=".2f",
            aspect="auto",
        )
        fig_load.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(family="-apple-system, sans-serif", size=12, color=INK),
            coloraxis_showscale=False,
        )
        st.markdown(f"**{T['pca_loadings']}**")
        st.plotly_chart(fig_load, use_container_width=True)

    # Plain-language interpretation
    interp = []
    if n_comp >= 1:
        interp.append(T["pca_interpret_pc1"].format(p=explained[0]))
    if n_comp >= 2:
        interp.append(T["pca_interpret_pc2"].format(p=explained[1]))
    if n_comp >= 3:
        interp.append(T["pca_interpret_pc3"].format(p=explained[2]))
    for line in interp:
        st.markdown(f"• {line}")

# Download
st.markdown("---")
download_df = prices.copy()
download_df["Index"] = index_series
st.download_button(
    T["download"],
    download_df.to_csv().encode("utf-8"),
    file_name=f"index_construction_{start_date}_{end_date}.csv",
    mime="text/csv",
)
