"""Generate a styled, bilingual (EN / 中文) PDF report for an Index Construction run.

Each chart / data section is followed by a 📊 takeaway callout that interprets
the numbers in plain language so the reader doesn't have to be a quant to
understand what the chart is saying.
"""
from __future__ import annotations

import os
from datetime import date
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# Match the dashboard's terracotta theme
ACCENT = "#C75B3C"
ACCENT_DARK = "#9E4429"
INK = "#3D2B1F"
MUTED = "#8A7968"
LINE = "#EDE3D6"
BG = "#FBF6F0"
CARD = "#FFFFFF"
CALLOUT_BG = "#FBEEE7"


# ---------- Font setup (run once per process) ----------
_CJK_FONT_NAME_RL = "Helvetica"          # detected on first call to _setup_fonts()
_CJK_FONT_NAME_MPL: str | None = None
_FONTS_READY = False


# Candidate TTF/OTF/TTC paths — first hit wins. Order matters: prefer
# Traditional Chinese coverage (TC variants), then Simplified, then Japanese.
_TTF_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Songti.ttc",
    # Streamlit Cloud Linux (after `fonts-noto-cjk` from packages.txt)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # Generic Linux fallbacks
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _find_ttf() -> str | None:
    for p in _TTF_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _setup_fonts() -> None:
    """Register a real TTF CJK font for reportlab and matplotlib (once per process)."""
    global _CJK_FONT_NAME_RL, _CJK_FONT_NAME_MPL, _FONTS_READY
    if _FONTS_READY:
        return

    # ---- reportlab ----
    ttf = _find_ttf()
    registered = False
    if ttf:
        # .ttc files contain multiple sub-fonts — try the first few until one works
        for idx in (0, 1, 2):
            try:
                pdfmetrics.registerFont(TTFont("CJK", ttf, subfontIndex=idx))
                _CJK_FONT_NAME_RL = "CJK"
                registered = True
                break
            except Exception:
                continue
    if not registered:
        # Last resort: MSung-Light covers Traditional Chinese better than STSong-Light
        for cid_name in ("MSung-Light", "STSong-Light"):
            try:
                pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
                _CJK_FONT_NAME_RL = cid_name
                break
            except Exception:
                continue

    # ---- matplotlib ----
    if ttf:
        # Register the same file with matplotlib so chart labels are not boxes
        try:
            fm.fontManager.addfont(ttf)
            # font_manager caches the name after addfont
            prop = fm.FontProperties(fname=ttf)
            _CJK_FONT_NAME_MPL = prop.get_name()
        except Exception:
            pass
    if _CJK_FONT_NAME_MPL is None:
        # Fall back to whatever's already in the font cache
        candidates = ["Noto Sans CJK TC", "Noto Sans CJK SC",
                      "PingFang TC", "PingFang SC", "Heiti TC", "Heiti SC",
                      "Hiragino Sans GB", "Microsoft JhengHei", "SimHei"]
        available = {f.name for f in fm.fontManager.ttflist}
        for c in candidates:
            if c in available:
                _CJK_FONT_NAME_MPL = c
                break
    if _CJK_FONT_NAME_MPL:
        matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT_NAME_MPL, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

    _FONTS_READY = True


# ---------- i18n text dictionary ----------
TXT = {
    "en": {
        "report_title": "Index Construction Report",
        "subtitle": "Generated {today} · Period {start} to {end} · {n} constituents · {weight} weighting",
        "h_perf": "Performance",
        "h_drawdown": "Drawdown",
        "h_risk_adj": "Risk-adjusted performance",
        "h_constituents": "Constituents",
        "h_sector": "Sector breakdown",
        "h_corr": "Component correlations",
        "h_pca": "Factor analysis (PCA)",
        "h_methodology": "Methodology & disclaimer",
        "takeaway_label": "Key takeaway",
        "metric": "Metric", "value": "Value",
        "final_value": "Final value",
        "period_return": "Period return",
        "ann_return": "Annualized return",
        "ann_vol": "Annualized vol",
        "max_dd": "Max drawdown",
        "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar", "beta": "Beta",
        "benchmark": "Benchmark",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector",
        "col_weight": "Weight %", "col_cap": "Market Cap",
        "title_perf_chart": "Index performance",
        "title_dd_chart": "Drawdown (% from rolling peak)",
        "title_sector_chart": "Sector breakdown",
        "title_corr_chart": "Component correlations",
        "caption_corr": "Pairwise correlation of daily returns. Dark cells = move together (low diversification); light or negative cells = genuine diversification.",
        "caption_pca": "Principal Component Analysis decomposes return variance into orthogonal factors.",
        "methodology": (
            "Prices are total-return-adjusted closes from Yahoo Finance via yfinance. "
            "Returns and volatility use daily log changes annualized with 252 trading days. "
            "Max drawdown is the largest peak-to-trough decline of the cumulative index. "
            "Beta = cov(index, benchmark) / var(benchmark) over daily returns. "
            "PCA is fitted on the daily return matrix using scikit-learn."
        ),
        "disclaimer": "This report is for educational and research purposes only. It does not constitute investment advice. Past performance is not indicative of future results.",
        # interpretation phrases
        "tk_perf": "The index returned <b>{tot:+.2f}%</b> ({ann:+.2f}% annualized). {bench_line}",
        "bench_outperf": "It <b>outperformed</b> {b} by <b>{d:.2f} pts</b>.",
        "bench_underperf": "It <b>underperformed</b> {b} by <b>{d:.2f} pts</b>.",
        "tk_dd": "Max drawdown was <b>{dd:.2f}%</b>. {bench_dd}",
        "tk_dd_bench_worse": "Your index was {gap:.1f} pts worse than {b}.",
        "tk_dd_bench_better": "Your index was {gap:.1f} pts shallower than {b} — better downside protection.",
        "tk_sector_concentrated": "Top sector ({sec}) is {pct:.0f}% — concentrated, consider broadening.",
        "tk_sector_themed": "Top sector ({sec}) is {pct:.0f}% — typical for a thematic basket.",
        "tk_sector_diverse": "Top sector ({sec}) is {pct:.0f}% — well diversified across sectors.",
        "tk_corr": "Average pairwise correlation is <b>{avg:.2f}</b>. Most correlated pair: {p_pair} ({p_v:.2f}). Least correlated: {n_pair} ({n_v:.2f}).",
        "tk_pca_concentrated": "PC1 explains {p1:.0f}% — a single market factor dominates. The components move in lockstep.",
        "tk_pca_typical": "PC1 explains {p1:.0f}% — broad market beta is the primary driver, typical for sector-themed baskets.",
        "tk_pca_diverse": "PC1 explains {p1:.0f}% — surprisingly diversified return drivers.",
        "tk_constituents": "Largest weight: <b>{max_t}</b> ({max_w:.1f}%). Smallest: {min_t} ({min_w:.1f}%).",
        # Sharpe interp
        "sharpe_excellent": "Sharpe {s:.2f} — Excellent risk-adjusted return.",
        "sharpe_good": "Sharpe {s:.2f} — Good. Returns comfortably justify the volatility taken.",
        "sharpe_moderate": "Sharpe {s:.2f} — Moderate. The index pays for risk, but not by much.",
        "sharpe_weak": "Sharpe {s:.2f} — Weak. You're barely compensated for the volatility.",
        "sharpe_negative": "Sharpe {s:.2f} — Negative. Underperformed the risk-free rate.",
        "sortino_explain": "Sortino {s:.2f} — like Sharpe but only penalizes downside volatility. Higher than Sharpe means volatility skews upward.",
        "calmar_explain": "Calmar {s:.2f} — annual return ÷ max drawdown. Above 1.0 means yearly gains exceed the worst dip.",
        "beta_high": "Beta {b:.2f} vs {bench} — High beta. Expect amplified moves.",
        "beta_market": "Beta {b:.2f} vs {bench} — Roughly market-like sensitivity.",
        "beta_defensive": "Beta {b:.2f} vs {bench} — Defensive, less sensitive than the benchmark.",
        "beta_low": "Beta {b:.2f} vs {bench} — Very low or negative correlation with the benchmark.",
    },
    "zh": {
        "report_title": "指數建構報告",
        "subtitle": "產出日期 {today}．期間 {start} 至 {end}．{n} 檔成分股．{weight} 加權",
        "h_perf": "績效表現",
        "h_drawdown": "回撤分析",
        "h_risk_adj": "風險調整後表現",
        "h_constituents": "成分股",
        "h_sector": "產業分布",
        "h_corr": "成分股相關係數",
        "h_pca": "因子分析 (PCA)",
        "h_methodology": "方法論與免責聲明",
        "takeaway_label": "📊 小結論",
        "metric": "指標", "value": "數值",
        "final_value": "最終值",
        "period_return": "區間報酬",
        "ann_return": "年化報酬",
        "ann_vol": "年化波動",
        "max_dd": "最大回撤",
        "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar", "beta": "Beta",
        "benchmark": "對標基準",
        "col_ticker": "代碼", "col_name": "名稱", "col_sector": "產業",
        "col_weight": "權重 %", "col_cap": "市值",
        "title_perf_chart": "指數走勢",
        "title_dd_chart": "回撤 (距歷史高點 %)",
        "title_sector_chart": "產業分布",
        "title_corr_chart": "成分股相關係數",
        "caption_corr": "成分股日報酬的兩兩相關係數矩陣。深色 = 同向動（分散不足）；淺色或負值 = 真正分散。",
        "caption_pca": "主成分分析將報酬變異拆解為彼此正交的因子。",
        "methodology": (
            "價格資料取自 Yahoo Finance（透過 yfinance），已含股息再投入調整。"
            "報酬與波動度以日對數報酬計算，年化採 252 交易日。"
            "最大回撤為累積指數從歷史高點至最低點的最大跌幅。"
            "Beta = cov(指數, 基準) / var(基準)，以日報酬計算。"
            "PCA 以 scikit-learn 對日報酬矩陣進行擬合。"
        ),
        "disclaimer": "本報告僅供教育與研究用途，不構成投資建議。過去績效不代表未來表現。",
        "tk_perf": "指數區間總報酬 <b>{tot:+.2f}%</b>（年化 {ann:+.2f}%）。{bench_line}",
        "bench_outperf": "<b>跑贏</b> {b} 約 <b>{d:.2f} 個百分點</b>。",
        "bench_underperf": "<b>跑輸</b> {b} 約 <b>{d:.2f} 個百分點</b>。",
        "tk_dd": "最大回撤為 <b>{dd:.2f}%</b>。{bench_dd}",
        "tk_dd_bench_worse": "比 {b} 多跌 {gap:.1f} 個百分點。",
        "tk_dd_bench_better": "比 {b} 少跌 {gap:.1f} 個百分點，下跌防禦較佳。",
        "tk_sector_concentrated": "最大產業 {sec} 占 {pct:.0f}%，集中度偏高，建議分散。",
        "tk_sector_themed": "最大產業 {sec} 占 {pct:.0f}%，主題型組合典型水準。",
        "tk_sector_diverse": "最大產業 {sec} 占 {pct:.0f}%，產業分布相當分散。",
        "tk_corr": "平均兩兩相關係數為 <b>{avg:.2f}</b>。最相關：{p_pair}（{p_v:.2f}）；最不相關：{n_pair}（{n_v:.2f}）。",
        "tk_pca_concentrated": "PC1 解釋了 {p1:.0f}% — 單一市場因子主宰，成分股幾乎同向動。",
        "tk_pca_typical": "PC1 解釋了 {p1:.0f}% — 整體市場 beta 為主驅動，主題型組合典型水準。",
        "tk_pca_diverse": "PC1 解釋了 {p1:.0f}% — 報酬驅動因子相當分散。",
        "tk_constituents": "最大權重：<b>{max_t}</b>（{max_w:.1f}%）；最小：{min_t}（{min_w:.1f}%）。",
        "sharpe_excellent": "Sharpe {s:.2f} — 風險調整後報酬極佳。",
        "sharpe_good": "Sharpe {s:.2f} — 表現良好，報酬足以補償所承擔的波動。",
        "sharpe_moderate": "Sharpe {s:.2f} — 中等水準，有補償但不算豐厚。",
        "sharpe_weak": "Sharpe {s:.2f} — 偏弱，幾乎沒補償到波動。",
        "sharpe_negative": "Sharpe {s:.2f} — 負值，跑輸無風險利率。",
        "sortino_explain": "Sortino {s:.2f} — 類似 Sharpe，但只懲罰下行波動。若高於 Sharpe，代表波動偏向上行。",
        "calmar_explain": "Calmar {s:.2f} — 年化報酬除以最大回撤。> 1.0 代表年度賺的比最痛時刻虧的多。",
        "beta_high": "Beta {b:.2f} vs {bench} — 高 beta，預期漲跌幅放大。",
        "beta_market": "Beta {b:.2f} vs {bench} — 與大盤接近的敏感度。",
        "beta_defensive": "Beta {b:.2f} vs {bench} — 防禦型，較不敏感。",
        "beta_low": "Beta {b:.2f} vs {bench} — 與基準關聯極低或為負。",
    },
}


# ---------- chart helpers ----------
def _style_axes(ax, lang: str):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, axis="y", color=LINE, linewidth=0.6)
    ax.title.set_color(INK)


def _fig_to_buf(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf


def _index_chart(index_series, benchmark, base, bench_label, t):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.fill_between(index_series.index, index_series.values, base, alpha=0.10, color=ACCENT)
    ax.plot(index_series.index, index_series.values, color=ACCENT, linewidth=2.0, label="Index")
    if benchmark is not None and not benchmark.empty:
        bench_norm = benchmark / benchmark.iloc[0] * base
        ax.plot(bench_norm.index, bench_norm.values, color=INK, linewidth=1.4,
                linestyle="--", label=f"{bench_label}")
    ax.axhline(base, color=MUTED, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_title(t["title_perf_chart"], loc="left", fontsize=11, weight="medium")
    _style_axes(ax, "")
    return _fig_to_buf(fig)


def _drawdown_chart(index_series, benchmark, bench_label, t):
    dd = (index_series / index_series.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.20, color=ACCENT_DARK)
    ax.plot(dd.index, dd.values, color=ACCENT_DARK, linewidth=1.2, label="Index")
    if benchmark is not None and not benchmark.empty:
        bdd = (benchmark / benchmark.cummax() - 1) * 100
        ax.plot(bdd.index, bdd.values, color=INK, linestyle="--", linewidth=1.0, label=bench_label)
        ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.axhline(0, color=LINE, linewidth=0.8)
    ax.set_title(t["title_dd_chart"], loc="left", fontsize=11, weight="medium")
    _style_axes(ax, "")
    return _fig_to_buf(fig)


def _sector_chart(sector_weights, t):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    palette = ["#C75B3C", "#3D2B1F", "#D9A679", "#8A7968", "#6B4F3A", "#E8C39E", "#A0522D", "#B07B5A"]
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    ax.pie(values, labels=labels, colors=palette[:len(labels)],
           autopct="%1.0f%%", textprops={"color": INK, "fontsize": 8.5},
           wedgeprops=dict(width=0.45, edgecolor=BG))
    ax.set_title(t["title_sector_chart"], loc="left", fontsize=11, weight="medium", color=INK)
    fig.patch.set_facecolor(BG)
    return _fig_to_buf(fig)


def _corr_chart(corr, t):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "terracotta_div", [INK, BG, ACCENT]
    )
    ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, color=INK, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(corr.index, color=INK, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color=INK, fontsize=7)
    ax.set_title(t["title_corr_chart"], loc="left", fontsize=11, weight="medium", color=INK)
    fig.patch.set_facecolor(BG)
    return _fig_to_buf(fig)


# ---------- takeaway computation ----------
def _sharpe_text(s, t):
    if s >= 2: return t["sharpe_excellent"].format(s=s)
    if s >= 1: return t["sharpe_good"].format(s=s)
    if s >= 0.5: return t["sharpe_moderate"].format(s=s)
    if s >= 0: return t["sharpe_weak"].format(s=s)
    return t["sharpe_negative"].format(s=s)


def _beta_text(b, bench, t):
    if b is None: return None
    if b >= 1.2: return t["beta_high"].format(b=b, bench=bench)
    if b >= 0.8: return t["beta_market"].format(b=b, bench=bench)
    if b >= 0.3: return t["beta_defensive"].format(b=b, bench=bench)
    return t["beta_low"].format(b=b, bench=bench)


def _sector_text(sector_weights, t):
    if not sector_weights:
        return ""
    sec, w = max(sector_weights.items(), key=lambda kv: kv[1])
    pct = w * 100
    if pct >= 50:
        return t["tk_sector_concentrated"].format(sec=sec, pct=pct)
    if pct >= 30:
        return t["tk_sector_themed"].format(sec=sec, pct=pct)
    return t["tk_sector_diverse"].format(sec=sec, pct=pct)


def _corr_takeaway(corr, t):
    # extract upper triangle pairs
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iat[i, j]))
    if not pairs:
        return ""
    avg = float(np.mean([p[2] for p in pairs]))
    most = max(pairs, key=lambda p: p[2])
    least = min(pairs, key=lambda p: p[2])
    return t["tk_corr"].format(
        avg=avg,
        p_pair=f"{most[0]}-{most[1]}", p_v=most[2],
        n_pair=f"{least[0]}-{least[1]}", n_v=least[2],
    )


def _pca_takeaway(explained, t):
    if not explained:
        return ""
    p1 = explained[0]
    if p1 >= 70: return t["tk_pca_concentrated"].format(p1=p1)
    if p1 >= 50: return t["tk_pca_typical"].format(p1=p1)
    return t["tk_pca_diverse"].format(p1=p1)


# ---------- main entry ----------
def generate_pdf_report(
    *,
    index_series: pd.Series,
    benchmark: pd.Series | None,
    bench_label: str,
    base_value: float,
    weights: dict[str, float],
    share_info: pd.DataFrame,
    prices: pd.DataFrame,
    sector_weights: dict[str, float],
    metrics: dict,
    weight_mode_label: str,
    period: tuple[date, date],
    pca_explained: list[float] | None,
    lang: str = "en",
) -> bytes:
    _setup_fonts()
    t = TXT.get(lang, TXT["en"])
    cjk = lang == "zh"
    # When rendering Chinese, use the CJK TTF for EVERY piece of text (including
    # English/ASCII) so weights, italics, and headings stay visually consistent.
    if cjk:
        body_font = bold_font = italic_font = _CJK_FONT_NAME_RL
    else:
        body_font = "Helvetica"
        bold_font = "Helvetica-Bold"
        italic_font = "Helvetica-Oblique"

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title=t["report_title"],
    )

    title_style = ParagraphStyle("Title", fontName=bold_font, fontSize=22,
                                 textColor=HexColor(ACCENT), alignment=TA_LEFT,
                                 spaceAfter=6, leading=26)
    sub_style = ParagraphStyle("Sub", fontName=body_font, fontSize=10,
                               textColor=HexColor(MUTED), spaceAfter=20, leading=14)
    h2_style = ParagraphStyle("H2", fontName=bold_font, fontSize=14,
                              textColor=HexColor(INK), spaceBefore=18, spaceAfter=8,
                              leading=18, keepWithNext=True)
    body_style = ParagraphStyle("Body", fontName=body_font, fontSize=10,
                                textColor=HexColor(INK), leading=15, spaceAfter=6)
    caption_style = ParagraphStyle("Caption", fontName=italic_font, fontSize=9,
                                   textColor=HexColor(MUTED), spaceAfter=10, leading=13)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=16,
                                  bulletIndent=4, spaceAfter=5, leading=15)
    takeaway_style = ParagraphStyle(
        "Takeaway", fontName=body_font, fontSize=10, textColor=HexColor(INK),
        leading=15, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=14,
        backColor=HexColor(CALLOUT_BG), borderColor=HexColor(ACCENT),
        borderWidth=0, borderPadding=10,
    )

    def takeaway(text: str):
        if not text:
            return
        story.append(Paragraph(
            f"<b>{t['takeaway_label']}：</b>{text}" if cjk
            else f"<b>{t['takeaway_label']}:</b> {text}",
            takeaway_style,
        ))

    story = []

    # ===== Header =====
    story.append(Paragraph(t["report_title"], title_style))
    story.append(Paragraph(
        t["subtitle"].format(
            today=date.today().isoformat(),
            start=period[0], end=period[1],
            n=len(prices.columns), weight=weight_mode_label,
        ),
        sub_style,
    ))

    # ===== KPI table =====
    start_val = index_series.iloc[0]
    end_val = index_series.iloc[-1]
    total_return = (end_val / start_val - 1) * 100
    days = (index_series.index[-1] - index_series.index[0]).days or 1
    ann_return = ((end_val / start_val) ** (365 / days) - 1) * 100
    vol = index_series.pct_change().dropna().std() * (252 ** 0.5) * 100

    kpi_data = [
        [t["metric"], t["value"], t["metric"], t["value"]],
        [t["final_value"], f"{end_val:,.2f}", t["sharpe"], f"{metrics['sharpe']:.2f}"],
        [t["period_return"], f"{total_return:+.2f}%", t["sortino"], f"{metrics['sortino']:.2f}"],
        [t["ann_return"], f"{ann_return:+.2f}%", t["calmar"], f"{metrics['calmar']:.2f}"],
        [t["ann_vol"], f"{vol:.2f}%", t["beta"], f"{metrics['beta']:.2f}" if metrics["beta"] is not None else "—"],
        [t["max_dd"], f"{metrics['max_dd']:.2f}%", t["benchmark"], bench_label],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[4.2 * cm, 3.2 * cm, 4.2 * cm, 3.2 * cm])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(ACCENT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTNAME", (0, 1), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(CARD), HexColor(BG)]),
        ("TEXTCOLOR", (0, 1), (-1, -1), HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 12))

    # ===== Performance chart + takeaway =====
    story.append(Paragraph(t["h_perf"], h2_style))
    story.append(Image(_index_chart(index_series, benchmark, base_value, bench_label, t),
                       width=17 * cm, height=6.5 * cm))

    bench_line = ""
    if benchmark is not None and not benchmark.empty:
        b_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        delta = total_return - b_ret
        key = "bench_outperf" if delta > 0 else "bench_underperf"
        bench_line = t[key].format(b=bench_label, d=abs(delta))
    takeaway(t["tk_perf"].format(tot=total_return, ann=ann_return, bench_line=bench_line))

    # ===== Drawdown chart + takeaway =====
    story.append(Paragraph(t["h_drawdown"], h2_style))
    story.append(Image(_drawdown_chart(index_series, benchmark, bench_label, t),
                       width=17 * cm, height=5 * cm))

    bench_dd_msg = ""
    if benchmark is not None and not benchmark.empty:
        bdd = ((benchmark / benchmark.cummax()) - 1).min() * 100
        gap = abs(metrics["max_dd"] - bdd)
        if metrics["max_dd"] < bdd:
            bench_dd_msg = t["tk_dd_bench_worse"].format(b=bench_label, gap=gap)
        else:
            bench_dd_msg = t["tk_dd_bench_better"].format(b=bench_label, gap=gap)
    takeaway(t["tk_dd"].format(dd=metrics["max_dd"], bench_dd=bench_dd_msg))

    story.append(PageBreak())

    # ===== Risk-adjusted (bullets) + takeaway =====
    story.append(Paragraph(t["h_risk_adj"], h2_style))
    bullets = [_sharpe_text(metrics["sharpe"], t),
               t["sortino_explain"].format(s=metrics["sortino"]),
               t["calmar_explain"].format(s=metrics["calmar"])]
    beta_line = _beta_text(metrics["beta"], bench_label, t)
    if beta_line:
        bullets.append(beta_line)
    for b in bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    # ===== Constituents table + takeaway =====
    story.append(Paragraph(t["h_constituents"], h2_style))
    rows = [[t["col_ticker"], t["col_name"], t["col_sector"], t["col_weight"], t["col_cap"]]]

    def _fmt_big(n):
        if not n: return "—"
        for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
            if abs(n) >= div: return f"{n / div:,.2f}{unit}"
        return f"{n:,.0f}"

    for tk in prices.columns:
        rows.append([
            tk,
            str(share_info.loc[tk, "Name"])[:32],
            str(share_info.loc[tk, "Sector"] or "—"),
            f"{weights[tk] * 100:.2f}",
            _fmt_big(share_info.loc[tk, "MarketCap"]),
        ])
    const_tbl = Table(rows, colWidths=[2.0 * cm, 5.5 * cm, 4.0 * cm, 2.5 * cm, 2.8 * cm],
                      repeatRows=1)
    const_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(ACCENT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTNAME", (0, 1), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(CARD), HexColor(BG)]),
        ("TEXTCOLOR", (0, 1), (-1, -1), HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(const_tbl)
    max_t = max(weights, key=weights.get)
    min_t = min(weights, key=weights.get)
    takeaway(t["tk_constituents"].format(
        max_t=max_t, max_w=weights[max_t] * 100,
        min_t=min_t, min_w=weights[min_t] * 100,
    ))

    # ===== Sector breakdown + takeaway =====
    if sector_weights:
        story.append(Paragraph(t["h_sector"], h2_style))
        story.append(Image(_sector_chart(sector_weights, t), width=11 * cm, height=8 * cm))
        takeaway(_sector_text(sector_weights, t))

    story.append(PageBreak())

    # ===== Correlation + takeaway =====
    if prices.shape[1] >= 2:
        story.append(Paragraph(t["h_corr"], h2_style))
        story.append(Paragraph(t["caption_corr"], caption_style))
        corr = prices.pct_change().dropna().corr()
        story.append(Image(_corr_chart(corr, t), width=13 * cm, height=10 * cm))
        takeaway(_corr_takeaway(corr, t))

    # ===== PCA + takeaway =====
    if pca_explained:
        story.append(Paragraph(t["h_pca"], h2_style))
        story.append(Paragraph(t["caption_pca"], caption_style))
        takeaway(_pca_takeaway(pca_explained, t))

    # ===== Methodology + disclaimer =====
    story.append(Spacer(1, 16))
    story.append(Paragraph(t["h_methodology"], h2_style))
    story.append(Paragraph(t["methodology"], body_style))
    # CJK font has no italic variant — skip the <i> tag to avoid placeholder boxes
    disclaimer_html = t["disclaimer"] if cjk else f"<i>{t['disclaimer']}</i>"
    story.append(Paragraph(disclaimer_html, caption_style))

    doc.build(story)
    return buf.getvalue()
