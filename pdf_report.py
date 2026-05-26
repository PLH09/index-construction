"""Generate a styled PDF report for an Index Construction run.

The PDF is intentionally English-only — keeps font handling portable on
Streamlit Cloud (no CJK font install needed) and keeps the export readable
for an international audience.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
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


# ---------- chart helpers (matplotlib, not plotly) ----------
def _style_axes(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, axis="y", color=LINE, linewidth=0.6)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _fig_to_buf(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf


def _index_chart(index_series: pd.Series, benchmark: pd.Series | None,
                 base: float, bench_label: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.fill_between(index_series.index, index_series.values, base, alpha=0.10, color=ACCENT)
    ax.plot(index_series.index, index_series.values, color=ACCENT, linewidth=2.0, label="Index")
    if benchmark is not None and not benchmark.empty:
        bench_norm = benchmark / benchmark.iloc[0] * base
        ax.plot(bench_norm.index, bench_norm.values, color=INK, linewidth=1.4,
                linestyle="--", label=f"vs {bench_label}")
    ax.axhline(base, color=MUTED, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_title("Index performance", loc="left", fontsize=11, weight="medium")
    _style_axes(ax)
    return _fig_to_buf(fig)


def _drawdown_chart(index_series: pd.Series) -> BytesIO:
    dd = (index_series / index_series.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.20, color=ACCENT_DARK)
    ax.plot(dd.index, dd.values, color=ACCENT_DARK, linewidth=1.2)
    ax.axhline(0, color=LINE, linewidth=0.8)
    ax.set_title("Drawdown (% from rolling peak)", loc="left", fontsize=11, weight="medium")
    _style_axes(ax)
    return _fig_to_buf(fig)


def _sector_chart(sector_weights: dict[str, float]) -> BytesIO:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    palette = ["#C75B3C", "#3D2B1F", "#D9A679", "#8A7968", "#6B4F3A", "#E8C39E", "#A0522D", "#B07B5A"]
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    ax.pie(values, labels=labels, colors=palette[:len(labels)],
           autopct="%1.0f%%", textprops={"color": INK, "fontsize": 8.5},
           wedgeprops=dict(width=0.45, edgecolor=BG))
    ax.set_title("Sector breakdown", loc="left", fontsize=11, weight="medium", color=INK)
    fig.patch.set_facecolor(BG)
    return _fig_to_buf(fig)


def _corr_chart(corr: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "terracotta_div", [INK, BG, ACCENT]
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, color=INK, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(corr.index, color=INK, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color=INK, fontsize=7)
    ax.set_title("Component correlations", loc="left", fontsize=11, weight="medium", color=INK)
    fig.patch.set_facecolor(BG)
    return _fig_to_buf(fig)


# ---------- text-explanation helpers ----------
def _interpret_sharpe(s: float) -> str:
    if s >= 2:
        return f"Sharpe {s:.2f} — Excellent risk-adjusted return."
    if s >= 1:
        return f"Sharpe {s:.2f} — Good. Returns comfortably justify the volatility taken."
    if s >= 0.5:
        return f"Sharpe {s:.2f} — Moderate. The index pays for risk, but not by much."
    if s >= 0:
        return f"Sharpe {s:.2f} — Weak. You're barely compensated for the volatility."
    return f"Sharpe {s:.2f} — Negative. The index underperformed the risk-free rate."


def _interpret_beta(b: float | None, bench: str) -> str:
    if b is None:
        return ""
    if b >= 1.2:
        return f"Beta {b:.2f} vs {bench} — High beta. Expect amplified moves in either direction."
    if b >= 0.8:
        return f"Beta {b:.2f} vs {bench} — Roughly market-like sensitivity."
    if b >= 0.3:
        return f"Beta {b:.2f} vs {bench} — Defensive. Less sensitive than the benchmark."
    return f"Beta {b:.2f} vs {bench} — Very low or negative correlation with the benchmark."


def _interpret_sector_concentration(sector_weights: dict[str, float]) -> str:
    if not sector_weights:
        return ""
    top_sec, top_w = max(sector_weights.items(), key=lambda kv: kv[1])
    pct = top_w * 100
    if pct >= 75:
        return f"Top sector ({top_sec}) is {pct:.0f}% of the index — extremely concentrated. Consider broadening."
    if pct >= 50:
        return f"Top sector ({top_sec}) accounts for {pct:.0f}% of the index — meaningfully concentrated."
    if pct >= 30:
        return f"Top sector ({top_sec}) accounts for {pct:.0f}% — typical for thematic baskets."
    return f"Top sector ({top_sec}) is {pct:.0f}% — well diversified across sectors."


def _interpret_pca(explained: list[float]) -> list[str]:
    lines = []
    if not len(explained):
        return lines
    pc1 = explained[0]
    if pc1 >= 70:
        lines.append(f"PC1 explains {pc1:.0f}% — a single 'broad market' factor dominates. The components move in lockstep.")
    elif pc1 >= 50:
        lines.append(f"PC1 explains {pc1:.0f}% — broad market beta is the primary driver, typical for sector-themed baskets.")
    else:
        lines.append(f"PC1 explains {pc1:.0f}% — surprisingly diversified return drivers across components.")
    if len(explained) >= 2:
        lines.append(f"PC2 explains {explained[1]:.0f}% — secondary tilt, often a style or sub-sector dimension.")
    return lines


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
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="Index Construction Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                 fontName="Helvetica-Bold", fontSize=20, textColor=HexColor(ACCENT),
                                 alignment=TA_LEFT, spaceAfter=4, leading=24)
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontName="Helvetica",
                               fontSize=10, textColor=HexColor(MUTED), spaceAfter=18)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=13, textColor=HexColor(INK), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica",
                                fontSize=10, textColor=HexColor(INK), leading=14, spaceAfter=6)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontName="Helvetica-Oblique",
                                   fontSize=9, textColor=HexColor(MUTED), spaceAfter=10)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=14, bulletIndent=4,
                                  spaceAfter=4)

    story = []

    # ===== Header =====
    story.append(Paragraph("Index Construction Report", title_style))
    story.append(Paragraph(
        f"Generated {date.today().isoformat()} · Period {period[0]} to {period[1]} · "
        f"{len(prices.columns)} constituents · {weight_mode_label} weighting",
        sub_style,
    ))

    # ===== Key metrics table =====
    start_val = index_series.iloc[0]
    end_val = index_series.iloc[-1]
    total_return = (end_val / start_val - 1) * 100
    days = (index_series.index[-1] - index_series.index[0]).days or 1
    ann_return = ((end_val / start_val) ** (365 / days) - 1) * 100
    vol = index_series.pct_change().dropna().std() * (252 ** 0.5) * 100

    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Final value", f"{end_val:,.2f}", "Sharpe", f"{metrics['sharpe']:.2f}"],
        ["Period return", f"{total_return:+.2f}%", "Sortino", f"{metrics['sortino']:.2f}"],
        ["Annualized return", f"{ann_return:+.2f}%", "Calmar", f"{metrics['calmar']:.2f}"],
        ["Annualized vol", f"{vol:.2f}%", "Beta", f"{metrics['beta']:.2f}" if metrics["beta"] is not None else "—"],
        ["Max drawdown", f"{metrics['max_dd']:.2f}%", "Benchmark", bench_label],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[4.2 * cm, 3.2 * cm, 4.2 * cm, 3.2 * cm])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(ACCENT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(CARD), HexColor(BG)]),
        ("TEXTCOLOR", (0, 1), (-1, -1), HexColor(INK)),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 12))

    # ===== Performance chart =====
    story.append(Paragraph("Performance", h2_style))
    story.append(Image(_index_chart(index_series, benchmark, base_value, bench_label),
                       width=17 * cm, height=6.5 * cm))
    perf_text = (
        f"The index moved from <b>{start_val:.2f}</b> to <b>{end_val:.2f}</b> over the period "
        f"({total_return:+.2f}% total, {ann_return:+.2f}% annualized). "
    )
    if benchmark is not None and not benchmark.empty:
        b_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        delta = total_return - b_ret
        verb = "outperformed" if delta > 0 else "underperformed"
        perf_text += (
            f"Over the same window, {bench_label} returned {b_ret:+.2f}%, so the index "
            f"{verb} the benchmark by <b>{abs(delta):.2f} percentage points</b>."
        )
    story.append(Paragraph(perf_text, body_style))

    # ===== Drawdown chart =====
    story.append(Image(_drawdown_chart(index_series), width=17 * cm, height=5 * cm))
    story.append(Paragraph(
        f"Maximum drawdown — the worst peak-to-trough fall — was <b>{metrics['max_dd']:.2f}%</b>. "
        "Drawdown depth and duration matter as much as average return; long flat periods "
        "after a deep drawdown are what shake investors out of a strategy.",
        body_style,
    ))

    # ===== Risk-adjusted interpretation =====
    story.append(Paragraph("Risk-adjusted performance", h2_style))
    bullets = [_interpret_sharpe(metrics["sharpe"])]
    bullets.append(
        f"Sortino {metrics['sortino']:.2f} — like Sharpe but only penalizes downside volatility. "
        "If much higher than Sharpe, the index's volatility is mostly to the upside."
    )
    bullets.append(
        f"Calmar {metrics['calmar']:.2f} — annualized return divided by max drawdown. "
        "Above 1.0 means you earned more (annually) than the worst dip cost you."
    )
    beta_line = _interpret_beta(metrics["beta"], bench_label)
    if beta_line:
        bullets.append(beta_line)
    for b in bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    story.append(PageBreak())

    # ===== Constituents table =====
    story.append(Paragraph("Constituents", h2_style))
    rows = [["Ticker", "Name", "Sector", "Weight %", "Market Cap"]]

    def _fmt_big(n):
        if not n: return "—"
        for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
            if abs(n) >= div: return f"{n / div:,.2f}{unit}"
        return f"{n:,.0f}"

    for t in prices.columns:
        rows.append([
            t,
            str(share_info.loc[t, "Name"])[:32],
            str(share_info.loc[t, "Sector"] or "—"),
            f"{weights[t] * 100:.2f}",
            _fmt_big(share_info.loc[t, "MarketCap"]),
        ])
    const_tbl = Table(rows, colWidths=[2.0 * cm, 5.5 * cm, 4.0 * cm, 2.5 * cm, 2.8 * cm],
                      repeatRows=1)
    const_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(ACCENT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(CARD), HexColor(BG)]),
        ("TEXTCOLOR", (0, 1), (-1, -1), HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor(LINE)),
        ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(const_tbl)
    story.append(Spacer(1, 14))

    # ===== Sector breakdown =====
    if sector_weights:
        story.append(Paragraph("Sector breakdown", h2_style))
        story.append(Image(_sector_chart(sector_weights), width=11 * cm, height=8 * cm))
        story.append(Paragraph(_interpret_sector_concentration(sector_weights), body_style))

    story.append(PageBreak())

    # ===== Correlation =====
    if prices.shape[1] >= 2:
        story.append(Paragraph("Component correlations", h2_style))
        story.append(Paragraph(
            "Pairwise correlation of daily returns. Dark cells = move together (low diversification); "
            "light or negative cells = genuine diversification.",
            caption_style,
        ))
        corr = prices.pct_change().dropna().corr()
        story.append(Image(_corr_chart(corr), width=13 * cm, height=10 * cm))

    # ===== PCA interpretation =====
    if pca_explained:
        story.append(Paragraph("Factor analysis (PCA)", h2_style))
        story.append(Paragraph(
            "Principal Component Analysis decomposes return variance into orthogonal factors.",
            caption_style,
        ))
        for line in _interpret_pca(list(pca_explained)):
            story.append(Paragraph(f"• {line}", bullet_style))

    # ===== Methodology / disclaimer =====
    story.append(Spacer(1, 20))
    story.append(Paragraph("Methodology & disclaimer", h2_style))
    story.append(Paragraph(
        "Prices are total-return adjusted closes from Yahoo Finance via the yfinance library. "
        "Returns and volatility are computed from daily log changes; annualization uses 252 trading days; "
        "max drawdown is the largest peak-to-trough decline of the cumulative index. "
        "Beta is calculated as cov(index, benchmark) / var(benchmark) over daily returns. "
        "PCA is fitted on the daily return matrix using scikit-learn.",
        body_style,
    ))
    story.append(Paragraph(
        "<i>This report is for educational and research purposes only. It does not constitute "
        "investment advice. Past performance is not indicative of future results.</i>",
        caption_style,
    ))

    doc.build(story)
    return buf.getvalue()
