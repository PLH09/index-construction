---
title: Index Construction
emoji: 📈
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# Index Construction

🔗 **Live demos**
- Hugging Face Spaces (primary, recommended): https://huggingface.co/spaces/Paulineeeeeee/index-construction
- Streamlit Cloud (mirror): https://index-construction.streamlit.app

A minimalist Streamlit dashboard for building and analyzing your own custom equity index.

Pick any set of tickers, choose a weighting scheme, and instantly see index performance, component breakdown, and share-capital details. Bilingual (English / 中文).

## Features

- **Custom universe**: any Yahoo Finance ticker (US, TW, HK, etc. — e.g. `AAPL`, `2330.TW`), plus 8 curated preset baskets
- **Four weighting modes**: equal-weight, market-cap, free-float market-cap, and custom sliders
- **Flexible periods**: 1M / 3M / 6M / YTD / 1Y / 3Y / 5Y presets or custom dates
- **Benchmark comparison**: overlay S&P 500 / NASDAQ / Dow / Russell 2000 / MSCI World / TAIEX
- **Risk metrics**: Sharpe, Sortino, Calmar, Beta, annualized return / volatility, max drawdown
- **Auto risk-free rate** from FRED Treasury series (3M / 1Y / 2Y / 10Y), or manual
- **Drawdown chart** with benchmark overlay
- **Shares & market cap table**: total shares outstanding, free-float shares, float %, with portfolio totals
- **Sector breakdown** donut (with bundled fallback when Yahoo rate-limits)
- **Factor analysis (PCA)** and **rolling correlation heatmap**
- **Scenario comparison**: save A / B / C runs and overlay them
- **Exports**: CSV of prices/index, and a bilingual PDF report with narrative takeaways
- **Bilingual UI** (English / 中文) with one-click language toggle
- **Warm minimalist design** — terracotta accent on cream background

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Data sources

- **[Yahoo Finance](https://finance.yahoo.com/)** via [`yfinance`](https://github.com/ranaroussi/yfinance) — prices, market cap, shares outstanding, free float, sector
- **[FRED](https://fred.stlouisfed.org/)** (Federal Reserve Economic Data) — Treasury rates used as the risk-free rate for Sharpe / Sortino / Calmar

> FRED works key-free via its CSV endpoint. On networks where that host is blocked, set a free `FRED_API_KEY` (https://fredaccount.stlouisfed.org/apikeys) in `.streamlit/secrets.toml` or your Space secrets — the app falls back to it automatically, and to a manual rate if both are unavailable.

## Stack

- Streamlit · yfinance · FRED · pandas · NumPy · Plotly · scikit-learn · Matplotlib · ReportLab
