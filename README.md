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

🔗 **Live demo:** https://index-construction.streamlit.app

A minimalist Streamlit dashboard for building and analyzing your own custom equity index.

Pick any set of tickers, choose a weighting scheme, and instantly see index performance, component breakdown, and share-capital details. Bilingual (English / 中文).

## Features

- **Custom universe**: any Yahoo Finance ticker (US, TW, HK, etc. — e.g. `AAPL`, `2330.TW`)
- **Three weighting modes**: equal-weight, market-cap, free-float market-cap
- **Flexible periods**: 1M / 3M / 6M / YTD / 1Y / 3Y / 5Y presets or custom dates
- **KPIs**: index value, period return, annualized return, annualized volatility, max drawdown
- **Shares & market cap table**: total shares outstanding, free-float shares, float %, with portfolio totals
- **Charts**: index performance, normalized component comparison, weight donut
- **CSV export** of prices and index series
- **Bilingual UI** with one-click language toggle
- **Warm minimalist design** — terracotta accent on cream background

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Data source

[Yahoo Finance](https://finance.yahoo.com/) via the [`yfinance`](https://github.com/ranaroussi/yfinance) library.

## Stack

- Streamlit · yfinance · pandas · Plotly
