# Technical Bottom Finder

A Streamlit-based technical analysis web app that estimates a **probable technical bottom support zone** for a stock ticker using a rules-based technical calculation engine.

The app is designed for research and education. It calculates support confluence, volatility-adjusted bottom zones, technical indicator scores, chart overlays, and historical backtest-style signal quality.

> **Important:** This tool is not investment advice. It does not predict the future or guarantee that a stock has bottomed.

---

## What the app does

Given a stock ticker, the app:

1. Downloads historical OHLCV data.
2. Calculates technical indicators.
3. Finds technical support and resistance levels.
4. Builds a probable bottom support range.
5. Scores the quality of the current bottom setup.
6. Backtests similar historical bottom-like signals.
7. Displays charts, indicator reads, and calculation details.

---

## Core features

- Probable technical bottom zone
- Invalidation level
- Final confidence score
- Support/resistance clustering
- Fixed-range volume profile
- ATR-adjusted support range
- RSI and MACD analysis
- Weekly RSI divergence
- Weekly MACD confirmation
- Yearly anchored VWAP
- 200-week SMA
- 2-year linear regression channel
- Coppock Curve
- Stan Weinstein Stage Analysis
- OBV trend break
- Historical volatility percentile
- ATR compression
- Backtest summary
- Detailed calculation transparency tab
- Optional production filters and index batch testing

---

## Project structure

Recommended GitHub repository structure:

```text
technical-bottom-finder/
│
├── app.py
├── requirements.txt
├── README.md
├── USER_GUIDE.md
└── .gitignore
```

If your Streamlit file is currently named `technical_bottom_finder_app.py`, you can either keep that name or rename it to `app.py`.

If you rename it:

```bash
ren technical_bottom_finder_app.py app.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/technical-bottom-finder.git
cd technical-bottom-finder
```

### 2. Create a virtual environment

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run locally

If your file is named `app.py`:

```bash
streamlit run app.py
```

If your file is named `technical_bottom_finder_app.py`:

```bash
streamlit run technical_bottom_finder_app.py
```

If the `streamlit` command is not recognized, use:

```bash
py -m streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Create a new app.
5. Select this repository.
6. Set the main file path to:

```text
app.py
```

or, if not renamed:

```text
technical_bottom_finder_app.py
```

7. Deploy.

You will receive a public URL that can be shared with colleagues.

---

## Example tickers

### US stocks

```text
AAPL
MSFT
NVDA
TSLA
ASML
```

### Indian NSE stocks

```text
RELIANCE.NS
HDFCBANK.NS
TCS.NS
INFY.NS
```

### Benchmarks

```text
^GSPC       S&P 500
^IXIC       Nasdaq Composite
^NSEI       Nifty 50
^VIX        CBOE Volatility Index
```

---

## Data source

The app uses `yfinance` for market data. This is suitable for research and prototypes, but for production use you should consider a more robust paid market data provider.

---

## Disclaimer

This tool is for educational and research purposes only. It is not financial, investment, trading, legal, tax, or professional advice.

The calculated support zones, scores, and indicators are based on historical market data and rule-based technical analysis. They may be wrong, delayed, incomplete, or unsuitable for a particular investment decision.

Always perform your own research and consult a qualified financial professional before making investment decisions.
