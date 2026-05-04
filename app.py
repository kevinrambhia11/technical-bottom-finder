"""
AI Technical Bottom Finder - Serious Version

Features included:
1. Support/resistance clustering
2. RSI bullish divergence detection
3. Volume-profile support zone
4. Candlestick reversal patterns
5. Multiple-timeframe confirmation
6. Backtesting module
7. Confidence score based on historical hit rate
8. Optional AI-style explanation layer using deterministic text generation
9. Survivorship-bias-free constituent CSV support for index-level testing
10. Slippage and liquidity filters
11. Sector-relative strength
12. Market-regime filter using index trend and volatility
13. SQLite signal database for daily monitoring

Install:
    pip install streamlit yfinance pandas numpy plotly scikit-learn

Run:
    streamlit run technical_bottom_finder_app.py

Ticker examples:
    US: AAPL, MSFT, NVDA, TSLA
    India NSE: RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS

Disclaimer:
    This app is for educational and research use only. It is not investment advice.
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# UI styling
# -----------------------------------------------------------------------------

APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #070b12;
        --panel: rgba(15, 23, 42, 0.78);
        --panel-soft: rgba(15, 23, 42, 0.52);
        --border: rgba(148, 163, 184, 0.18);
        --border-strong: rgba(148, 163, 184, 0.30);
        --text: #f8fafc;
        --muted: #94a3b8;
        --muted-2: #64748b;
        --green: #22c55e;
        --amber: #f59e0b;
        --red: #ef4444;
        --blue: #60a5fa;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(37, 99, 235, 0.14), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(16, 185, 129, 0.08), transparent 24%),
            #070b12;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.64);
        border: 1px solid var(--border);
        padding: 14px 16px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    .hero-card {
        background:
            linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(2, 6, 23, 0.92)),
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.20), transparent 34%);
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 28px;
        padding: 28px 32px;
        margin: 10px 0 20px 0;
        box-shadow: 0 24px 70px rgba(0,0,0,0.24);
    }

    .classification-card {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 18px 20px;
        margin: 14px 0 18px 0;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.54);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 20px 22px;
        margin: 16px 0;
        box-shadow: 0 10px 28px rgba(0,0,0,0.10);
    }

    .signal-card {
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 15px 16px;
        min-height: 132px;
        margin-bottom: 12px;
    }

    .signal-pass {
        border-left: 4px solid var(--green);
    }

    .signal-fail {
        border-left: 4px solid var(--muted-2);
    }

    .signal-mixed {
        border-left: 4px solid var(--amber);
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 500;
    }

    .micro-muted {
        color: var(--muted-2);
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .big-level {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    .pill {
        display:inline-flex;
        align-items:center;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        background: rgba(15, 23, 42, 0.72);
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .pill-green { color: #86efac; border-color: rgba(34, 197, 94, 0.35); }
    .pill-amber { color: #fcd34d; border-color: rgba(245, 158, 11, 0.35); }
    .pill-slate { color: #cbd5e1; border-color: rgba(148, 163, 184, 0.25); }

    .explain-box {
        background: rgba(2, 6, 23, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 15px 16px;
        color: #cbd5e1;
        line-height: 1.55;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px 999px 0 0;
        padding: 10px 16px;
        font-weight: 650;
    }

    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.72);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.14);
    }
</style>
"""


# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Technical Bottom Finder",
    page_icon="📉",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Zone:
    lower: float
    upper: float
    center: float
    strength: float
    source: str


@dataclass
class SignalResult:
    name: str
    points: float
    max_points: float
    passed: bool
    explanation: str


@dataclass
class BacktestResult:
    hit_rate: float
    avg_forward_return: float
    median_forward_return: float
    sample_size: int
    avg_max_drawdown: float
    confidence_adjustment: float


# -----------------------------------------------------------------------------
# Database storage for daily monitoring
# -----------------------------------------------------------------------------

DB_PATH = Path("technical_bottom_signals.db")


def init_signal_db(db_path: Path = DB_PATH) -> None:
    """Create the local SQLite signal table if it does not exist."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                ticker TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                latest_close REAL,
                base_score REAL,
                final_score REAL,
                verdict TEXT,
                bottom_zone_lower REAL,
                bottom_zone_upper REAL,
                invalidation REAL,
                backtest_hit_rate REAL,
                backtest_sample_size INTEGER,
                payload_json TEXT,
                PRIMARY KEY (ticker, signal_date)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_signal_record(
    ticker: str,
    signal_date: pd.Timestamp,
    latest_close: float,
    base_score: float,
    final_score: float,
    context: Dict[str, object],
    signals: List[SignalResult],
    backtest: BacktestResult,
    db_path: Path = DB_PATH,
) -> None:
    """Insert or update the latest signal for one ticker/date."""
    init_signal_db(db_path)

    payload = {
        "signals": [
            {
                "name": s.name,
                "points": s.points,
                "max_points": s.max_points,
                "passed": s.passed,
                "explanation": s.explanation,
            }
            for s in signals
        ],
        "backtest": {
            "hit_rate": backtest.hit_rate,
            "avg_forward_return": backtest.avg_forward_return,
            "median_forward_return": backtest.median_forward_return,
            "sample_size": backtest.sample_size,
            "avg_max_drawdown": backtest.avg_max_drawdown,
            "confidence_adjustment": backtest.confidence_adjustment,
        },
    }

    row = (
        ticker.upper(),
        pd.to_datetime(signal_date).strftime("%Y-%m-%d"),
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        float(latest_close),
        float(base_score),
        float(final_score),
        verdict_from_score(final_score),
        float(context["bottom_zone_lower"]),
        float(context["bottom_zone_upper"]),
        float(context["invalidation"]),
        float(backtest.hit_rate),
        int(backtest.sample_size),
        json.dumps(payload),
    )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO signals (
                ticker, signal_date, created_at, latest_close, base_score, final_score,
                verdict, bottom_zone_lower, bottom_zone_upper, invalidation,
                backtest_hit_rate, backtest_sample_size, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def load_signal_history(db_path: Path = DB_PATH, limit: int = 500) -> pd.DataFrame:
    """Load recent stored signals from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT
                ticker,
                signal_date,
                created_at,
                latest_close,
                base_score,
                final_score,
                verdict,
                bottom_zone_lower,
                bottom_zone_upper,
                invalidation,
                backtest_hit_rate,
                backtest_sample_size
            FROM signals
            ORDER BY signal_date DESC, final_score DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance using yfinance."""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV into weekly/monthly candles."""
    if df.empty:
        return df

    out = pd.DataFrame()
    out["Open"] = df["Open"].resample(rule).first()
    out["High"] = df["High"].resample(rule).max()
    out["Low"] = df["Low"].resample(rule).min()
    out["Close"] = df["Close"].resample(rule).last()
    out["Volume"] = df["Volume"].resample(rule).sum()
    return out.dropna()


# -----------------------------------------------------------------------------
# Indicator calculations
# -----------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(20)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the latest value within each rolling window."""
    def pct_rank(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        return float(np.sum(values <= values[-1]) / len(values) * 100)

    return series.rolling(window).apply(pct_rank, raw=True)


def add_yearly_anchored_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add yearly anchored VWAP, reset at the first trading day of each calendar year."""
    out = df.copy()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    pv = typical_price * out["Volume"]
    year_key = out.index.year
    out["Yearly_AVWAP"] = pv.groupby(year_key).cumsum() / out["Volume"].groupby(year_key).cumsum()
    return out


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume and a short moving average for OBV trend checks."""
    out = df.copy()
    direction = np.sign(out["Close"].diff()).fillna(0)
    out["OBV"] = (direction * out["Volume"]).cumsum()
    out["OBV_EMA20"] = out["OBV"].ewm(span=20, adjust=False).mean()
    out["OBV_Rolling_High20"] = out["OBV"].rolling(20).max()
    return out


def add_historical_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add 20-day annualized historical volatility and its 252-day percentile."""
    out = df.copy()
    log_returns = np.log(out["Close"] / out["Close"].shift(1))
    out["HV20"] = log_returns.rolling(20).std() * np.sqrt(252)
    out["HV20_Pctile_252"] = rolling_percentile_rank(out["HV20"], 252)
    return out


def add_atr_compression_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR percentage and percentile metrics used to detect volatility compression."""
    out = df.copy()
    out["ATR_Pct"] = out["ATR14"] / out["Close"]
    out["ATR_Pct_SMA20"] = out["ATR_Pct"].rolling(20).mean()
    out["ATR_Pctile_252"] = rolling_percentile_rank(out["ATR_Pct"], 252)
    return out


def add_linear_regression_channel(
    df: pd.DataFrame,
    window: int = 504,
    deviations: float = 2.0,
) -> pd.DataFrame:
    """Add a rolling 2-year linear regression channel.

    The channel is calculated on closing prices over approximately 504 trading
    days, with upper/lower bands set at +/- 2 residual standard deviations.
    """
    out = df.copy()
    close = out["Close"].astype(float).values

    mid = np.full(len(out), np.nan)
    upper = np.full(len(out), np.nan)
    lower = np.full(len(out), np.nan)
    slope = np.full(len(out), np.nan)

    if len(out) < window:
        out["LRC_Mid_2Y"] = mid
        out["LRC_Upper_2Y"] = upper
        out["LRC_Lower_2Y"] = lower
        out["LRC_Slope_2Y"] = slope
        out["LRC_Position_2Y"] = np.nan
        return out

    x_axis = np.arange(window)
    for i in range(window - 1, len(out)):
        y = close[i - window + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        beta, alpha = np.polyfit(x_axis, y, 1)
        fitted = alpha + beta * x_axis
        residual_std = np.std(y - fitted)
        mid[i] = fitted[-1]
        upper[i] = fitted[-1] + deviations * residual_std
        lower[i] = fitted[-1] - deviations * residual_std
        slope[i] = beta

    out["LRC_Mid_2Y"] = mid
    out["LRC_Upper_2Y"] = upper
    out["LRC_Lower_2Y"] = lower
    out["LRC_Slope_2Y"] = slope
    out["LRC_Position_2Y"] = (out["Close"] - out["LRC_Lower_2Y"]) / (out["LRC_Upper_2Y"] - out["LRC_Lower_2Y"])
    return out


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average with linearly increasing weights."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators used by the model."""
    df = df.copy()

    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA100"] = sma(df["Close"], 100)
    df["SMA200"] = sma(df["Close"], 200)

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)

    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df["Close"])
    df["ATR14"] = calculate_atr(df, 14)
    df["ADX14"] = calculate_adx(df, 14)

    df["BB_Mid"] = sma(df["Close"], 20)
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

    df["Volume_SMA20"] = sma(df["Volume"], 20)
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    df["Ret_1D"] = df["Close"].pct_change()
    df["Rolling_52W_High"] = df["High"].rolling(252).max()
    df["Rolling_52W_Low"] = df["Low"].rolling(252).min()
    df["Drawdown_52W"] = df["Close"] / df["Rolling_52W_High"] - 1

    df = add_yearly_anchored_vwap(df)
    df = add_obv(df)
    df = add_historical_volatility(df)
    df = add_atr_compression_metrics(df)
    df = add_linear_regression_channel(df, window=504, deviations=2.0)

    return df


# -----------------------------------------------------------------------------
# Swing points and support/resistance clustering
# -----------------------------------------------------------------------------

def find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Find local swing lows."""
    lows = []
    low_series = df["Low"]

    for i in range(window, len(df) - window):
        left = low_series.iloc[i - window:i]
        right = low_series.iloc[i + 1:i + window + 1]
        current = low_series.iloc[i]

        if current <= left.min() and current <= right.min():
            lows.append(
                {
                    "Date": df.index[i],
                    "Price": float(current),
                    "Volume": float(df["Volume"].iloc[i]),
                    "RSI14": float(df["RSI14"].iloc[i]) if "RSI14" in df.columns else np.nan,
                }
            )

    return pd.DataFrame(lows)


def find_swing_highs(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Find local swing highs."""
    highs = []
    high_series = df["High"]

    for i in range(window, len(df) - window):
        left = high_series.iloc[i - window:i]
        right = high_series.iloc[i + 1:i + window + 1]
        current = high_series.iloc[i]

        if current >= left.max() and current >= right.max():
            highs.append(
                {
                    "Date": df.index[i],
                    "Price": float(current),
                    "Volume": float(df["Volume"].iloc[i]),
                    "RSI14": float(df["RSI14"].iloc[i]) if "RSI14" in df.columns else np.nan,
                }
            )

    return pd.DataFrame(highs)


def cluster_price_levels(
    prices: pd.Series,
    current_price: float,
    atr: float,
    source: str,
    min_samples: int = 2,
) -> List[Zone]:
    """Cluster nearby price levels using DBSCAN.

    eps is based on ATR, which makes clustering volatility-aware.
    """
    clean = pd.Series(prices).dropna()
    clean = clean[clean > 0]

    if len(clean) < min_samples:
        return []

    eps = max(float(atr) * 0.75, current_price * 0.01)
    X = clean.values.reshape(-1, 1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    zones: List[Zone] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_prices = clean.values[labels == label]
        lower = float(np.min(cluster_prices))
        upper = float(np.max(cluster_prices))
        center = float(np.mean(cluster_prices))
        touches = len(cluster_prices)
        distance_penalty = abs(current_price - center) / max(current_price, 1)
        strength = touches * max(0.25, 1 - distance_penalty * 5)

        zones.append(
            Zone(
                lower=lower,
                upper=upper,
                center=center,
                strength=float(strength),
                source=source,
            )
        )

    zones = sorted(zones, key=lambda z: z.strength, reverse=True)
    return zones


def get_support_resistance_zones(df: pd.DataFrame, lookback: int = 252) -> Tuple[List[Zone], List[Zone]]:
    """Detect support and resistance zones from clustered swing lows/highs."""
    recent = df.tail(lookback).copy()
    current_price = float(recent["Close"].iloc[-1])
    atr = float(recent["ATR14"].iloc[-1]) if not pd.isna(recent["ATR14"].iloc[-1]) else current_price * 0.02

    swing_lows = find_swing_lows(recent, window=5)
    swing_highs = find_swing_highs(recent, window=5)

    supports = []
    resistances = []

    if not swing_lows.empty:
        support_candidates = swing_lows[swing_lows["Price"] <= current_price * 1.05]["Price"]
        supports = cluster_price_levels(support_candidates, current_price, atr, source="Swing-low cluster")

    if not swing_highs.empty:
        resistance_candidates = swing_highs[swing_highs["Price"] >= current_price * 0.95]["Price"]
        resistances = cluster_price_levels(resistance_candidates, current_price, atr, source="Swing-high cluster")

    return supports, resistances


# -----------------------------------------------------------------------------
# Volume-profile support
# -----------------------------------------------------------------------------

def volume_profile_zone(df: pd.DataFrame, lookback: int = 252, bins: int = 40) -> Optional[Zone]:
    """Approximate fixed-range volume profile support.

    This improved version distributes each candle's volume across the price bins
    touched by its high-low range instead of assigning all volume to the close.
    That makes the high-volume node less noisy and more representative of where
    trading actually occurred during the range.
    """
    recent = df.tail(lookback).copy()
    if recent.empty or len(recent) < 50:
        return None

    current_price = float(recent["Close"].iloc[-1])
    min_price = float(recent["Low"].min())
    max_price = float(recent["High"].max())

    if min_price <= 0 or max_price <= min_price:
        return None

    edges = np.linspace(min_price, max_price, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    volume_by_bin = np.zeros(bins)

    for _, row in recent.iterrows():
        low = float(row["Low"])
        high = float(row["High"])
        volume = float(row["Volume"])
        if high <= low or volume <= 0:
            continue

        touched = np.where((edges[:-1] <= high) & (edges[1:] >= low))[0]
        if len(touched) == 0:
            continue
        volume_by_bin[touched] += volume / len(touched)

    vp = pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "center": centers,
            "volume": volume_by_bin,
        }
    )

    if vp.empty or vp["volume"].sum() <= 0:
        return None

    # Prefer high-volume nodes at or below/slightly above current price.
    support_nodes = vp[vp["center"] <= current_price * 1.03].copy()
    if support_nodes.empty:
        return None

    support_nodes["distance"] = (current_price - support_nodes["center"]).abs() / current_price
    support_nodes["volume_rank"] = support_nodes["volume"].rank(pct=True)
    support_nodes["score"] = support_nodes["volume_rank"] - support_nodes["distance"] * 2

    best = support_nodes.sort_values("score", ascending=False).iloc[0]
    return Zone(
        lower=float(best["lower"]),
        upper=float(best["upper"]),
        center=float(best["center"]),
        strength=float(best["score"] * 10),
        source="Fixed-range volume-profile high-volume node",
    )


# -----------------------------------------------------------------------------
# Robust support-zone engine
# -----------------------------------------------------------------------------

def get_200_week_sma_level(daily_df: pd.DataFrame) -> Optional[float]:
    """Return latest 200-week SMA level if enough weekly history exists."""
    weekly = resample_ohlcv(daily_df, "W-FRI")
    if len(weekly) < 205:
        return None
    sma200w = weekly["Close"].rolling(200).mean().dropna()
    if sma200w.empty:
        return None
    return float(sma200w.iloc[-1])


def fibonacci_support_levels(df: pd.DataFrame, lookback: int = 504) -> List[Dict[str, object]]:
    """Return Fibonacci retracement support candidates from the recent major range.

    Uses the high/low over the selected lookback window. The levels are useful
    when price is retracing a large advance and approaching common institutional
    retracement zones such as 38.2%, 50%, 61.8%, and 78.6%.
    """
    recent = df.tail(lookback).copy()
    if recent.empty or len(recent) < 60:
        return []

    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    current = float(recent["Close"].iloc[-1])
    if high <= low or low <= 0:
        return []

    diff = high - low
    retracements = {
        "Fib 38.2%": high - 0.382 * diff,
        "Fib 50.0%": high - 0.500 * diff,
        "Fib 61.8%": high - 0.618 * diff,
        "Fib 78.6%": high - 0.786 * diff,
    }

    rows = []
    for name, level in retracements.items():
        # Keep levels that are relevant to current price, not levels far away.
        if current * 0.70 <= level <= current * 1.15:
            rows.append(
                {
                    "name": name,
                    "level": float(level),
                    "weight": 2.0,
                    "category": "Fibonacci",
                    "reason": f"{name} retracement from the recent {lookback}-day major range.",
                }
            )
    return rows


def build_support_components(
    df: pd.DataFrame,
    support_zone: Optional[Zone],
    vp_zone: Optional[Zone],
    current_price: float,
    atr: float,
) -> List[Dict[str, object]]:
    """Build support candidates from several independent technical methods."""
    latest = df.iloc[-1]
    components: List[Dict[str, object]] = []

    def add_component(name: str, level: object, weight: float, category: str, reason: str) -> None:
        if level is None:
            return
        try:
            level_float = float(level)
        except Exception:
            return
        if math.isnan(level_float) or level_float <= 0:
            return
        # Retain nearby support references. Allow slightly above current price
        # because some reclaim/reference levels can sit just above price.
        if not (current_price * 0.60 <= level_float <= current_price * 1.12):
            return
        distance_atr = abs(current_price - level_float) / max(atr, 1e-9)
        proximity_score = max(0.20, 1 - min(distance_atr / 5, 0.80))
        components.append(
            {
                "name": name,
                "level": level_float,
                "base_weight": float(weight),
                "effective_weight": float(weight * proximity_score),
                "distance_atr": float(distance_atr),
                "category": category,
                "reason": reason,
            }
        )

    if support_zone:
        add_component(
            "Clustered swing-low support",
            support_zone.center,
            min(max(support_zone.strength, 2.0), 8.0),
            "Price action",
            "Cluster of prior local lows detected with volatility-aware DBSCAN clustering.",
        )

    if vp_zone:
        add_component(
            "Fixed-range volume-profile support",
            vp_zone.center,
            4.5,
            "Volume profile",
            "High-volume node from the fixed-range volume profile.",
        )

    # Rolling lows capture obvious horizontal supports even if swing clustering misses them.
    for window, weight in [(20, 1.5), (60, 2.5), (126, 2.5), (252, 3.0)]:
        if len(df) >= window:
            level = float(df["Low"].tail(window).min())
            add_component(
                f"{window}-day rolling low",
                level,
                weight,
                "Rolling support",
                f"Lowest traded level over the last {window} trading days.",
            )

    # Moving averages and reference levels can become dynamic support.
    reference_levels = [
        ("20-DMA", latest.get("SMA20", np.nan), 1.5, "Moving average", "Short-term trend reference."),
        ("50-DMA", latest.get("SMA50", np.nan), 2.0, "Moving average", "Intermediate trend reference."),
        ("200-DMA", latest.get("SMA200", np.nan), 3.0, "Moving average", "Long-term institutional trend reference."),
        ("Yearly anchored VWAP", latest.get("Yearly_AVWAP", np.nan), 3.0, "VWAP", "Current-year volume-weighted cost-basis reference."),
        ("2Y regression lower band", latest.get("LRC_Lower_2Y", np.nan), 2.5, "Regression channel", "Lower band of the 2-year linear regression channel."),
        ("2Y regression midline", latest.get("LRC_Mid_2Y", np.nan), 1.5, "Regression channel", "Midline of the 2-year linear regression channel."),
        ("Lower Bollinger Band", latest.get("BB_Lower", np.nan), 1.5, "Volatility band", "Lower 20-day Bollinger Band."),
    ]

    for name, level, weight, category, reason in reference_levels:
        add_component(name, level, weight, category, reason)

    sma200w = get_200_week_sma_level(df)
    add_component(
        "200-week SMA",
        sma200w,
        4.0,
        "Long-cycle moving average",
        "Long-cycle support level widely watched by trend followers.",
    )

    for fib in fibonacci_support_levels(df, lookback=min(504, len(df))):
        add_component(
            str(fib["name"]),
            fib["level"],
            float(fib["weight"]),
            str(fib["category"]),
            str(fib["reason"]),
        )

    if not components:
        fallback = float(df["Low"].tail(60).min())
        add_component(
            "Fallback 60-day low",
            fallback,
            1.0,
            "Fallback",
            "Used because no stronger support reference was detected.",
        )

    return components


def robust_support_confluence(
    df: pd.DataFrame,
    support_zone: Optional[Zone],
    vp_zone: Optional[Zone],
    atr: float,
) -> Dict[str, object]:
    """Choose the best support zone using weighted confluence.

    Instead of simply averaging two levels, this function pools several support
    references, clusters nearby levels, and selects the cluster with the best
    combination of weight, proximity, and number of independent confirmations.
    """
    latest = df.iloc[-1]
    current_price = float(latest["Close"])
    atr = atr if atr > 0 and not math.isnan(atr) else current_price * 0.02

    components = build_support_components(df, support_zone, vp_zone, current_price, atr)
    levels = np.array([c["level"] for c in components]).reshape(-1, 1)
    eps = max(atr * 0.75, current_price * 0.012)
    labels = DBSCAN(eps=eps, min_samples=1).fit(levels).labels_

    clusters = []
    for label in sorted(set(labels)):
        members = [components[i] for i in range(len(components)) if labels[i] == label]
        total_weight = sum(float(m["effective_weight"]) for m in members)
        if total_weight <= 0:
            continue
        weighted_center = sum(float(m["level"]) * float(m["effective_weight"]) for m in members) / total_weight
        lower = min(float(m["level"]) for m in members)
        upper = max(float(m["level"]) for m in members)
        source_count = len(set(str(m["category"]) for m in members))
        distance_atr = abs(current_price - weighted_center) / max(atr, 1e-9)
        proximity_score = max(0.10, 1 - min(distance_atr / 6, 0.90))
        independence_bonus = 1 + min(source_count - 1, 4) * 0.12
        cluster_score = total_weight * proximity_score * independence_bonus
        clusters.append(
            {
                "label": int(label),
                "center": float(weighted_center),
                "lower": float(lower),
                "upper": float(upper),
                "total_weight": float(total_weight),
                "source_count": int(source_count),
                "distance_atr": float(distance_atr),
                "cluster_score": float(cluster_score),
                "members": members,
            }
        )

    if not clusters:
        fallback = float(df["Low"].tail(60).min())
        selected = {
            "label": -1,
            "center": fallback,
            "lower": fallback,
            "upper": fallback,
            "total_weight": 1.0,
            "source_count": 1,
            "distance_atr": abs(current_price - fallback) / max(atr, 1e-9),
            "cluster_score": 1.0,
            "members": components,
        }
    else:
        # Prefer support near/below price, but allow levels just above price if they
        # are part of a reclaim/confluence area.
        eligible = [c for c in clusters if c["center"] <= current_price * 1.08]
        if not eligible:
            eligible = clusters
        selected = sorted(eligible, key=lambda c: c["cluster_score"], reverse=True)[0]

    atr_pctile = latest.get("ATR_Pctile_252", np.nan)
    hv_pctile = latest.get("HV20_Pctile_252", np.nan)
    vol_pressure = np.nanmax([atr_pctile, hv_pctile]) if not (pd.isna(atr_pctile) and pd.isna(hv_pctile)) else 50

    # Dynamic width: high volatility gets a wider support zone; low volatility gets tighter.
    if vol_pressure >= 85:
        zone_atr_multiplier = 1.30
        invalidation_atr_multiplier = 2.00
    elif vol_pressure <= 35:
        zone_atr_multiplier = 0.80
        invalidation_atr_multiplier = 1.30
    else:
        zone_atr_multiplier = 1.00
        invalidation_atr_multiplier = 1.50

    # Strong confluence allows a slightly tighter zone because several methods
    # agree on a similar support area.
    if selected["cluster_score"] >= 8 and selected["source_count"] >= 3:
        zone_atr_multiplier *= 0.90

    confluence_points = 0.0
    if selected["cluster_score"] >= 10 and selected["source_count"] >= 3:
        confluence_points = 15.0
    elif selected["cluster_score"] >= 6:
        confluence_points = 10.0
    elif selected["cluster_score"] >= 3:
        confluence_points = 5.0

    blended_support = float(selected["center"])
    bottom_zone_lower = blended_support - zone_atr_multiplier * atr
    bottom_zone_upper = blended_support + zone_atr_multiplier * atr
    invalidation = blended_support - invalidation_atr_multiplier * atr

    return {
        "components": components,
        "clusters": clusters,
        "selected_cluster": selected,
        "blended_support": blended_support,
        "bottom_zone_lower": float(bottom_zone_lower),
        "bottom_zone_upper": float(bottom_zone_upper),
        "invalidation": float(invalidation),
        "zone_atr_multiplier": float(zone_atr_multiplier),
        "invalidation_atr_multiplier": float(invalidation_atr_multiplier),
        "confluence_points": float(confluence_points),
        "confluence_score": float(selected["cluster_score"]),
    }


# -----------------------------------------------------------------------------
# RSI bullish divergence
# -----------------------------------------------------------------------------

def detect_rsi_bullish_divergence(df: pd.DataFrame, lookback: int = 120) -> Tuple[bool, str]:
    """Detect bullish divergence: price makes lower low, RSI makes higher low."""
    recent = df.tail(lookback).copy()
    swings = find_swing_lows(recent, window=4)

    if swings.empty or len(swings) < 2:
        return False, "Not enough swing lows to evaluate RSI divergence."

    last_two = swings.tail(2)
    first = last_two.iloc[0]
    second = last_two.iloc[1]

    price_lower_low = second["Price"] < first["Price"]
    rsi_higher_low = second["RSI14"] > first["RSI14"]

    if price_lower_low and rsi_higher_low:
        explanation = (
            f"Bullish RSI divergence detected: price made a lower low "
            f"({first['Price']:.2f} → {second['Price']:.2f}) while RSI made a higher low "
            f"({first['RSI14']:.1f} → {second['RSI14']:.1f})."
        )
        return True, explanation

    return False, "No bullish RSI divergence detected in the recent swing lows."


# -----------------------------------------------------------------------------
# Candlestick reversal patterns
# -----------------------------------------------------------------------------

def detect_candlestick_patterns(df: pd.DataFrame) -> Tuple[List[str], float]:
    """Detect simple bullish reversal candlestick patterns.

    Returns:
        patterns: names of detected patterns
        score: strength score from 0 to 15
    """
    if len(df) < 3:
        return [], 0.0

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    o, h, l, c = latest["Open"], latest["High"], latest["Low"], latest["Close"]
    po, ph, pl, pc = prev["Open"], prev["High"], prev["Low"], prev["Close"]
    p2o, p2c = prev2["Open"], prev2["Close"]

    body = abs(c - o)
    candle_range = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    patterns: List[str] = []
    score = 0.0

    # Hammer: small body near high, long lower wick.
    if lower_wick >= 2 * max(body, 1e-9) and upper_wick <= body * 1.2 and c > l + candle_range * 0.55:
        patterns.append("Hammer")
        score += 5

    # Bullish engulfing: previous red candle, current green candle engulfs body.
    if pc < po and c > o and c >= po and o <= pc:
        patterns.append("Bullish engulfing")
        score += 6

    # Piercing pattern: prior red, current opens lower but closes above midpoint of prior body.
    prior_mid = (po + pc) / 2
    if pc < po and c > o and o < pc and c > prior_mid and c < po:
        patterns.append("Piercing pattern")
        score += 4

    # Morning star: red candle, small indecision candle, strong green close into first candle body.
    prev2_red = p2c < p2o
    prev_small = abs(pc - po) <= abs(p2c - p2o) * 0.45
    current_green = c > o
    closes_into_first = c > (p2o + p2c) / 2
    if prev2_red and prev_small and current_green and closes_into_first:
        patterns.append("Morning star")
        score += 7

    # Bullish outside day: wider range and higher close.
    if l < pl and h > ph and c > pc and c > o:
        patterns.append("Bullish outside day")
        score += 4

    return patterns, min(score, 15.0)


# -----------------------------------------------------------------------------
# Multi-timeframe confirmation
# -----------------------------------------------------------------------------

def timeframe_state(df: pd.DataFrame) -> Dict[str, object]:
    """Return trend/momentum state for one timeframe."""
    if df.empty or len(df) < 60:
        return {
            "valid": False,
            "score": 0,
            "summary": "Insufficient data",
        }

    required_cols = ["Close", "SMA20", "RSI14", "MACD_Hist", "BB_Lower"]
    x = add_indicators(df).dropna(subset=required_cols).copy()
    if len(x) < 5:
        return {
            "valid": False,
            "score": 0,
            "summary": "Insufficient indicator history",
        }

    latest = x.iloc[-1]
    prev = x.iloc[-2]

    score = 0
    notes = []

    if latest["Close"] > latest["SMA20"]:
        score += 1
        notes.append("above 20-period SMA")

    if latest["RSI14"] > prev["RSI14"] and latest["RSI14"] > 35:
        score += 1
        notes.append("RSI improving")

    if latest["MACD_Hist"] > prev["MACD_Hist"]:
        score += 1
        notes.append("MACD histogram improving")

    if latest["Close"] > latest["BB_Lower"]:
        score += 1
        notes.append("above lower Bollinger Band")

    summary = ", ".join(notes) if notes else "no clear confirmation"
    return {
        "valid": True,
        "score": score,
        "summary": summary,
        "close": float(latest["Close"]),
        "rsi": float(latest["RSI14"]),
    }


def multi_timeframe_confirmation(daily_df: pd.DataFrame) -> Tuple[float, Dict[str, Dict[str, object]]]:
    """Check daily, weekly, and monthly confirmation."""
    weekly = resample_ohlcv(daily_df, "W-FRI")
    monthly = resample_ohlcv(daily_df, "ME")

    states = {
        "Daily": timeframe_state(daily_df),
        "Weekly": timeframe_state(weekly),
        "Monthly": timeframe_state(monthly),
    }

    weights = {
        "Daily": 0.40,
        "Weekly": 0.40,
        "Monthly": 0.20,
    }

    weighted = 0.0
    max_possible = 0.0
    for tf, state in states.items():
        if state["valid"]:
            weighted += float(state["score"]) / 4 * weights[tf]
            max_possible += weights[tf]

    if max_possible == 0:
        return 0.0, states

    confirmation_score = weighted / max_possible * 20
    return float(confirmation_score), states


# -----------------------------------------------------------------------------
# Advanced bottom indicators
# -----------------------------------------------------------------------------

def detect_weekly_rsi_bullish_divergence(daily_df: pd.DataFrame, lookback_weeks: int = 156) -> Tuple[bool, str]:
    """Detect bullish RSI divergence on weekly candles."""
    weekly = resample_ohlcv(daily_df, "W-FRI")
    if len(weekly) < 60:
        return False, "Not enough weekly history to evaluate weekly RSI divergence."

    weekly = add_indicators(weekly).dropna(subset=["RSI14", "Low"]).tail(lookback_weeks)
    swings = find_swing_lows(weekly, window=3)

    if swings.empty or len(swings) < 2:
        return False, "Not enough weekly swing lows to evaluate weekly RSI divergence."

    first = swings.iloc[-2]
    second = swings.iloc[-1]

    price_lower_low = second["Price"] < first["Price"]
    rsi_higher_low = second["RSI14"] > first["RSI14"]

    if price_lower_low and rsi_higher_low:
        return True, (
            f"Weekly bullish RSI divergence detected: weekly price made a lower low "
            f"({first['Price']:.2f} → {second['Price']:.2f}) while weekly RSI made a higher low "
            f"({first['RSI14']:.1f} → {second['RSI14']:.1f})."
        )

    return False, "No weekly bullish RSI divergence detected."


def weekly_200_sma_signal(daily_df: pd.DataFrame) -> SignalResult:
    """Evaluate the 200-week SMA as a long-cycle support/reclaim level."""
    weekly = resample_ohlcv(daily_df, "W-FRI")
    if len(weekly) < 205:
        return SignalResult(
            "200-week SMA",
            0,
            10,
            False,
            "Not enough weekly history to calculate a reliable 200-week SMA.",
        )

    weekly["SMA200W"] = weekly["Close"].rolling(200).mean()
    latest = weekly.dropna().iloc[-1]
    prev = weekly.dropna().iloc[-2]
    close = float(latest["Close"])
    sma200w = float(latest["SMA200W"])
    distance = close / sma200w - 1

    if close > sma200w and float(prev["Close"]) <= float(prev["SMA200W"]):
        return SignalResult(
            "200-week SMA",
            10,
            10,
            True,
            f"Weekly price reclaimed the 200-week SMA near {sma200w:.2f}.",
        )
    if abs(distance) <= 0.05:
        return SignalResult(
            "200-week SMA",
            7,
            10,
            True,
            f"Price is within 5% of the 200-week SMA near {sma200w:.2f}, a major long-cycle reference level.",
        )
    if close > sma200w:
        return SignalResult(
            "200-week SMA",
            5,
            10,
            True,
            f"Weekly price remains above the 200-week SMA near {sma200w:.2f}.",
        )

    return SignalResult(
        "200-week SMA",
        0,
        10,
        False,
        f"Weekly price is below the 200-week SMA near {sma200w:.2f}; long-cycle support has not been reclaimed.",
    )


def yearly_anchored_vwap_signal(df: pd.DataFrame) -> SignalResult:
    """Evaluate yearly anchored VWAP reclaim/support."""
    x = df.dropna(subset=["Yearly_AVWAP", "ATR14"]).copy()
    if len(x) < 2:
        return SignalResult("Yearly anchored VWAP", 0, 8, False, "Not enough data to evaluate yearly anchored VWAP.")

    latest = x.iloc[-1]
    prev = x.iloc[-2]
    close = float(latest["Close"])
    avwap = float(latest["Yearly_AVWAP"])
    atr = float(latest["ATR14"])

    if close > avwap and float(prev["Close"]) <= float(prev["Yearly_AVWAP"]):
        return SignalResult("Yearly anchored VWAP", 8, 8, True, f"Price reclaimed the yearly anchored VWAP near {avwap:.2f}.")
    if close > avwap:
        return SignalResult("Yearly anchored VWAP", 5, 8, True, f"Price is above the yearly anchored VWAP near {avwap:.2f}.")
    if abs(close - avwap) <= atr:
        return SignalResult("Yearly anchored VWAP", 4, 8, True, f"Price is within 1 ATR of the yearly anchored VWAP near {avwap:.2f}.")

    return SignalResult("Yearly anchored VWAP", 0, 8, False, f"Price is below yearly anchored VWAP near {avwap:.2f}; reclaim is not confirmed.")


def linear_regression_channel_signal(df: pd.DataFrame) -> SignalResult:
    """Evaluate price position inside the 2-year linear regression channel."""
    x = df.dropna(subset=["LRC_Lower_2Y", "LRC_Mid_2Y", "LRC_Upper_2Y", "LRC_Position_2Y"]).copy()
    if len(x) < 2:
        return SignalResult("2-year linear regression channel", 0, 10, False, "Not enough history to calculate the 2-year linear regression channel.")

    latest = x.iloc[-1]
    prev = x.iloc[-2]
    close = float(latest["Close"])
    lower = float(latest["LRC_Lower_2Y"])
    mid = float(latest["LRC_Mid_2Y"])
    position = float(latest["LRC_Position_2Y"])

    if close > lower and float(prev["Close"]) <= float(prev["LRC_Lower_2Y"]):
        return SignalResult("2-year linear regression channel", 10, 10, True, f"Price reclaimed the lower 2-year regression channel near {lower:.2f}.")
    if 0 <= position <= 0.25:
        return SignalResult("2-year linear regression channel", 7, 10, True, f"Price is in the lower quartile of the 2-year regression channel; lower band is near {lower:.2f}.")
    if close > mid:
        return SignalResult("2-year linear regression channel", 4, 10, True, f"Price is above the 2-year regression midline near {mid:.2f}.")

    return SignalResult("2-year linear regression channel", 0, 10, False, "Price is not near or reclaiming the lower 2-year regression channel.")


def coppock_curve_signal(daily_df: pd.DataFrame) -> SignalResult:
    """Evaluate the Coppock Curve using monthly candles.

    Standard approximation: 10-period weighted moving average of the sum of
    14-month and 11-month rate of change.
    """
    monthly = resample_ohlcv(daily_df, "ME")
    if len(monthly) < 30:
        return SignalResult("Coppock Curve", 0, 10, False, "Not enough monthly history to calculate the Coppock Curve.")

    close = monthly["Close"]
    roc14 = close.pct_change(14) * 100
    roc11 = close.pct_change(11) * 100
    coppock = weighted_moving_average(roc14 + roc11, 10)
    monthly = monthly.assign(Coppock=coppock).dropna()

    if len(monthly) < 3:
        return SignalResult("Coppock Curve", 0, 10, False, "Not enough valid Coppock Curve history.")

    latest = float(monthly["Coppock"].iloc[-1])
    prev = float(monthly["Coppock"].iloc[-2])
    prev2 = float(monthly["Coppock"].iloc[-3])

    if latest > 0 and prev <= 0:
        return SignalResult("Coppock Curve", 10, 10, True, f"Coppock Curve crossed above zero ({latest:.2f}), a major long-term momentum reversal signal.")
    if latest > prev and prev <= prev2 and latest < 0:
        return SignalResult("Coppock Curve", 8, 10, True, f"Coppock Curve turned up from below zero ({latest:.2f}), often an early bottoming signal.")
    if latest > prev:
        return SignalResult("Coppock Curve", 5, 10, True, f"Coppock Curve is rising ({latest:.2f}), showing improving long-term momentum.")

    return SignalResult("Coppock Curve", 0, 10, False, f"Coppock Curve is not rising yet ({latest:.2f}).")


def weekly_macd_bullish_cross_signal(daily_df: pd.DataFrame) -> SignalResult:
    """Detect a weekly MACD bullish cross."""
    weekly = add_indicators(resample_ohlcv(daily_df, "W-FRI")).dropna(subset=["MACD", "MACD_Signal", "MACD_Hist"])
    if len(weekly) < 3:
        return SignalResult("Weekly MACD bullish cross", 0, 10, False, "Not enough weekly data to evaluate weekly MACD.")

    latest = weekly.iloc[-1]
    prev = weekly.iloc[-2]

    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        return SignalResult("Weekly MACD bullish cross", 10, 10, True, "Weekly MACD crossed above its signal line.")
    if latest["MACD_Hist"] > prev["MACD_Hist"] and latest["MACD_Hist"] < 0:
        return SignalResult("Weekly MACD bullish cross", 6, 10, True, "Weekly MACD histogram is improving below zero; bullish cross is not confirmed yet.")
    if latest["MACD"] > latest["MACD_Signal"]:
        return SignalResult("Weekly MACD bullish cross", 5, 10, True, "Weekly MACD is above its signal line, but the cross is not new.")

    return SignalResult("Weekly MACD bullish cross", 0, 10, False, "Weekly MACD bullish cross has not occurred.")


def stage_analysis_signal(daily_df: pd.DataFrame) -> Tuple[SignalResult, Dict[str, object]]:
    """Stan Weinstein-style stage analysis using weekly closes and the 30-week SMA."""
    weekly = resample_ohlcv(daily_df, "W-FRI")
    if len(weekly) < 40:
        return SignalResult("Stage Analysis", 0, 10, False, "Not enough weekly history for Stage Analysis."), {}

    weekly["SMA10W"] = weekly["Close"].rolling(10).mean()
    weekly["SMA30W"] = weekly["Close"].rolling(30).mean()
    weekly["SMA200W"] = weekly["Close"].rolling(200).mean()
    weekly = weekly.dropna(subset=["SMA30W"])

    if len(weekly) < 12:
        return SignalResult("Stage Analysis", 0, 10, False, "Not enough valid weekly moving-average history for Stage Analysis."), {}

    latest = weekly.iloc[-1]
    close = float(latest["Close"])
    sma30 = float(latest["SMA30W"])
    sma10 = float(latest["SMA10W"]) if not pd.isna(latest["SMA10W"]) else np.nan
    sma30_10w_ago = float(weekly["SMA30W"].iloc[-10])
    slope_10w = sma30 / sma30_10w_ago - 1 if sma30_10w_ago else 0
    above_30w = close > sma30
    flattening = abs(slope_10w) <= 0.03

    if above_30w and slope_10w > 0.02 and (pd.isna(sma10) or close >= sma10 * 0.97):
        stage = "Stage 2: advancing phase"
        points = 10
        passed = True
        explanation = f"Stan Weinstein Stage Analysis indicates {stage}; price is above a rising 30-week SMA near {sma30:.2f}."
    elif flattening and abs(close / sma30 - 1) <= 0.08:
        stage = "Stage 1: base-building phase"
        points = 8
        passed = True
        explanation = f"Stan Weinstein Stage Analysis indicates {stage}; price is near a flattening 30-week SMA near {sma30:.2f}."
    elif close < sma30 and slope_10w < -0.02:
        stage = "Stage 4: declining phase"
        points = 0
        passed = False
        explanation = f"Stan Weinstein Stage Analysis indicates {stage}; price is below a falling 30-week SMA near {sma30:.2f}."
    else:
        stage = "Stage 3 or transition phase"
        points = 3
        passed = False
        explanation = f"Stan Weinstein Stage Analysis indicates {stage}; trend is not yet a clean base or advance."

    details = {
        "stage": stage,
        "weekly_close": close,
        "sma30w": sma30,
        "sma200w": float(latest["SMA200W"]) if "SMA200W" in latest and not pd.isna(latest["SMA200W"]) else np.nan,
        "sma30w_slope_10w": slope_10w,
    }
    return SignalResult("Stage Analysis", points, 10, passed, explanation), details


def obv_trend_break_signal(df: pd.DataFrame) -> SignalResult:
    """Detect whether OBV has broken above a recent trend/accumulation range."""
    x = df.dropna(subset=["OBV", "OBV_EMA20", "OBV_Rolling_High20"]).copy()
    if len(x) < 30:
        return SignalResult("OBV trend break", 0, 10, False, "Not enough OBV history to evaluate trend break.")

    latest = x.iloc[-1]
    prev = x.iloc[-2]
    obv_slope_10 = float(x["OBV"].iloc[-1] - x["OBV"].iloc[-11]) if len(x) >= 11 else 0
    prior_20_high = float(x["OBV"].rolling(20).max().shift(1).iloc[-1])

    if latest["OBV"] > prior_20_high and latest["OBV"] > latest["OBV_EMA20"]:
        return SignalResult("OBV trend break", 10, 10, True, "OBV broke above its prior 20-day high and is above its 20-day EMA, suggesting accumulation.")
    if latest["OBV"] > latest["OBV_EMA20"] and obv_slope_10 > 0:
        return SignalResult("OBV trend break", 6, 10, True, "OBV is above its 20-day EMA and trending higher, but a breakout is not fully confirmed.")
    if latest["OBV"] > prev["OBV"]:
        return SignalResult("OBV trend break", 3, 10, True, "OBV improved versus the prior candle, but trend-break confirmation is weak.")

    return SignalResult("OBV trend break", 0, 10, False, "OBV has not broken its recent trend range.")


def historical_volatility_percentile_signal(df: pd.DataFrame) -> SignalResult:
    """Evaluate historical volatility percentile for bottom-quality context."""
    x = df.dropna(subset=["HV20", "HV20_Pctile_252"]).copy()
    if len(x) < 30:
        return SignalResult("HV percentile", 0, 8, False, "Not enough historical volatility data to calculate percentile.")

    latest = x.iloc[-1]
    pct = float(latest["HV20_Pctile_252"])
    hv = float(latest["HV20"])
    pct_10 = float(x["HV20_Pctile_252"].iloc[-10]) if len(x) >= 10 else pct

    if pct_10 >= 75 and pct <= pct_10 - 10:
        return SignalResult("HV percentile", 8, 8, True, f"Historical volatility percentile is cooling from an elevated level ({pct_10:.0f} → {pct:.0f}), suggesting panic may be fading.")
    if pct <= 50:
        return SignalResult("HV percentile", 5, 8, True, f"Historical volatility is not elevated; 20-day HV is {hv:.1%} at the {pct:.0f}th percentile.")
    if pct >= 85:
        return SignalResult("HV percentile", 0, 8, False, f"Historical volatility is extremely elevated at the {pct:.0f}th percentile, which increases knife-catching risk.")

    return SignalResult("HV percentile", 2, 8, False, f"Historical volatility is mid/high at the {pct:.0f}th percentile; volatility confirmation is mixed.")


def atr_compression_signal(df: pd.DataFrame) -> SignalResult:
    """Evaluate ATR compression after a decline."""
    x = df.dropna(subset=["ATR_Pct", "ATR_Pct_SMA20", "ATR_Pctile_252"]).copy()
    if len(x) < 30:
        return SignalResult("ATR compression", 0, 8, False, "Not enough ATR history to evaluate compression.")

    latest = x.iloc[-1]
    atr_pct = float(latest["ATR_Pct"])
    atr_pctile = float(latest["ATR_Pctile_252"])
    atr_avg = float(latest["ATR_Pct_SMA20"])

    if atr_pctile <= 35 and atr_pct < atr_avg:
        return SignalResult("ATR compression", 8, 8, True, f"ATR is compressed at the {atr_pctile:.0f}th percentile and below its 20-day average, indicating volatility contraction.")
    if atr_pctile <= 50:
        return SignalResult("ATR compression", 4, 8, True, f"ATR is moderately compressed at the {atr_pctile:.0f}th percentile.")

    return SignalResult("ATR compression", 0, 8, False, f"ATR is not compressed; ATR% is at the {atr_pctile:.0f}th percentile.")


# -----------------------------------------------------------------------------
# Production filters: liquidity, slippage, relative strength, market regime
# -----------------------------------------------------------------------------

def liquidity_filter_signal(
    df: pd.DataFrame,
    min_avg_volume: float,
    min_avg_traded_value: float,
    max_atr_pct: float,
) -> Tuple[SignalResult, Dict[str, float]]:
    """Check whether the stock is liquid enough for the strategy."""
    x = df.dropna(subset=["Close", "Volume", "ATR_Pct"]).copy()
    if len(x) < 20:
        return SignalResult("Liquidity filter", 0, 10, False, "Not enough data to evaluate liquidity."), {}

    recent = x.tail(20)
    avg_volume_20 = float(recent["Volume"].mean())
    avg_traded_value_20 = float((recent["Close"] * recent["Volume"]).mean())
    latest_atr_pct = float(x["ATR_Pct"].iloc[-1])

    volume_ok = avg_volume_20 >= min_avg_volume
    traded_value_ok = avg_traded_value_20 >= min_avg_traded_value
    volatility_ok = latest_atr_pct <= max_atr_pct

    passed_count = sum([volume_ok, traded_value_ok, volatility_ok])
    points = {3: 10, 2: 5, 1: 2, 0: 0}[passed_count]
    passed = passed_count == 3

    explanation = (
        f"20-day average volume is {avg_volume_20:,.0f}, 20-day average traded value is "
        f"{avg_traded_value_20:,.0f}, and ATR% is {latest_atr_pct:.1%}. "
        f"Liquidity filter {'passed' if passed else 'is mixed/failed'}."
    )

    details = {
        "avg_volume_20": avg_volume_20,
        "avg_traded_value_20": avg_traded_value_20,
        "latest_atr_pct": latest_atr_pct,
    }
    return SignalResult("Liquidity filter", points, 10, passed, explanation), details


def slippage_filter_signal(df: pd.DataFrame, order_value: float) -> Tuple[SignalResult, Dict[str, float]]:
    """Estimate rough slippage using order size, average traded value, and ATR%."""
    x = df.dropna(subset=["Close", "Volume", "ATR_Pct"]).copy()
    if len(x) < 20:
        return SignalResult("Slippage filter", 0, 8, False, "Not enough data to estimate slippage."), {}

    recent = x.tail(20)
    avg_traded_value_20 = float((recent["Close"] * recent["Volume"]).mean())
    atr_pct = float(x["ATR_Pct"].iloc[-1])

    if avg_traded_value_20 <= 0:
        return SignalResult("Slippage filter", 0, 8, False, "Average traded value is not available for slippage estimation."), {}

    participation = max(order_value, 0) / avg_traded_value_20

    base_bps = 5.0
    impact_bps = np.sqrt(max(participation, 0)) * 100
    volatility_bps = atr_pct * 50
    estimated_slippage_bps = float(base_bps + impact_bps + volatility_bps)

    if estimated_slippage_bps <= 25 and participation <= 0.02:
        points = 8
        passed = True
    elif estimated_slippage_bps <= 75 and participation <= 0.05:
        points = 4
        passed = True
    else:
        points = 0
        passed = False

    explanation = (
        f"Estimated one-way slippage is approximately {estimated_slippage_bps:.1f} bps for an order value of "
        f"{order_value:,.0f}. Estimated participation is {participation:.2%} of 20-day average traded value."
    )

    details = {
        "avg_traded_value_20": avg_traded_value_20,
        "participation": participation,
        "estimated_slippage_bps": estimated_slippage_bps,
    }
    return SignalResult("Slippage filter", points, 8, passed, explanation), details


def sector_relative_strength_signal(stock_df: pd.DataFrame, sector_df: Optional[pd.DataFrame]) -> SignalResult:
    """Measure stock performance relative to a sector/industry benchmark."""
    if sector_df is None or sector_df.empty:
        return SignalResult(
            "Sector-relative strength",
            0,
            0,
            False,
            "No sector benchmark ticker was provided, so sector-relative strength was excluded from the score.",
        )

    merged = pd.DataFrame({"Stock": stock_df["Close"], "Sector": sector_df["Close"]}).dropna()
    if len(merged) < 220:
        return SignalResult("Sector-relative strength", 0, 10, False, "Not enough overlapping stock/sector data for relative-strength analysis.")

    merged["RS"] = merged["Stock"] / merged["Sector"]
    merged["RS_SMA50"] = merged["RS"].rolling(50).mean()
    merged["RS_SMA200"] = merged["RS"].rolling(200).mean()
    merged = merged.dropna()

    latest = merged.iloc[-1]
    rs_slope_20 = float(merged["RS"].iloc[-1] / merged["RS"].iloc[-21] - 1) if len(merged) >= 21 else 0

    if latest["RS"] > latest["RS_SMA50"] and latest["RS_SMA50"] > latest["RS_SMA200"] and rs_slope_20 > 0:
        return SignalResult("Sector-relative strength", 10, 10, True, "Stock is outperforming its sector benchmark with relative strength above rising medium/long-term averages.")
    if latest["RS"] > latest["RS_SMA50"] and rs_slope_20 > 0:
        return SignalResult("Sector-relative strength", 6, 10, True, "Stock relative strength is improving versus the sector benchmark.")
    if rs_slope_20 > 0:
        return SignalResult("Sector-relative strength", 3, 10, True, "Stock has short-term relative-strength improvement, but the broader RS trend is not confirmed.")

    return SignalResult("Sector-relative strength", 0, 10, False, "Stock is not yet showing relative-strength improvement versus the sector benchmark.")


def market_regime_filter_signal(
    market_df: Optional[pd.DataFrame],
    volatility_df: Optional[pd.DataFrame] = None,
) -> Tuple[SignalResult, Dict[str, object]]:
    """Use index trend and volatility to determine whether bottom signals deserve support."""
    if market_df is None or market_df.empty:
        return SignalResult(
            "Market-regime filter",
            0,
            0,
            False,
            "No market benchmark ticker was provided, so market-regime filtering was excluded from the score.",
        ), {}

    m = add_indicators(market_df).dropna(subset=["Close", "SMA50", "SMA200", "HV20_Pctile_252"]).copy()
    if len(m) < 260:
        return SignalResult("Market-regime filter", 0, 12, False, "Not enough benchmark history for market-regime filtering."), {}

    latest = m.iloc[-1]
    close = float(latest["Close"])
    sma50 = float(latest["SMA50"])
    sma200 = float(latest["SMA200"])
    sma200_slope_20 = float(sma200 / m["SMA200"].iloc[-21] - 1) if len(m) >= 21 else 0
    market_hv_pctile = float(latest["HV20_Pctile_252"])

    points = 0
    notes = []

    if close > sma200:
        points += 4
        notes.append("benchmark is above its 200-DMA")
    if sma200_slope_20 > 0:
        points += 3
        notes.append("200-DMA is rising")
    if close > sma50:
        points += 2
        notes.append("benchmark is above its 50-DMA")
    if market_hv_pctile <= 75:
        points += 2
        notes.append(f"benchmark realized volatility is not extreme ({market_hv_pctile:.0f}th percentile)")

    vol_note = ""
    if volatility_df is not None and not volatility_df.empty and len(volatility_df) >= 60:
        v = volatility_df.dropna(subset=["Close"]).copy()
        v["SMA50"] = v["Close"].rolling(50).mean()
        v = v.dropna()
        if len(v) > 0:
            v_latest = float(v["Close"].iloc[-1])
            v_sma50 = float(v["SMA50"].iloc[-1])
            if v_latest <= v_sma50 * 1.10:
                points += 1
                vol_note = f" volatility index is contained near/below its 50-DMA ({v_latest:.2f} vs {v_sma50:.2f})."
            else:
                vol_note = f" volatility index is elevated versus its 50-DMA ({v_latest:.2f} vs {v_sma50:.2f})."

    points = min(points, 12)
    passed = points >= 8
    summary = ", ".join(notes) if notes else "benchmark trend/volatility backdrop is weak"
    explanation = f"Market regime: {summary}.{vol_note}"

    details = {
        "benchmark_close": close,
        "benchmark_sma50": sma50,
        "benchmark_sma200": sma200,
        "benchmark_sma200_slope_20": sma200_slope_20,
        "benchmark_hv_pctile": market_hv_pctile,
    }
    return SignalResult("Market-regime filter", points, 12, passed, explanation), details


# -----------------------------------------------------------------------------
# Survivorship-bias-free constituent-history support
# -----------------------------------------------------------------------------

def parse_constituent_history(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Parse a point-in-time constituent CSV.

    Recommended columns:
        Ticker, StartDate, EndDate, Sector

    EndDate may be blank for currently active constituents.
    If only Ticker is supplied, the list is treated as a static universe and is
    not survivorship-bias-free.
    """
    raw = pd.read_csv(uploaded_file)
    if raw.empty:
        return pd.DataFrame(), "The uploaded CSV is empty."

    col_map = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}

    ticker_col = None
    for candidate in ["ticker", "symbol", "constituent", "security", "nse_symbol", "bbg_ticker"]:
        if candidate in col_map:
            ticker_col = col_map[candidate]
            break

    if ticker_col is None:
        return pd.DataFrame(), "CSV must contain a ticker/symbol column."

    start_col = None
    for candidate in ["startdate", "start_date", "date_added", "added_date", "entry_date", "from", "date"]:
        if candidate in col_map:
            start_col = col_map[candidate]
            break

    end_col = None
    for candidate in ["enddate", "end_date", "date_removed", "removed_date", "exit_date", "to"]:
        if candidate in col_map:
            end_col = col_map[candidate]
            break

    sector_col = None
    for candidate in ["sector", "industry", "gics_sector"]:
        if candidate in col_map:
            sector_col = col_map[candidate]
            break

    out = pd.DataFrame()
    out["Ticker"] = raw[ticker_col].astype(str).str.strip().str.upper()
    out = out[out["Ticker"] != ""]

    if start_col is not None:
        out["StartDate"] = pd.to_datetime(raw[start_col], errors="coerce")
    else:
        out["StartDate"] = pd.Timestamp("1900-01-01")

    if end_col is not None:
        out["EndDate"] = pd.to_datetime(raw[end_col], errors="coerce")
    else:
        out["EndDate"] = pd.NaT

    if sector_col is not None:
        out["Sector"] = raw[sector_col].astype(str)
    else:
        out["Sector"] = ""

    has_point_in_time = start_col is not None or end_col is not None
    note = (
        "Point-in-time constituent file detected. Index testing can avoid survivorship bias if the file contains historical adds/removals."
        if has_point_in_time
        else "Only a static ticker list was detected. This is useful for batch testing, but it is not survivorship-bias-free."
    )
    return out.drop_duplicates(), note


def active_constituents_on(constituents: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Filter constituent history to active members on a specific date."""
    if constituents.empty:
        return pd.DataFrame()

    asof = pd.to_datetime(as_of_date).normalize()
    x = constituents.copy()
    x["StartDate"] = pd.to_datetime(x["StartDate"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    x["EndDate"] = pd.to_datetime(x["EndDate"], errors="coerce")

    active = x[(x["StartDate"] <= asof) & (x["EndDate"].isna() | (x["EndDate"] >= asof))].copy()
    return active.sort_values("Ticker")


# -----------------------------------------------------------------------------
# Technical bottom scoring model
# -----------------------------------------------------------------------------

def nearest_zone(zones: List[Zone], current_price: float, direction: str = "support") -> Optional[Zone]:
    if not zones:
        return None

    if direction == "support":
        candidates = [z for z in zones if z.center <= current_price * 1.05]
    else:
        candidates = [z for z in zones if z.center >= current_price * 0.95]

    if not candidates:
        candidates = zones

    return sorted(candidates, key=lambda z: abs(current_price - z.center))[0]


def calculate_bottom_score(
    df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
    volatility_df: Optional[pd.DataFrame] = None,
    min_avg_volume: float = 100_000,
    min_avg_traded_value: float = 5_000_000,
    max_atr_pct: float = 0.12,
    order_value: float = 100_000,
    include_production_filters: bool = False,
) -> Tuple[float, List[SignalResult], Dict[str, object]]:
    """Calculate the bottom score using technical signals."""
    core_cols = [
        "Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "SMA200",
        "RSI14", "MACD", "MACD_Signal", "MACD_Hist", "ATR14", "BB_Lower",
        "Volume_SMA20", "Volume_Ratio", "Rolling_52W_High", "Rolling_52W_Low", "Drawdown_52W",
    ]
    x = df.dropna(subset=[c for c in core_cols if c in df.columns]).copy()
    if len(x) < 260:
        raise ValueError("Need at least ~1 year of daily data after core indicator calculations to calculate a robust score.")

    latest = x.iloc[-1]
    prev = x.iloc[-2]

    current_price = float(latest["Close"])
    atr = float(latest["ATR14"])
    atr = atr if atr > 0 and not math.isnan(atr) else current_price * 0.02

    supports, resistances = get_support_resistance_zones(x, lookback=252)
    support_zone = nearest_zone(supports, current_price, "support")
    resistance_zone = nearest_zone(resistances, current_price, "resistance")
    vp_zone = volume_profile_zone(x, lookback=252, bins=40)
    divergence, divergence_msg = detect_rsi_bullish_divergence(x, lookback=120)
    weekly_divergence, weekly_divergence_msg = detect_weekly_rsi_bullish_divergence(x, lookback_weeks=156)
    patterns, candle_score = detect_candlestick_patterns(x)
    mtf_score, mtf_states = multi_timeframe_confirmation(x)

    weekly_200w = weekly_200_sma_signal(x)
    yearly_avwap = yearly_anchored_vwap_signal(x)
    lrc_signal = linear_regression_channel_signal(x)
    coppock_signal_result = coppock_curve_signal(x)
    weekly_macd_cross = weekly_macd_bullish_cross_signal(x)
    stage_signal, stage_details = stage_analysis_signal(x)
    obv_signal = obv_trend_break_signal(x)
    hv_signal = historical_volatility_percentile_signal(x)
    atr_compression = atr_compression_signal(x)
    liquidity_details = {}
    slippage_details = {}
    market_regime_details = {}

    if include_production_filters:
        liquidity_signal, liquidity_details = liquidity_filter_signal(
            x,
            min_avg_volume=min_avg_volume,
            min_avg_traded_value=min_avg_traded_value,
            max_atr_pct=max_atr_pct,
        )
        slippage_signal, slippage_details = slippage_filter_signal(x, order_value=order_value)
        sector_rs_signal = sector_relative_strength_signal(x, sector_df)
        market_regime_signal_result, market_regime_details = market_regime_filter_signal(market_df, volatility_df)

    signals: List[SignalResult] = []

    # 1. Price near clustered support
    support_points = 0.0
    if support_zone:
        distance_to_support = abs(current_price - support_zone.center)
        if distance_to_support <= atr:
            support_points = 20
            passed = True
            explanation = (
                f"Price is within 1 ATR of clustered support near {support_zone.center:.2f}."
            )
        elif distance_to_support <= 2 * atr:
            support_points = 12
            passed = True
            explanation = (
                f"Price is within 2 ATR of clustered support near {support_zone.center:.2f}."
            )
        else:
            passed = False
            explanation = (
                f"Nearest clustered support is around {support_zone.center:.2f}, but price is not close enough."
            )
    else:
        passed = False
        explanation = "No reliable clustered support zone found."

    signals.append(SignalResult("Clustered support proximity", support_points, 20, passed, explanation))

    # 2. RSI oversold and recovering
    rsi_points = 0.0
    if latest["RSI14"] < 35 and latest["RSI14"] > prev["RSI14"]:
        rsi_points = 15
        passed = True
        explanation = f"RSI is oversold and rising ({latest['RSI14']:.1f})."
    elif latest["RSI14"] < 40 and latest["RSI14"] > prev["RSI14"]:
        rsi_points = 10
        passed = True
        explanation = f"RSI is below 40 and improving ({latest['RSI14']:.1f})."
    else:
        passed = False
        explanation = f"RSI is not giving a strong oversold-recovery signal ({latest['RSI14']:.1f})."
    signals.append(SignalResult("RSI recovery", rsi_points, 15, passed, explanation))

    # 3. MACD momentum improvement
    macd_points = 0.0
    macd_hist_slope_3 = latest["MACD_Hist"] - x["MACD_Hist"].iloc[-4]
    if latest["MACD_Hist"] > prev["MACD_Hist"] and macd_hist_slope_3 > 0:
        macd_points = 15
        passed = True
        explanation = "MACD histogram is improving over both 1-day and 3-day windows."
    elif latest["MACD_Hist"] > prev["MACD_Hist"]:
        macd_points = 9
        passed = True
        explanation = "MACD histogram improved versus the prior candle."
    else:
        passed = False
        explanation = "MACD histogram is not yet improving."
    signals.append(SignalResult("MACD momentum improvement", macd_points, 15, passed, explanation))

    # 4. Bollinger Band reclaim / mean reversion
    bb_points = 0.0
    if latest["Close"] > latest["BB_Lower"] and prev["Close"] < prev["BB_Lower"]:
        bb_points = 10
        passed = True
        explanation = "Price reclaimed the lower Bollinger Band after closing below it."
    elif latest["Close"] > latest["BB_Lower"] and latest["RSI14"] < 45:
        bb_points = 5
        passed = True
        explanation = "Price is holding above the lower Bollinger Band while momentum remains depressed."
    else:
        passed = False
        explanation = "No Bollinger Band reclaim signal."
    signals.append(SignalResult("Bollinger Band reclaim", bb_points, 10, passed, explanation))

    # 5. Volume capitulation / accumulation
    volume_points = 0.0
    if latest["Volume_Ratio"] >= 1.8 and latest["Close"] >= latest["Open"]:
        volume_points = 15
        passed = True
        explanation = f"High-volume bullish candle; volume is {latest['Volume_Ratio']:.1f}x the 20-day average."
    elif latest["Volume_Ratio"] >= 1.5:
        volume_points = 10
        passed = True
        explanation = f"Volume spike detected; volume is {latest['Volume_Ratio']:.1f}x the 20-day average."
    else:
        passed = False
        explanation = f"No major volume spike; volume is {latest['Volume_Ratio']:.1f}x the 20-day average."
    signals.append(SignalResult("Volume confirmation", volume_points, 15, passed, explanation))

    # 6. Price reclaiming short-term moving average
    ma_points = 0.0
    if latest["Close"] > latest["SMA20"] and prev["Close"] <= prev["SMA20"]:
        ma_points = 10
        passed = True
        explanation = "Price reclaimed the 20-day moving average."
    elif latest["Close"] > latest["SMA20"]:
        ma_points = 6
        passed = True
        explanation = "Price is above the 20-day moving average."
    else:
        passed = False
        explanation = "Price has not reclaimed the 20-day moving average."
    signals.append(SignalResult("20-DMA reclaim", ma_points, 10, passed, explanation))

    # 7. Deep correction / near 52-week low context
    dd_points = 0.0
    near_52w_low = current_price <= float(latest["Rolling_52W_Low"]) * 1.12
    deeply_corrected = float(latest["Drawdown_52W"]) <= -0.25
    if deeply_corrected and near_52w_low:
        dd_points = 10
        passed = True
        explanation = f"Stock is deeply corrected ({latest['Drawdown_52W']:.1%}) and near its 52-week low zone."
    elif deeply_corrected:
        dd_points = 5
        passed = True
        explanation = f"Stock is deeply corrected from its 52-week high ({latest['Drawdown_52W']:.1%})."
    else:
        passed = False
        explanation = f"Stock is not in a deep correction versus its 52-week high ({latest['Drawdown_52W']:.1%})."
    signals.append(SignalResult("Deep correction context", dd_points, 10, passed, explanation))

    # 8. RSI bullish divergence
    div_points = 10 if divergence else 0
    signals.append(SignalResult("RSI bullish divergence", div_points, 10, divergence, divergence_msg))

    # 9. Volume-profile support
    vp_points = 0.0
    if vp_zone:
        vp_distance = abs(current_price - vp_zone.center)
        if vp_distance <= atr:
            vp_points = 10
            passed = True
            explanation = f"Price is close to a volume-profile high-volume support node near {vp_zone.center:.2f}."
        elif vp_distance <= 2 * atr:
            vp_points = 6
            passed = True
            explanation = f"Price is moderately close to volume-profile support near {vp_zone.center:.2f}."
        else:
            passed = False
            explanation = f"Volume-profile support exists near {vp_zone.center:.2f}, but price is not close enough."
    else:
        passed = False
        explanation = "No reliable volume-profile support node detected."
    signals.append(SignalResult("Volume-profile support", vp_points, 10, passed, explanation))

    # 10. Support confluence quality
    support_estimate = robust_support_confluence(x, support_zone, vp_zone, atr)
    confluence_points = float(support_estimate["confluence_points"])
    confluence_score = float(support_estimate["confluence_score"])
    selected_cluster = support_estimate["selected_cluster"]
    source_count = int(selected_cluster.get("source_count", 0))
    confluence_passed = confluence_points >= 10
    signals.append(
        SignalResult(
            "Support confluence quality",
            confluence_points,
            15,
            confluence_passed,
            f"Support confluence score is {confluence_score:.1f}; selected support cluster has {source_count} independent source type(s) around {support_estimate['blended_support']:.2f}.",
        )
    )

    # 11. Candlestick reversal patterns
    candle_points = candle_score
    candle_passed = candle_score > 0
    candle_explanation = (
        f"Detected bullish reversal pattern(s): {', '.join(patterns)}."
        if patterns
        else "No bullish reversal candlestick pattern detected on the latest candle."
    )
    signals.append(SignalResult("Candlestick reversal", candle_points, 15, candle_passed, candle_explanation))

    # 12. Multiple-timeframe confirmation
    mtf_passed = mtf_score >= 10
    signals.append(
        SignalResult(
            "Multiple-timeframe confirmation",
            mtf_score,
            20,
            mtf_passed,
            f"Multi-timeframe confirmation score is {mtf_score:.1f}/20.",
        )
    )

    # 13. 200-week SMA
    signals.append(weekly_200w)

    # 14. Yearly anchored VWAP
    signals.append(yearly_avwap)

    # 15. 2-year linear regression channel
    signals.append(lrc_signal)

    # 16. Weekly RSI bullish divergence
    signals.append(
        SignalResult(
            "Weekly RSI bullish divergence",
            10 if weekly_divergence else 0,
            10,
            weekly_divergence,
            weekly_divergence_msg,
        )
    )

    # 17. Coppock Curve
    signals.append(coppock_signal_result)

    # 18. Weekly MACD bullish cross
    signals.append(weekly_macd_cross)

    # 19. Fixed-range volume profile is already included above as volume-profile support.
    # Keep this as a named confirmation so the signal table explicitly shows it.
    if vp_zone:
        signals.append(
            SignalResult(
                "Volume Profile Fixed Range",
                min(vp_points, 8),
                8,
                vp_points > 0,
                f"Fixed-range volume profile over the lookback window identifies a high-volume node near {vp_zone.center:.2f}.",
            )
        )
    else:
        signals.append(
            SignalResult(
                "Volume Profile Fixed Range",
                0,
                8,
                False,
                "No reliable fixed-range volume-profile high-volume node was detected.",
            )
        )

    # 20. Stan Weinstein Stage Analysis
    signals.append(stage_signal)

    # 21. OBV trend break
    signals.append(obv_signal)

    # 22. Historical volatility percentile
    signals.append(hv_signal)

    # 23. ATR compression
    signals.append(atr_compression)

    if include_production_filters:
        # 23. Liquidity filter
        signals.append(liquidity_signal)

        # 24. Slippage filter
        signals.append(slippage_signal)

        # 25. Sector-relative strength
        signals.append(sector_rs_signal)

        # 26. Market-regime filter
        signals.append(market_regime_signal_result)

    raw_score = sum(s.points for s in signals)
    max_score = sum(s.max_points for s in signals)
    normalized_score = raw_score / max_score * 100 if max_score else 0

    # Build final support zone using the robust support-confluence engine.
    blended_support = float(support_estimate["blended_support"])
    bottom_zone_lower = float(support_estimate["bottom_zone_lower"])
    bottom_zone_upper = float(support_estimate["bottom_zone_upper"])
    invalidation = float(support_estimate["invalidation"])

    context = {
        "current_price": current_price,
        "atr": atr,
        "support_zone": support_zone,
        "resistance_zone": resistance_zone,
        "volume_profile_zone": vp_zone,
        "blended_support": blended_support,
        "bottom_zone_lower": bottom_zone_lower,
        "bottom_zone_upper": bottom_zone_upper,
        "invalidation": invalidation,
        "support_components": support_estimate.get("components", []),
        "support_clusters": support_estimate.get("clusters", []),
        "selected_support_cluster": support_estimate.get("selected_cluster", {}),
        "zone_atr_multiplier": support_estimate.get("zone_atr_multiplier", 1.0),
        "invalidation_atr_multiplier": support_estimate.get("invalidation_atr_multiplier", 1.5),
        "support_confluence_score": support_estimate.get("confluence_score", np.nan),
        "patterns": patterns,
        "mtf_states": mtf_states,
        "stage_details": stage_details,
        "liquidity_details": liquidity_details,
        "slippage_details": slippage_details,
        "market_regime_details": market_regime_details,
        "raw_score": raw_score,
        "max_score": max_score,
    }

    return float(normalized_score), signals, context


# -----------------------------------------------------------------------------
# Backtesting module
# -----------------------------------------------------------------------------

def max_drawdown_during_period(prices: pd.Series) -> float:
    """Return max drawdown over a short forward window."""
    if prices.empty:
        return 0.0
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    return float(drawdown.min())


def historical_signal_proxy_score(df: pd.DataFrame, i: int) -> float:
    """Lightweight proxy of bottom score for historical backtesting.

    This avoids calling the full clustering engine thousands of times.
    It uses the same spirit as the live model: support proximity, RSI recovery,
    MACD improvement, Bollinger reclaim, volume, and drawdown context.
    """
    if i < 260:
        return np.nan

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    window = df.iloc[i - 252:i + 1]

    close = row["Close"]
    atr = row["ATR14"]
    if pd.isna(atr) or atr <= 0:
        atr = close * 0.02

    score = 0.0

    recent_low = window["Low"].rolling(20).min().iloc[-1]
    support_distance = abs(close - recent_low)
    if support_distance <= atr:
        score += 20
    elif support_distance <= 2 * atr:
        score += 12

    if row["RSI14"] < 35 and row["RSI14"] > prev["RSI14"]:
        score += 15
    elif row["RSI14"] < 40 and row["RSI14"] > prev["RSI14"]:
        score += 10

    if row["MACD_Hist"] > prev["MACD_Hist"]:
        score += 15

    if row["Close"] > row["BB_Lower"] and prev["Close"] < prev["BB_Lower"]:
        score += 10
    elif row["Close"] > row["BB_Lower"] and row["RSI14"] < 45:
        score += 5

    if row["Volume_Ratio"] >= 1.8 and row["Close"] >= row["Open"]:
        score += 15
    elif row["Volume_Ratio"] >= 1.5:
        score += 10

    if row["Close"] > row["SMA20"] and prev["Close"] <= prev["SMA20"]:
        score += 10
    elif row["Close"] > row["SMA20"]:
        score += 6

    deeply_corrected = row["Drawdown_52W"] <= -0.25
    near_low = close <= row["Rolling_52W_Low"] * 1.12
    if deeply_corrected and near_low:
        score += 10
    elif deeply_corrected:
        score += 5

    return min(score, 100)


def run_backtest(
    df: pd.DataFrame,
    signal_threshold: float = 60,
    forward_days: int = 63,
    success_return: float = 0.05,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """Backtest whether historical bottom signals led to positive forward returns.

    Args:
        df: Daily dataframe with indicators.
        signal_threshold: Proxy score needed to trigger a historical signal.
        forward_days: Forward holding period, e.g. 21 = 1 month, 63 = 3 months.
        success_return: Return threshold counted as a successful bottom.
    """
    required_cols = [
        "Open", "High", "Low", "Close", "Volume", "SMA20", "RSI14", "MACD_Hist",
        "ATR14", "BB_Lower", "Volume_Ratio", "Rolling_52W_Low", "Drawdown_52W",
    ]
    x = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()
    records = []

    if len(x) < 320 + forward_days:
        return BacktestResult(0, 0, 0, 0, 0, 0), pd.DataFrame()

    # Avoid repeated adjacent signals by enforcing a cooldown period.
    last_signal_i = -999
    cooldown = max(10, forward_days // 3)

    for i in range(260, len(x) - forward_days):
        if i - last_signal_i < cooldown:
            continue

        proxy_score = historical_signal_proxy_score(x, i)
        if pd.isna(proxy_score) or proxy_score < signal_threshold:
            continue

        entry_price = float(x["Close"].iloc[i])
        exit_price = float(x["Close"].iloc[i + forward_days])
        forward_window = x["Close"].iloc[i:i + forward_days + 1]
        fwd_return = exit_price / entry_price - 1
        mdd = max_drawdown_during_period(forward_window)
        success = fwd_return >= success_return

        records.append(
            {
                "Date": x.index[i],
                "Entry": entry_price,
                "Exit": exit_price,
                "Proxy_Score": proxy_score,
                "Forward_Return": fwd_return,
                "Max_Drawdown": mdd,
                "Success": success,
            }
        )
        last_signal_i = i

    trades = pd.DataFrame(records)
    if trades.empty:
        return BacktestResult(0, 0, 0, 0, 0, -10), trades

    hit_rate = float(trades["Success"].mean())
    avg_forward_return = float(trades["Forward_Return"].mean())
    median_forward_return = float(trades["Forward_Return"].median())
    avg_max_drawdown = float(trades["Max_Drawdown"].mean())
    sample_size = int(len(trades))

    # Confidence adjustment based on realized historical signal quality.
    # This is deliberately conservative.
    if sample_size < 5:
        confidence_adjustment = -5
    elif hit_rate >= 0.65 and avg_forward_return > 0:
        confidence_adjustment = 10
    elif hit_rate >= 0.55 and avg_forward_return > 0:
        confidence_adjustment = 5
    elif hit_rate < 0.45 or avg_forward_return < 0:
        confidence_adjustment = -10
    else:
        confidence_adjustment = 0

    result = BacktestResult(
        hit_rate=hit_rate,
        avg_forward_return=avg_forward_return,
        median_forward_return=median_forward_return,
        sample_size=sample_size,
        avg_max_drawdown=avg_max_drawdown,
        confidence_adjustment=float(confidence_adjustment),
    )

    return result, trades


# -----------------------------------------------------------------------------
# Explanation layer
# -----------------------------------------------------------------------------

def verdict_from_score(score: float) -> str:
    if score >= 80:
        return "Strong technical bottom setup"
    if score >= 65:
        return "Possible technical bottom forming"
    if score >= 50:
        return "Watchlist zone; confirmation still needed"
    return "Weak bottom setup"


def generate_explanation(
    ticker: str,
    final_score: float,
    base_score: float,
    backtest: BacktestResult,
    signals: List[SignalResult],
    context: Dict[str, object],
) -> str:
    """Generate a deterministic AI-style explanation.

    You can replace this with an LLM call later by sending the same structured
    data to your model.
    """
    passed = [s for s in signals if s.points > 0]
    failed = [s for s in signals if s.points == 0]

    strongest = sorted(passed, key=lambda x: x.points / max(x.max_points, 1), reverse=True)[:4]
    weakest = failed[:3]

    bottom_lower = context["bottom_zone_lower"]
    bottom_upper = context["bottom_zone_upper"]
    invalidation = context["invalidation"]
    current_price = context["current_price"]

    text = []
    text.append(f"### AI Technical View for {ticker.upper()}")
    text.append("")
    text.append(
        f"The model classifies this as **{verdict_from_score(final_score)}**. "
        f"The base technical score is **{base_score:.1f}/100**, and the backtest-adjusted confidence score is "
        f"**{final_score:.1f}/100**."
    )
    text.append("")
    text.append(
        f"The estimated technical bottom zone is **{bottom_lower:,.2f} to {bottom_upper:,.2f}**. "
        f"Current price is **{current_price:,.2f}**. A clean breakdown below **{invalidation:,.2f}** would weaken the setup."
    )
    text.append("")

    if strongest:
        text.append("**Main supporting signals:**")
        for s in strongest:
            text.append(f"- {s.explanation}")
        text.append("")

    if weakest:
        text.append("**Main missing confirmations:**")
        for s in weakest:
            text.append(f"- {s.explanation}")
        text.append("")

    if backtest.sample_size > 0:
        text.append(
            f"Historically, similar proxy signals occurred **{backtest.sample_size}** times in the tested period. "
            f"The hit rate was **{backtest.hit_rate:.1%}**, with an average forward return of "
            f"**{backtest.avg_forward_return:.1%}** over the selected forward window."
        )
    else:
        text.append(
            "The historical sample size was too small for a reliable backtest-based adjustment. "
            "The confidence score should therefore be treated cautiously."
        )

    text.append("")
    text.append(
        "This is a technical-analysis framework, not a prediction engine. "
        "The bottom zone should be used as a risk-management area, not as a guaranteed reversal point."
    )

    return "\n".join(text)


# -----------------------------------------------------------------------------
# UI interpretation helpers
# -----------------------------------------------------------------------------

def signal_status_label(signal: SignalResult) -> str:
    """Translate numeric score into bottom-model language.

    A zero score does not mean the stock is bad. It means this indicator is not
    currently evidence of a bottoming setup.
    """
    if signal.max_points == 0:
        return "Excluded"
    ratio = signal.points / max(signal.max_points, 1)
    if ratio >= 0.70:
        return "Bottom signal active"
    if ratio > 0:
        return "Partial / mixed"
    return "Not a bottom signal"


def signal_card_class(signal: SignalResult) -> str:
    label = signal_status_label(signal)
    if label == "Supportive":
        return "signal-pass"
    if label == "Mixed":
        return "signal-mixed"
    return "signal-fail"


def build_indicator_verdict_df(signals: List[SignalResult]) -> pd.DataFrame:
    rows = []
    for s in signals:
        ratio = s.points / max(s.max_points, 1) if s.max_points else 0
        if ratio >= 0.70:
            model_read = "Supports bottom thesis"
        elif ratio > 0:
            model_read = "Partial / early evidence"
        else:
            model_read = "No bottom evidence now"

        rows.append(
            {
                "Indicator": s.name,
                "Bottom-model read": model_read,
                "Score": f"{s.points:.1f} / {s.max_points:.0f}" if s.max_points else "Excluded",
                "What it says for this stock": s.explanation,
            }
        )
    return pd.DataFrame(rows)


def render_indicator_cards(signals: List[SignalResult]) -> None:
    """Render a compact indicator readout without overwhelming the page."""
    supportive = [s for s in signals if s.max_points and s.points / max(s.max_points, 1) >= 0.70]
    mixed = [s for s in signals if s.max_points and 0 < s.points / max(s.max_points, 1) < 0.70]
    inactive = [s for s in signals if s.max_points and s.points == 0]

    st.markdown(
        f"""
        <div class="explain-box">
            <b>How to read this section:</b> a score of <b>0</b> means the indicator is
            <b>not currently a bottom signal</b>. It does not automatically mean the stock is weak.
            For example, high RSI can be good momentum, but it is not an oversold-bottom signal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Bottom signals active", len(supportive))
    c2.metric("Partial / mixed signals", len(mixed))
    c3.metric("No bottom signal now", len(inactive))

    top_supportive = sorted(supportive, key=lambda x: x.points / max(x.max_points, 1), reverse=True)[:4]
    top_mixed = sorted(mixed, key=lambda x: x.points / max(x.max_points, 1), reverse=True)[:4]
    top_inactive = inactive[:4]

    groups = [
        ("Active bottom evidence", top_supportive, "pill-green"),
        ("Partial / early evidence", top_mixed, "pill-amber"),
        ("Key missing bottom evidence", top_inactive, "pill-slate"),
    ]

    cols = st.columns(3)
    for col, (title, group, pill_class) in zip(cols, groups):
        with col:
            st.markdown(f"#### {title}")
            if not group:
                st.markdown("<div class='small-muted'>None</div>", unsafe_allow_html=True)
            for s in group:
                status = signal_status_label(s)
                card_class = signal_card_class(s)
                st.markdown(
                    f"""
                    <div class="signal-card {card_class}">
                        <div class="micro-muted">{status}</div>
                        <div style="font-size:1.02rem;font-weight:750;margin-top:4px;">{s.name}</div>
                        <div style="margin:7px 0 8px 0;"><span class="pill {pill_class}">{s.points:.1f} / {s.max_points:.0f}</span></div>
                        <div style="line-height:1.45;color:#cbd5e1;">{s.explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("View every indicator read", expanded=False):
        st.dataframe(build_indicator_verdict_df(signals), use_container_width=True, hide_index=True)


def classify_setup_type(df: pd.DataFrame, signals: List[SignalResult], final_score: float) -> Tuple[str, str]:
    """Classify whether the stock is a bottom setup, trend continuation, or no setup."""
    latest = df.dropna(subset=["Close", "SMA20", "SMA50", "SMA200", "RSI14", "Drawdown_52W"]).iloc[-1]
    close = float(latest["Close"])
    rsi = float(latest["RSI14"])
    drawdown = float(latest["Drawdown_52W"])
    above_20 = close > float(latest["SMA20"])
    above_50 = close > float(latest["SMA50"])
    above_200 = close > float(latest["SMA200"])

    active_bottom_signals = sum(
        1 for s in signals if s.max_points and s.points / max(s.max_points, 1) >= 0.70
    )

    if final_score >= 75:
        return (
            "Strong bottom setup",
            "Multiple bottom-specific signals are active. The support zone has stronger confirmation and the setup quality is high.",
        )
    if final_score >= 60:
        return (
            "Possible bottom setup",
            "The stock has enough bottoming evidence to monitor closely, but follow-through and risk management remain important.",
        )
    if drawdown > -0.12 and above_20 and above_50 and above_200 and rsi >= 50:
        return (
            "Trend continuation / not a bottom setup",
            "The stock appears closer to an ongoing uptrend than a bottoming situation. Many bottom indicators can show zero here because the stock is not deeply corrected or oversold.",
        )
    if rsi >= 70 and above_20 and above_50:
        return (
            "Momentum / potentially extended",
            "The stock has strong momentum rather than a bottoming profile. This can be bullish, but it is not the same as a technical-bottom setup.",
        )
    if active_bottom_signals <= 2 and final_score < 40:
        return (
            "No actionable bottom setup yet",
            "The model does not see enough bottom-specific confirmation. The support zone may exist, but reversal evidence is weak.",
        )
    return (
        "Mixed technical setup",
        "Some signals are constructive, but the evidence is not strong enough to classify the stock as a high-confidence bottom setup.",
    )


def render_final_verdict_card(
    ticker: str,
    final_score: float,
    base_score: float,
    lower: float,
    upper: float,
    invalidation: float,
    latest_close: float,
    backtest: BacktestResult,
    setup_type: str,
    setup_summary: str,
) -> None:
    verdict = verdict_from_score(final_score)
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="micro-muted">Technical Bottom Assessment</div>
            <h2 style="margin: 6px 0 8px 0;">{ticker.upper()} — {verdict}</h2>
            <div style="margin: 12px 0 4px 0;">
                <span class="pill pill-slate">Setup type: {setup_type}</span>
            </div>
            <div class="small-muted" style="line-height:1.55;max-width:980px;">{setup_summary}</div>
            <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:22px;">
                <div><div class="small-muted">Probable technical bottom zone</div><div class="big-level">{lower:,.2f} – {upper:,.2f}</div></div>
                <div><div class="small-muted">Invalidation below</div><div class="big-level">{invalidation:,.2f}</div></div>
                <div><div class="small-muted">Current price</div><div class="big-level">{latest_close:,.2f}</div></div>
                <div><div class="small-muted">Final confidence</div><div class="big-level">{final_score:.1f}/100</div></div>
            </div>
            <div style="margin-top:18px;line-height:1.55;color:#cbd5e1;">
                Base technical score: <b>{base_score:.1f}/100</b> · Backtest adjustment:
                <b>{backtest.confidence_adjustment:+.0f}</b> · Historical proxy signals: <b>{backtest.sample_size}</b> ·
                Hit rate: <b>{backtest.hit_rate:.1%}</b> · Average forward return: <b>{backtest.avg_forward_return:.1%}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_calculation_breakdown(
    ticker: str,
    df: pd.DataFrame,
    context: Dict[str, object],
    signals: List[SignalResult],
    backtest: BacktestResult,
    base_score: float,
    final_score: float,
) -> None:
    """Show transparent calculation details for the bottom zone and confidence score."""
    latest = df.iloc[-1]
    current_price = float(context.get("current_price", latest["Close"]))
    atr = float(context.get("atr", latest.get("ATR14", np.nan)))
    bottom_lower = float(context.get("bottom_zone_lower", np.nan))
    bottom_upper = float(context.get("bottom_zone_upper", np.nan))
    invalidation = float(context.get("invalidation", np.nan))
    raw_score = float(context.get("raw_score", np.nan))
    max_score = float(context.get("max_score", np.nan))

    support_zone = context.get("support_zone")
    vp_zone = context.get("volume_profile_zone")
    support_components = context.get("support_components", [])
    selected_support_cluster = context.get("selected_support_cluster", {})
    blended_support = float(context.get("blended_support", np.nan))
    zone_atr_multiplier = float(context.get("zone_atr_multiplier", 1.0))
    invalidation_atr_multiplier = float(context.get("invalidation_atr_multiplier", 1.5))

    if math.isnan(blended_support):
        blended_support = float(df["Low"].tail(60).min())

    st.markdown("## Calculation transparency")
    st.caption("This section shows the exact mechanics used by the model. It is designed so the user can audit the bottom-zone and confidence-score calculation.")

    st.markdown("### 1. Bottom-zone calculation")
    st.markdown(
        """
        The model does **not** estimate a single bottom price. It estimates a **technical support zone**.

        The current formula is:

        ```text
        1. Build multiple support references
           - swing-low clusters
           - fixed-range volume-profile nodes
           - rolling lows
           - moving averages
           - yearly anchored VWAP
           - 2-year regression channel levels
           - 200-week SMA
           - Fibonacci retracement levels

        2. Cluster nearby support references

        3. Select the best support cluster based on:
           - total weighted confluence
           - proximity to current price
           - number of independent source types

        4. Calculate the zone:
           Bottom zone lower = Blended support - ATR(14) × zone multiplier
           Bottom zone upper = Blended support + ATR(14) × zone multiplier
           Invalidation level = Blended support - ATR(14) × invalidation multiplier
        ```
        """
    )

    if support_components:
        support_df = pd.DataFrame(support_components)
        display_cols = ["name", "category", "level", "base_weight", "effective_weight", "distance_atr", "reason"]
        support_df = support_df[[c for c in display_cols if c in support_df.columns]].copy()
        if "level" in support_df.columns:
            support_df["level"] = support_df["level"].map(lambda x: f"{x:,.2f}")
        if "base_weight" in support_df.columns:
            support_df["base_weight"] = support_df["base_weight"].map(lambda x: f"{x:.2f}")
        if "effective_weight" in support_df.columns:
            support_df["effective_weight"] = support_df["effective_weight"].map(lambda x: f"{x:.2f}")
        if "distance_atr" in support_df.columns:
            support_df["distance_atr"] = support_df["distance_atr"].map(lambda x: f"{x:.2f}x")
        st.dataframe(support_df, use_container_width=True, hide_index=True)
    else:
        st.info("No support components were available; the model used a fallback recent low.")

    calc_rows = [
        {"Step": "Current price", "Formula / source": "Latest adjusted close", "Value": f"{current_price:,.2f}"},
        {"Step": "ATR(14)", "Formula / source": "14-period Average True Range", "Value": f"{atr:,.2f}"},
        {"Step": "Blended support", "Formula / source": "Average of support references above", "Value": f"{blended_support:,.2f}"},
        {"Step": "Selected support cluster score", "Formula / source": "Weighted confluence × proximity × source-diversity bonus", "Value": f"{context.get('support_confluence_score', np.nan):,.2f}"},
        {"Step": "Zone ATR multiplier", "Formula / source": "Dynamic multiplier based on ATR/HV percentile and confluence quality", "Value": f"{zone_atr_multiplier:.2f}x"},
        {"Step": "Invalidation ATR multiplier", "Formula / source": "Dynamic multiplier based on volatility regime", "Value": f"{invalidation_atr_multiplier:.2f}x"},
        {"Step": "Bottom zone lower", "Formula / source": f"{blended_support:,.2f} - {zone_atr_multiplier:.2f} × {atr:,.2f}", "Value": f"{bottom_lower:,.2f}"},
        {"Step": "Bottom zone upper", "Formula / source": f"{blended_support:,.2f} + {zone_atr_multiplier:.2f} × {atr:,.2f}", "Value": f"{bottom_upper:,.2f}"},
        {"Step": "Invalidation level", "Formula / source": f"{blended_support:,.2f} - {invalidation_atr_multiplier:.2f} × {atr:,.2f}", "Value": f"{invalidation:,.2f}"},
    ]
    st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)

    st.markdown("### 2. Selected support cluster")
    st.markdown(
        "The model clusters all nearby support references and chooses the cluster with the best weighted confluence. "
        "The selected cluster is the basis for the blended support level."
    )

    if selected_support_cluster:
        selected_summary = pd.DataFrame(
            [
                {"Metric": "Selected cluster center", "Value": f"{selected_support_cluster.get('center', np.nan):,.2f}"},
                {"Metric": "Cluster lower reference", "Value": f"{selected_support_cluster.get('lower', np.nan):,.2f}"},
                {"Metric": "Cluster upper reference", "Value": f"{selected_support_cluster.get('upper', np.nan):,.2f}"},
                {"Metric": "Total effective weight", "Value": f"{selected_support_cluster.get('total_weight', np.nan):,.2f}"},
                {"Metric": "Independent source types", "Value": f"{selected_support_cluster.get('source_count', 0)}"},
                {"Metric": "Distance from current price", "Value": f"{selected_support_cluster.get('distance_atr', np.nan):,.2f} ATR"},
                {"Metric": "Final cluster score", "Value": f"{selected_support_cluster.get('cluster_score', np.nan):,.2f}"},
            ]
        )
        st.dataframe(selected_summary, use_container_width=True, hide_index=True)

        members = selected_support_cluster.get("members", [])
        if members:
            members_df = pd.DataFrame(members)
            member_cols = ["name", "category", "level", "effective_weight", "distance_atr", "reason"]
            members_df = members_df[[c for c in member_cols if c in members_df.columns]].copy()
            if "level" in members_df.columns:
                members_df["level"] = members_df["level"].map(lambda x: f"{x:,.2f}")
            if "effective_weight" in members_df.columns:
                members_df["effective_weight"] = members_df["effective_weight"].map(lambda x: f"{x:.2f}")
            if "distance_atr" in members_df.columns:
                members_df["distance_atr"] = members_df["distance_atr"].map(lambda x: f"{x:.2f}x")
            st.dataframe(members_df, use_container_width=True, hide_index=True)
    else:
        st.info("No selected support cluster was available.")

    st.markdown("### 3. Base technical score calculation")
    st.markdown(
        """
        Each indicator contributes points. The base score is normalized to 100:

        ```text
        Base technical score = Sum of points earned ÷ Sum of max possible points × 100
        ```
        """
    )

    score_rows = []
    for s in signals:
        contribution = s.points / max_score * 100 if max_score and not np.isnan(max_score) else np.nan
        score_rows.append(
            {
                "Indicator": s.name,
                "Points earned": f"{s.points:.1f}",
                "Max points": f"{s.max_points:.0f}",
                "Contribution to 100-pt score": f"{contribution:.1f}" if not np.isnan(contribution) else "-",
                "Interpretation": s.explanation,
            }
        )

    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    score_summary = pd.DataFrame(
        [
            {"Metric": "Raw points earned", "Value": f"{raw_score:.1f}"},
            {"Metric": "Maximum possible points", "Value": f"{max_score:.1f}"},
            {"Metric": "Base technical score", "Value": f"{base_score:.1f}/100"},
        ]
    )
    st.dataframe(score_summary, use_container_width=True, hide_index=True)

    st.markdown("### 4. Backtest adjustment and final confidence")
    st.markdown(
        """
        After calculating the live technical score, the model applies a historical backtest adjustment:

        ```text
        Final confidence = Base technical score + Backtest adjustment
        Final confidence is clipped between 0 and 100
        ```

        The backtest adjustment is based on how similar historical proxy signals performed.
        """
    )

    backtest_rows = [
        {"Metric": "Historical proxy signals", "Value": f"{backtest.sample_size}"},
        {"Metric": "Hit rate", "Value": f"{backtest.hit_rate:.1%}"},
        {"Metric": "Average forward return", "Value": f"{backtest.avg_forward_return:.1%}"},
        {"Metric": "Median forward return", "Value": f"{backtest.median_forward_return:.1%}"},
        {"Metric": "Average max drawdown", "Value": f"{backtest.avg_max_drawdown:.1%}"},
        {"Metric": "Backtest adjustment", "Value": f"{backtest.confidence_adjustment:+.0f}"},
        {"Metric": "Final confidence", "Value": f"{final_score:.1f}/100"},
    ]
    st.dataframe(pd.DataFrame(backtest_rows), use_container_width=True, hide_index=True)

    st.markdown("### 5. Latest raw indicator values")
    latest_rows = [
        {"Metric": "Close", "Value": latest.get("Close", np.nan)},
        {"Metric": "RSI14", "Value": latest.get("RSI14", np.nan)},
        {"Metric": "MACD", "Value": latest.get("MACD", np.nan)},
        {"Metric": "MACD Signal", "Value": latest.get("MACD_Signal", np.nan)},
        {"Metric": "MACD Histogram", "Value": latest.get("MACD_Hist", np.nan)},
        {"Metric": "ATR14", "Value": latest.get("ATR14", np.nan)},
        {"Metric": "ATR%", "Value": latest.get("ATR_Pct", np.nan)},
        {"Metric": "ATR% Percentile", "Value": latest.get("ATR_Pctile_252", np.nan)},
        {"Metric": "HV20", "Value": latest.get("HV20", np.nan)},
        {"Metric": "HV20 Percentile", "Value": latest.get("HV20_Pctile_252", np.nan)},
        {"Metric": "Volume Ratio", "Value": latest.get("Volume_Ratio", np.nan)},
        {"Metric": "20-DMA", "Value": latest.get("SMA20", np.nan)},
        {"Metric": "50-DMA", "Value": latest.get("SMA50", np.nan)},
        {"Metric": "200-DMA", "Value": latest.get("SMA200", np.nan)},
        {"Metric": "Yearly Anchored VWAP", "Value": latest.get("Yearly_AVWAP", np.nan)},
        {"Metric": "2Y LRC Lower", "Value": latest.get("LRC_Lower_2Y", np.nan)},
        {"Metric": "2Y LRC Mid", "Value": latest.get("LRC_Mid_2Y", np.nan)},
        {"Metric": "2Y LRC Upper", "Value": latest.get("LRC_Upper_2Y", np.nan)},
        {"Metric": "OBV", "Value": latest.get("OBV", np.nan)},
    ]
    latest_df = pd.DataFrame(latest_rows)
    latest_df["Value"] = latest_df["Value"].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "-")
    st.dataframe(latest_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_price_chart(
    df: pd.DataFrame,
    context: Dict[str, object],
    overlays: Optional[List[str]] = None,
    title: str = "Price chart",
) -> go.Figure:
    """Main price chart with user-selected overlays.

    The default chart is intentionally clean. Heavy overlays such as the
    regression channel, VWAP, support bands, and Bollinger Bands are shown only
    when selected by the user.
    """
    x = df.copy()
    selected = set(overlays or [])

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x.index,
            open=x["Open"],
            high=x["High"],
            low=x["Low"],
            close=x["Close"],
            name="Price",
        )
    )

    overlay_map = {
        "SMA20": ("SMA20", "20-DMA"),
        "SMA50": ("SMA50", "50-DMA"),
        "SMA200": ("SMA200", "200-DMA"),
        "Yearly anchored VWAP": ("Yearly_AVWAP", "Yearly AVWAP"),
        "2Y regression midline": ("LRC_Mid_2Y", "2Y LRC Mid"),
        "2Y regression upper band": ("LRC_Upper_2Y", "2Y LRC Upper"),
        "2Y regression lower band": ("LRC_Lower_2Y", "2Y LRC Lower"),
        "Bollinger upper band": ("BB_Upper", "Bollinger Upper"),
        "Bollinger lower band": ("BB_Lower", "Bollinger Lower"),
    }

    for option, (col, name) in overlay_map.items():
        if option in selected and col in x.columns and x[col].notna().any():
            fig.add_trace(go.Scatter(x=x.index, y=x[col], mode="lines", name=name))

    if "Estimated bottom zone" in selected:
        lower = context.get("bottom_zone_lower")
        upper = context.get("bottom_zone_upper")
        if lower is not None and upper is not None:
            fig.add_hrect(
                y0=lower,
                y1=upper,
                line_width=0,
                fillcolor="LightGreen",
                opacity=0.18,
                annotation_text="Estimated bottom zone",
                annotation_position="top left",
            )

    if "Invalidation level" in selected and context.get("invalidation") is not None:
        fig.add_hline(
            y=context["invalidation"],
            line_dash="dash",
            annotation_text="Invalidation",
            annotation_position="bottom left",
        )

    if "Clustered support" in selected:
        support_zone = context.get("support_zone")
        if support_zone:
            fig.add_hline(
                y=support_zone.center,
                line_dash="dot",
                annotation_text=f"Clustered support {support_zone.center:.2f}",
                annotation_position="bottom right",
            )

    if "Volume-profile support" in selected:
        vp_zone = context.get("volume_profile_zone")
        if vp_zone:
            fig.add_hline(
                y=vp_zone.center,
                line_dash="dot",
                annotation_text=f"Volume support {vp_zone.center:.2f}",
                annotation_position="top right",
            )

    if "Nearest resistance" in selected:
        resistance_zone = context.get("resistance_zone")
        if resistance_zone:
            fig.add_hline(
                y=resistance_zone.center,
                line_dash="dot",
                annotation_text=f"Resistance {resistance_zone.center:.2f}",
                annotation_position="top right",
            )

    fig.update_layout(
        title=title,
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_rsi(df: pd.DataFrame, title: str = "Daily RSI 14") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI14"))
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig.update_layout(
        title=title,
        height=320,
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_macd(df: pd.DataFrame, title: str = "Daily MACD") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram"))
    fig.update_layout(
        title=title,
        height=340,
        yaxis_title="MACD",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_backtest_trades(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(height=280, title="No historical signals found")
        return fig

    fig.add_trace(
        go.Bar(
            x=trades["Date"],
            y=trades["Forward_Return"],
            name="Forward Return",
        )
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        height=320,
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_price_with_200w(df: pd.DataFrame) -> go.Figure:
    weekly = resample_ohlcv(df, "W-FRI")
    weekly["SMA200W"] = weekly["Close"].rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["Close"], mode="lines", name="Weekly Close"))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["SMA200W"], mode="lines", name="200-Week SMA"))
    fig.update_layout(height=360, title="200-Week SMA", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_avwap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Yearly_AVWAP"], mode="lines", name="Yearly Anchored VWAP"))
    fig.update_layout(height=360, title="Yearly Anchored VWAP", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_lrc(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    for col, name in [("LRC_Upper_2Y", "2Y LRC Upper"), ("LRC_Mid_2Y", "2Y LRC Mid"), ("LRC_Lower_2Y", "2Y LRC Lower")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name))
    fig.update_layout(height=380, title="2-Year Linear Regression Channel", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_volume_profile(df: pd.DataFrame, lookback: int = 252, bins: int = 40) -> go.Figure:
    recent = df.tail(lookback).copy()
    fig = go.Figure()
    if recent.empty:
        fig.update_layout(height=360, title="Fixed Range Volume Profile")
        return fig

    price_bins = np.linspace(float(recent["Low"].min()), float(recent["High"].max()), bins + 1)
    bucket = pd.cut(recent["Close"], price_bins, include_lowest=True)
    volume_by_bucket = recent.groupby(bucket, observed=False)["Volume"].sum()
    centers = [(interval.left + interval.right) / 2 for interval in volume_by_bucket.index]
    fig.add_trace(go.Bar(x=volume_by_bucket.values, y=centers, orientation="h", name="Volume by Price"))
    fig.update_layout(height=430, title="Fixed Range Volume Profile", xaxis_title="Volume", yaxis_title="Price", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_obv(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], mode="lines", name="OBV"))
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV_EMA20"], mode="lines", name="OBV EMA20"))
    fig.update_layout(height=340, title="On-Balance Volume Trend", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_hv_percentile(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["HV20_Pctile_252"], mode="lines", name="HV20 Percentile"))
    fig.add_hline(y=85, line_dash="dash", annotation_text="Extreme volatility")
    fig.add_hline(y=50, line_dash="dash", annotation_text="Median")
    fig.update_layout(height=330, title="Historical Volatility Percentile", yaxis_range=[0, 100], margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_atr_compression(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR_Pct"], mode="lines", name="ATR%"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR_Pct_SMA20"], mode="lines", name="ATR% SMA20"))
    fig.update_layout(height=330, title="ATR Compression", yaxis_tickformat=".1%", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_coppock_curve(df: pd.DataFrame) -> go.Figure:
    monthly = resample_ohlcv(df, "ME")
    close = monthly["Close"]
    coppock = weighted_moving_average(close.pct_change(14) * 100 + close.pct_change(11) * 100, 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=coppock, mode="lines", name="Coppock Curve"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=330, title="Coppock Curve", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_weekly_macd(df: pd.DataFrame) -> go.Figure:
    weekly = add_indicators(resample_ohlcv(df, "W-FRI"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["MACD"], mode="lines", name="Weekly MACD"))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["MACD_Signal"], mode="lines", name="Weekly Signal"))
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["MACD_Hist"], name="Weekly Histogram"))
    fig.update_layout(height=350, title="Weekly MACD", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_weekly_rsi(df: pd.DataFrame) -> go.Figure:
    weekly = add_indicators(resample_ohlcv(df, "W-FRI"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["RSI14"], mode="lines", name="Weekly RSI14"))
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig.update_layout(height=330, title="Weekly RSI / Divergence Context", yaxis_range=[0, 100], margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_stage_analysis(df: pd.DataFrame) -> go.Figure:
    weekly = resample_ohlcv(df, "W-FRI")
    weekly["SMA10W"] = weekly["Close"].rolling(10).mean()
    weekly["SMA30W"] = weekly["Close"].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["Close"], mode="lines", name="Weekly Close"))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["SMA10W"], mode="lines", name="10-Week SMA"))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["SMA30W"], mode="lines", name="30-Week SMA"))
    fig.update_layout(height=360, title="Stan Weinstein Stage Analysis", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_indicator_chart_grid(
    df: pd.DataFrame,
    chart_lookback: int,
    selected_charts: Optional[List[str]] = None,
) -> None:
    """Render selected indicator charts in a clean two-column grid."""
    x = df.tail(chart_lookback).copy()
    empty_context = {
        "bottom_zone_lower": None,
        "bottom_zone_upper": None,
        "invalidation": None,
        "support_zone": None,
        "volume_profile_zone": None,
        "resistance_zone": None,
    }

    chart_builders = {
        "Price + key moving averages": lambda: plot_price_chart(
            x,
            empty_context,
            overlays=["SMA20", "SMA50", "SMA200"],
            title="Price with key moving averages",
        ),
        "200-week SMA": lambda: plot_price_with_200w(df),
        "Yearly anchored VWAP": lambda: plot_avwap(x),
        "2-year linear regression channel": lambda: plot_lrc(x),
        "Daily RSI": lambda: plot_rsi(x, title="Daily RSI 14"),
        "Weekly RSI / divergence context": lambda: plot_weekly_rsi(df),
        "Daily MACD": lambda: plot_macd(x, title="Daily MACD"),
        "Weekly MACD": lambda: plot_weekly_macd(df),
        "Coppock Curve": lambda: plot_coppock_curve(df),
        "Fixed range volume profile": lambda: plot_volume_profile(df, lookback=min(chart_lookback, 504)),
        "Stan Weinstein Stage Analysis": lambda: plot_stage_analysis(df),
        "OBV trend": lambda: plot_obv(x),
        "Historical volatility percentile": lambda: plot_hv_percentile(x),
        "ATR compression": lambda: plot_atr_compression(x),
    }

    chosen = selected_charts or list(chart_builders.keys())
    chosen = [name for name in chosen if name in chart_builders]

    if not chosen:
        st.info("Select at least one chart to display.")
        return

    for i in range(0, len(chosen), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(chosen):
                continue
            chart_name = chosen[idx]
            with col:
                try:
                    st.plotly_chart(chart_builders[chart_name](), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not render {chart_name}: {exc}")


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def score_badge(score: float) -> str:
    if score >= 80:
        return "🟢 Strong"
    if score >= 65:
        return "🟡 Possible"
    if score >= 50:
        return "🟠 Watchlist"
    return "🔴 Weak"


def display_signal_table(signals: List[SignalResult]) -> None:
    rows = []
    for s in signals:
        rows.append(
            {
                "Signal": s.name,
                "Points": f"{s.points:.1f} / {s.max_points:.0f}",
                "Passed": "Yes" if s.passed else "No",
                "Explanation": s.explanation,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def display_mtf_states(states: Dict[str, Dict[str, object]]) -> None:
    rows = []
    for tf, state in states.items():
        rows.append(
            {
                "Timeframe": tf,
                "Valid": state.get("valid", False),
                "Score": f"{state.get('score', 0)} / 4",
                "RSI": f"{state.get('rsi', np.nan):.1f}" if state.get("valid", False) else "-",
                "Summary": state.get("summary", "-"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------

def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.title("AI Technical Bottom Finder")
    st.caption("A scrollable technical-bottom dashboard with indicator-by-indicator verdicts, support-zone estimation, confirmation quality, and dedicated charts.")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="AAPL", help="Use .NS for NSE stocks, e.g. RELIANCE.NS")
        period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=1)
        chart_lookback = st.slider("Chart lookback days", 120, 1000, 365, step=30)

        st.header("Main chart display")
        main_chart_overlays = st.multiselect(
            "Choose overlays for the main price chart",
            options=[
                "Estimated bottom zone",
                "Invalidation level",
                "Clustered support",
                "Volume-profile support",
                "Nearest resistance",
                "SMA20",
                "SMA50",
                "SMA200",
                "Yearly anchored VWAP",
                "2Y regression midline",
                "2Y regression upper band",
                "2Y regression lower band",
                "Bollinger upper band",
                "Bollinger lower band",
            ],
            default=["Estimated bottom zone", "Invalidation level", "Clustered support", "Volume-profile support", "SMA20", "SMA50"],
            help="Keep fewer overlays selected for a cleaner chart. Add more when you want to inspect a specific indicator.",
        )

        st.header("Backtest settings")
        signal_threshold = st.slider("Historical signal threshold", 40, 85, 60, step=5)
        forward_days = st.selectbox("Forward return window", [21, 42, 63, 126], index=2, format_func=lambda x: f"{x} trading days")
        success_return = st.slider("Success return threshold", 0.00, 0.25, 0.05, step=0.01, format="%.2f")

        st.header("Optional advanced tools")
        show_advanced_tools = st.checkbox(
            "Show production filters / batch testing",
            value=False,
            help="Keep this off for normal single-stock technical bottom analysis. Turn it on only for liquidity, slippage, market-regime, sector-relative strength, database storage, or index batch testing."
        )

        sector_ticker = ""
        market_ticker = ""
        volatility_ticker = ""
        order_value = 100000.0
        min_avg_volume = 100000.0
        min_avg_traded_value = 5000000.0
        max_atr_pct = 0.12
        save_to_db = False
        constituents_file = None
        max_batch_names = 25

        if show_advanced_tools:
            ticker_upper_for_defaults = ticker.strip().upper()
            default_market = "^NSEI" if ticker_upper_for_defaults.endswith(".NS") else "^GSPC"
            default_vol = "^INDIAVIX" if ticker_upper_for_defaults.endswith(".NS") else "^VIX"

            with st.expander("Advanced production filters", expanded=True):
                sector_ticker = st.text_input("Sector/industry benchmark ticker", value="", help="Optional. Examples: XLK, XLF, ^CNXIT, ^CNXFINANCE if available from Yahoo Finance.")
                market_ticker = st.text_input("Market benchmark ticker", value=default_market)
                volatility_ticker = st.text_input("Volatility benchmark ticker", value=default_vol)
                order_value = st.number_input("Assumed order value for slippage", min_value=0.0, value=100000.0, step=25000.0)
                min_avg_volume = st.number_input("Minimum 20-day average volume", min_value=0.0, value=100000.0, step=50000.0)
                min_avg_traded_value = st.number_input("Minimum 20-day average traded value", min_value=0.0, value=5000000.0, step=1000000.0)
                max_atr_pct = st.slider("Maximum ATR% allowed", 0.02, 0.30, 0.12, step=0.01, format="%.2f")
                save_to_db = st.checkbox("Store signal in local SQLite database", value=False)

            with st.expander("Index testing", expanded=False):
                constituents_file = st.file_uploader(
                    "Upload point-in-time constituents CSV",
                    type=["csv"],
                    help="Recommended columns: Ticker, StartDate, EndDate, Sector. EndDate can be blank for active members.",
                )
                max_batch_names = st.slider("Max constituents for batch run", 5, 100, 25, step=5)

        run_button = st.button("Analyze", type="primary")

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    if not run_button and "last_ticker" not in st.session_state:
        st.info("Enter a ticker and click Analyze.")
        return

    st.session_state["last_ticker"] = ticker

    with st.spinner("Fetching market data and calculating technical signals..."):
        raw = fetch_ohlcv(ticker.strip().upper(), period=period, interval="1d")

    if raw.empty:
        st.error("No market data found. Check the ticker symbol or try a different period.")
        return

    df = add_indicators(raw).copy()
    core_required_cols = [
        "Open", "High", "Low", "Close", "Volume", "SMA20", "RSI14", "MACD_Hist",
        "ATR14", "BB_Lower", "Volume_Ratio", "Rolling_52W_Low", "Drawdown_52W",
    ]
    core_df = df.dropna(subset=[c for c in core_required_cols if c in df.columns]).copy()

    if len(core_df) < 260:
        st.error("Not enough historical data after core indicator calculations. Try a longer history period. For the full model, 5y or 10y is recommended.")
        return

    sector_df = pd.DataFrame()
    market_df = pd.DataFrame()
    volatility_df = pd.DataFrame()

    if sector_ticker.strip():
        sector_df = fetch_ohlcv(sector_ticker.strip().upper(), period=period, interval="1d")

    if market_ticker.strip():
        market_df = fetch_ohlcv(market_ticker.strip().upper(), period=period, interval="1d")

    if volatility_ticker.strip():
        volatility_df = fetch_ohlcv(volatility_ticker.strip().upper(), period=period, interval="1d")

    try:
        base_score, signals, context = calculate_bottom_score(
            df,
            sector_df=sector_df if not sector_df.empty else None,
            market_df=market_df if not market_df.empty else None,
            volatility_df=volatility_df if not volatility_df.empty else None,
            min_avg_volume=min_avg_volume,
            min_avg_traded_value=min_avg_traded_value,
            max_atr_pct=max_atr_pct,
            order_value=order_value,
            include_production_filters=show_advanced_tools,
        )
    except Exception as exc:
        st.error(f"Could not calculate score: {exc}")
        return

    with st.spinner("Running historical backtest..."):
        backtest, trades = run_backtest(
            df,
            signal_threshold=signal_threshold,
            forward_days=forward_days,
            success_return=success_return,
        )

    final_score = float(np.clip(base_score + backtest.confidence_adjustment, 0, 100))

    if save_to_db:
        save_signal_record(
            ticker=ticker.strip().upper(),
            signal_date=df.index[-1],
            latest_close=context["current_price"],
            base_score=base_score,
            final_score=final_score,
            context=context,
            signals=signals,
            backtest=backtest,
        )

    setup_type, setup_summary = classify_setup_type(df, signals, final_score)

    latest_close = context["current_price"]
    lower = context["bottom_zone_lower"]
    upper = context["bottom_zone_upper"]
    invalidation = context["invalidation"]

    main_tab, charts_tab, calculations_tab = st.tabs(["Main Analysis", "All Indicator Charts", "Calculations"])

    with main_tab:
        render_final_verdict_card(
            ticker=ticker,
            final_score=final_score,
            base_score=base_score,
            lower=lower,
            upper=upper,
            invalidation=invalidation,
            latest_close=latest_close,
            backtest=backtest,
            setup_type=setup_type,
            setup_summary=setup_summary,
        )

        st.plotly_chart(
            plot_price_chart(
                df.tail(chart_lookback),
                context,
                overlays=main_chart_overlays,
                title="Main price chart — selected overlays only",
            ),
            use_container_width=True,
        )

        st.markdown("## Final verdict and probable technical bottom support")
        explanation = generate_explanation(
            ticker=ticker,
            final_score=final_score,
            base_score=base_score,
            backtest=backtest,
            signals=signals,
            context=context,
        )
        st.markdown(explanation)

        st.markdown("## What each technical indicator is saying")
        render_indicator_cards(signals)

        st.markdown("## Detected support / resistance zones")
        zone_rows = []
        for key, label in [
            ("support_zone", "Clustered Support"),
            ("volume_profile_zone", "Volume Profile Support"),
            ("resistance_zone", "Nearest Resistance"),
        ]:
            z = context.get(key)
            if z:
                zone_rows.append(
                    {
                        "Zone": label,
                        "Lower": f"{z.lower:,.2f}",
                        "Center": f"{z.center:,.2f}",
                        "Upper": f"{z.upper:,.2f}",
                        "Strength": f"{z.strength:.2f}",
                        "Source": z.source,
                    }
                )
        if zone_rows:
            st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No major support/resistance zones detected.")

        st.markdown("## Multiple-timeframe confirmation")
        display_mtf_states(context["mtf_states"])

        st.markdown("## Backtest summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signals", f"{backtest.sample_size}")
        c2.metric("Hit Rate", f"{backtest.hit_rate:.1%}")
        c3.metric("Avg Forward Return", f"{backtest.avg_forward_return:.1%}")
        c4.metric("Median Forward Return", f"{backtest.median_forward_return:.1%}")
        c5.metric("Confidence Adj.", f"{backtest.confidence_adjustment:+.0f}")

        st.plotly_chart(plot_backtest_trades(trades), use_container_width=True)

        if not trades.empty:
            display_trades = trades.copy()
            display_trades["Date"] = display_trades["Date"].dt.strftime("%Y-%m-%d")
            display_trades["Forward_Return"] = display_trades["Forward_Return"].map(lambda x: f"{x:.1%}")
            display_trades["Max_Drawdown"] = display_trades["Max_Drawdown"].map(lambda x: f"{x:.1%}")
            display_trades["Entry"] = display_trades["Entry"].map(lambda x: f"{x:,.2f}")
            display_trades["Exit"] = display_trades["Exit"].map(lambda x: f"{x:,.2f}")
            with st.expander("Open historical signal trades", expanded=False):
                st.dataframe(display_trades, use_container_width=True, hide_index=True)
        else:
            st.warning("No historical proxy signals found with the selected threshold. Lower the threshold or use a longer history.")

        with st.expander("Index testing and stored signal database", expanded=False):
            st.subheader("Index-level testing")
            st.markdown(
                "Upload a **point-in-time constituent history** to avoid survivorship bias. "
                "The app expects at least a `Ticker` column. For proper survivorship-bias-free testing, include `StartDate` and `EndDate`."
            )

            if constituents_file is not None:
                constituents, note = parse_constituent_history(constituents_file)
                st.info(note)

                if not constituents.empty:
                    active = active_constituents_on(constituents, df.index[-1])
                    st.write(f"Active constituents on {df.index[-1].strftime('%Y-%m-%d')}: **{len(active)}**")
                    st.dataframe(active.head(200), use_container_width=True, hide_index=True)

                    if st.button("Run batch analysis on active constituents", key="run_batch_constituents"):
                        rows = []
                        tickers_to_run = active["Ticker"].dropna().astype(str).str.upper().unique()[:max_batch_names]
                        progress = st.progress(0)

                        for idx, batch_ticker in enumerate(tickers_to_run):
                            try:
                                batch_raw = fetch_ohlcv(batch_ticker, period=period, interval="1d")
                                if batch_raw.empty:
                                    rows.append({"Ticker": batch_ticker, "Status": "No data"})
                                    continue

                                batch_df = add_indicators(batch_raw).copy()
                                batch_core_df = batch_df.dropna(subset=[c for c in core_required_cols if c in batch_df.columns]).copy()
                                if len(batch_core_df) < 260:
                                    rows.append({"Ticker": batch_ticker, "Status": "Insufficient history"})
                                    continue

                                batch_score, batch_signals, batch_context = calculate_bottom_score(
                                    batch_df,
                                    sector_df=None,
                                    market_df=market_df if not market_df.empty else None,
                                    volatility_df=volatility_df if not volatility_df.empty else None,
                                    min_avg_volume=min_avg_volume,
                                    min_avg_traded_value=min_avg_traded_value,
                                    max_atr_pct=max_atr_pct,
                                    order_value=order_value,
                                )
                                batch_backtest, _ = run_backtest(
                                    batch_df,
                                    signal_threshold=signal_threshold,
                                    forward_days=forward_days,
                                    success_return=success_return,
                                )
                                batch_final = float(np.clip(batch_score + batch_backtest.confidence_adjustment, 0, 100))

                                rows.append(
                                    {
                                        "Ticker": batch_ticker,
                                        "Status": "OK",
                                        "Latest Close": batch_context["current_price"],
                                        "Base Score": batch_score,
                                        "Final Score": batch_final,
                                        "Verdict": verdict_from_score(batch_final),
                                        "Bottom Zone Lower": batch_context["bottom_zone_lower"],
                                        "Bottom Zone Upper": batch_context["bottom_zone_upper"],
                                        "Invalidation": batch_context["invalidation"],
                                    }
                                )

                                if save_to_db:
                                    save_signal_record(
                                        ticker=batch_ticker,
                                        signal_date=batch_df.index[-1],
                                        latest_close=batch_context["current_price"],
                                        base_score=batch_score,
                                        final_score=batch_final,
                                        context=batch_context,
                                        signals=batch_signals,
                                        backtest=batch_backtest,
                                    )

                            except Exception as exc:
                                rows.append({"Ticker": batch_ticker, "Status": f"Error: {exc}"})
                            finally:
                                progress.progress((idx + 1) / max(len(tickers_to_run), 1))

                        batch_results = pd.DataFrame(rows)
                        if not batch_results.empty:
                            if "Final Score" in batch_results.columns:
                                batch_results = batch_results.sort_values("Final Score", ascending=False, na_position="last")
                            st.dataframe(batch_results, use_container_width=True, hide_index=True)
                else:
                    st.warning("No usable constituents were found in the uploaded CSV.")
            else:
                st.warning("No constituent-history CSV uploaded. Index-level testing will only use the single ticker above.")

            st.subheader("Stored signal database")
            history = load_signal_history(limit=500)
            if history.empty:
                st.write("No stored signals yet. Run an analysis with database storage enabled.")
            else:
                st.dataframe(history, use_container_width=True, hide_index=True)
                csv = history.to_csv(index=False).encode("utf-8")
                st.download_button("Download stored signals CSV", csv, "technical_bottom_signals.csv", "text/csv")

        with st.expander("Developer notes", expanded=False):
            st.subheader("Developer notes")
            st.markdown(
                """
                **How this model works**

                The app uses a hybrid technical framework:

                - **Support clustering:** Finds recent swing lows and clusters nearby levels using DBSCAN.
                - **RSI divergence:** Checks whether price made a lower swing low while RSI made a higher swing low.
                - **Volume profile:** Finds high-volume price buckets that may act as support.
                - **Candlestick patterns:** Detects hammer, bullish engulfing, piercing pattern, morning star, and bullish outside day.
                - **Multiple timeframe confirmation:** Scores daily, weekly, and monthly trend/momentum alignment.
                - **200-week SMA:** Checks long-cycle support/reclaim on weekly candles.
                - **Yearly anchored VWAP:** Tracks whether price is reclaiming the current year's volume-weighted cost basis.
                - **2-year linear regression channel:** Checks whether price is near/reclaiming the lower long-term regression band.
                - **Weekly RSI divergence:** Detects lower weekly price lows with higher weekly RSI lows.
                - **Coppock Curve:** Uses monthly momentum to detect long-term bottoming turns.
                - **Weekly MACD cross:** Checks for weekly momentum regime improvement.
                - **Stage Analysis:** Applies a Stan Weinstein-style weekly 30-week SMA stage model.
                - **OBV trend break:** Looks for accumulation through On-Balance Volume breakouts.
                - **HV percentile and ATR compression:** Measures whether volatility is cooling/compressing after a decline.
                - **Backtesting:** Finds prior bottom-like signals and measures forward returns.
                - **Survivorship-bias-free index testing:** Supports point-in-time constituent CSVs with `Ticker`, `StartDate`, and `EndDate`.
                - **Liquidity and slippage filters:** Penalizes illiquid names and order sizes that may be too large versus average traded value.
                - **Sector-relative strength:** Compares the stock against a sector or industry benchmark ticker.
                - **Market-regime filter:** Checks benchmark trend and volatility before giving full credit to bottom signals.
                - **Signal database:** Stores daily signals locally in SQLite for monitoring and later review.

                **Important implementation note**

                The live score uses the full engine. The backtest uses a lighter proxy score so the app remains fast.
                For production, you should vectorize or batch the full signal engine and run a more rigorous walk-forward test.
                """
            )

    with charts_tab:
        st.markdown("## Charts for all major indicators")
        st.caption("Select only the charts you want to inspect. This keeps the chart pack readable instead of showing every indicator at once.")

        all_chart_options = [
            "Price + key moving averages",
            "200-week SMA",
            "Yearly anchored VWAP",
            "2-year linear regression channel",
            "Daily RSI",
            "Weekly RSI / divergence context",
            "Daily MACD",
            "Weekly MACD",
            "Coppock Curve",
            "Fixed range volume profile",
            "Stan Weinstein Stage Analysis",
            "OBV trend",
            "Historical volatility percentile",
            "ATR compression",
        ]

        selected_indicator_charts = st.multiselect(
            "Choose indicator charts to display",
            options=all_chart_options,
            default=[
                "Price + key moving averages",
                "Yearly anchored VWAP",
                "2-year linear regression channel",
                "Daily RSI",
                "Weekly RSI / divergence context",
                "Daily MACD",
                "Weekly MACD",
                "Fixed range volume profile",
                "OBV trend",
                "ATR compression",
            ],
        )

        plot_indicator_chart_grid(df, chart_lookback, selected_indicator_charts)

    with calculations_tab:
        render_calculation_breakdown(
            ticker=ticker,
            df=df,
            context=context,
            signals=signals,
            backtest=backtest,
            base_score=base_score,
            final_score=final_score,
        )

    st.caption("Educational research tool only. Not investment advice.")


if __name__ == "__main__":
    main()
