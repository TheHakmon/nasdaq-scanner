#!/usr/bin/env python3
"""
NASDAQ Technical Breakout Scanner
==================================
Scans a curated universe of NASDAQ-listed stocks (market cap > $5B) for
technical breakouts above the 150-day moving average accompanied by
above-average volume — a Stage 2 / Minervini-style entry setup.

Outputs:
  scan_results.json   — full structured results (read by the dashboard)
  scan_results.md     — human-readable summary (used for WhatsApp / email body)

Run daily after US market close (16:30 ET).
Requires: yfinance, pandas, numpy
"""

from __future__ import annotations
import base64
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

def _pip_install(pkg: str) -> None:
    """Install a package, handling pip flag differences across versions."""
    import subprocess
    # Try the modern flag first (pip >= 23.0.1, needed on macOS 12.4+ to bypass PEP 668)
    for args in (
        [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", pkg],
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", pkg],
        [sys.executable, "-m", "pip", "install", "--quiet", pkg],
    ):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            return
    # If we get here, all attempts failed — re-raise the last error visibly
    raise RuntimeError(f"Failed to install {pkg}. Last stderr:\n{r.stderr}")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...", file=sys.stderr)
    _pip_install("yfinance")
    import yfinance as yf

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    print("Installing matplotlib...", file=sys.stderr)
    _pip_install("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
UNIVERSE_PATH = HERE / "nasdaq_universe.json"
PERSONAL_PATH = HERE / "personal_watchlist.json"
RESULTS_JSON = HERE / "scan_results.json"
RESULTS_MD = HERE / "scan_results.md"
EMAIL_CONFIG_PATH = Path.home() / ".nasdaq_scanner_email.json"
TELEGRAM_CONFIG_PATH = Path.home() / ".nasdaq_scanner_telegram.json"

MARKET_CAP_MIN = 5_000_000_000        # $5B
HISTORY_PERIOD = "1y"
BENCHMARK = "SPY"
BREAKOUT_LOOKBACK_DAYS = 5            # how recently the cross above MA150 happened
VOLUME_MULTIPLIER = 1.5               # day's volume vs 50-day average
RSI_LOW, RSI_HIGH = 50, 75
MIN_DOLLAR_VOLUME = 10_000_000        # $10M avg daily $-volume (liquidity)
EARNINGS_WARN_DAYS = 7                # warn if earnings within N days
ATR_STOP_MULT = 2.0                   # stop = entry - 2 * ATR
ATR_TARGET1_MULT = 4.0                # target1 = entry + 4 * ATR (1:2 R/R)
ATR_TARGET2_MULT = 6.0                # target2 = entry + 6 * ATR (1:3 R/R)
NEAR_52W_HIGH_PCT = 5.0               # "near 52-week high" if within 5% of it
NEWS_PER_TICKER = 3                   # how many news headlines to include
ADX_TREND_MIN = 25                   # ADX above this = strong trend
EXTENDED_MA50_PCT = 15.0             # > this % above MA50 = "extended" (chasing risk)
STRONG_GROWTH_PCT = 0.20             # earnings/revenue YoY growth >= 20% = strong

# Map yfinance sector names to SPDR sector ETFs for sector-strength check.
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=n).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sector_strength(hist: pd.DataFrame) -> dict | None:
    """Given a sector ETF's price history, classify its current strength.
    Returns dict with strength label, recent returns, and MA position."""
    if hist is None or len(hist) < 50:
        return None
    close = hist["Close"]
    last = float(close.iloc[-1])
    ma50_last = float(sma(close, 50).iloc[-1])
    above_ma50 = last > ma50_last
    ret_1m = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
    ret_3m = (last / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None

    # Classify
    if above_ma50 and ret_1m is not None and ret_1m > 3:
        strength = "strong"
        label = "סקטור חזק"
        color = "#059669"
        emoji = "🔥"
    elif above_ma50 and ret_1m is not None and ret_1m > 0:
        strength = "moderate"
        label = "סקטור מתפקד"
        color = "#10b981"
        emoji = "✓"
    elif not above_ma50 and ret_1m is not None and ret_1m < -3:
        strength = "weak"
        label = "סקטור חלש"
        color = "#dc2626"
        emoji = "⚠️"
    else:
        strength = "neutral"
        label = "סקטור נייטרלי"
        color = "#6b7280"
        emoji = "•"

    return {
        "strength": strength,
        "label": label,
        "color": color,
        "emoji": emoji,
        "above_ma50": above_ma50,
        "ret_1m": round(ret_1m, 2) if ret_1m is not None else None,
        "ret_3m": round(ret_3m, 2) if ret_3m is not None else None,
    }


def atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. Requires columns: High, Low, Close."""
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures TREND STRENGTH (not direction).
    ADX > 25 ≈ strong trend; < 20 ≈ choppy/no trend."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=hist.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=hist.index)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rs_raw_score(close: pd.Series) -> float | None:
    """IBD-style weighted multi-period return — the basis for the RS Rating.
    More weight on recent performance. Ranked across the universe in main()."""
    if len(close) < 70:
        return None
    def ret(n):
        if len(close) < n + 1:
            return None
        return close.iloc[-1] / close.iloc[-n] - 1
    r3, r6, r9, r12 = ret(63), ret(126), ret(189), ret(252)
    # Use available periods; weight recent quarter most.
    parts, weights = [], []
    for r, w in ((r3, 0.4), (r6, 0.2), (r9, 0.2), (r12, 0.2)):
        if r is not None:
            parts.append(r * w)
            weights.append(w)
    if not parts:
        return None
    return sum(parts) / sum(weights)


def weekly_above_ma(close: pd.Series, weeks: int = 30) -> tuple[bool, float | None]:
    """Resample daily prices to weekly close and check if price > MA(weeks).
    MA30 weekly ~ MA150 daily — a higher-timeframe confirmation."""
    if len(close) < weeks * 5 + 5:
        return False, None
    wk = close.resample("W-FRI").last().dropna()
    ma = wk.rolling(weeks, min_periods=weeks).mean()
    if pd.isna(ma.iloc[-1]):
        return False, None
    return bool(wk.iloc[-1] > ma.iloc[-1]), float(ma.iloc[-1])


def relative_strength(stock_close: pd.Series, bench_close: pd.Series) -> float:
    """Ratio of 3-month % return vs benchmark. >1 means stronger than benchmark."""
    if len(stock_close) < 65 or len(bench_close) < 65:
        return float("nan")
    s_ret = stock_close.iloc[-1] / stock_close.iloc[-65] - 1
    b_ret = bench_close.iloc[-1] / bench_close.iloc[-65] - 1
    return float((1 + s_ret) / (1 + b_ret))


def build_chart_data(hist: pd.DataFrame, ma50: pd.Series, ma150: pd.Series,
                     ma200: pd.Series, days: int = 130) -> dict:
    """Extract the last `days` of OHLC and MA series in a compact JSON-friendly
    shape that TradingView Lightweight Charts can consume directly."""
    sub = hist.iloc[-days:]
    ma50 = ma50.reindex(sub.index)
    ma150 = ma150.reindex(sub.index)
    ma200 = ma200.reindex(sub.index)

    def fmt_time(ts):
        return ts.strftime("%Y-%m-%d")

    ohlc = []
    for ts, row in sub.iterrows():
        ohlc.append({
            "time": fmt_time(ts),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        })

    def series_of(s):
        out = []
        for ts, v in s.items():
            if pd.isna(v):
                continue
            out.append({"time": fmt_time(ts), "value": round(float(v), 2)})
        return out

    # Detect the most recent breakout cross (close crossed up through MA150)
    breakout_marker = None
    try:
        close = sub["Close"]
        cross_up = (close.shift(1) < ma150.shift(1)) & (close >= ma150)
        cross_dates = sub.index[cross_up.fillna(False).values]
        if len(cross_dates) > 0:
            d = cross_dates[-1]
            breakout_marker = {
                "time": fmt_time(d),
                "price": round(float(close.loc[d]), 2),
            }
    except Exception:
        pass

    return {
        "ohlc": ohlc,
        "ma50": series_of(ma50),
        "ma150": series_of(ma150),
        "ma200": series_of(ma200),
        "breakout_marker": breakout_marker,
    }


def make_chart_b64(ticker: str, hist: pd.DataFrame,
                   ma50: pd.Series, ma150: pd.Series, ma200: pd.Series) -> str | None:
    """Generate a price+volume chart as base64-encoded PNG.

    Top panel: price line, MA50/150/200 overlays, breakout zone shaded,
               annotation arrow at the breakout cross.
    Bottom panel: volume bars colored by up/down day, with 50-day avg line.
    """
    try:
        close = hist["Close"]
        volume = hist["Volume"]
        # Use only the visible window (last ~6 months for clarity)
        view = close.iloc[-130:]
        view_idx = view.index
        view_ma50 = ma50.reindex(view_idx)
        view_ma150 = ma150.reindex(view_idx)
        view_ma200 = ma200.reindex(view_idx)
        view_vol = volume.reindex(view_idx)
        view_vol_avg = view_vol.rolling(50, min_periods=20).mean()

        fig = plt.figure(figsize=(7.5, 4.5), dpi=110, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_v = fig.add_subplot(gs[1], sharex=ax)

        # Price + MAs
        ax.plot(view_idx, view.values, color="#111827", lw=1.5, label="Close")
        ax.plot(view_idx, view_ma50.values, color="#3b82f6", lw=1.2, label="MA50")
        ax.plot(view_idx, view_ma150.values, color="#dc2626", lw=1.6, label="MA150 (key)")
        ax.plot(view_idx, view_ma200.values, color="#9ca3af", lw=1.0,
                ls="--", label="MA200")

        # Shade region above MA150 (the bullish zone we care about)
        above = view.values > view_ma150.values
        ax.fill_between(view_idx, view_ma150.values, view.values,
                        where=above, color="#10b981", alpha=0.12,
                        interpolate=True, label="_above_ma150")

        # Mark the breakout cross point (last day where close was below MA150
        # before going above)
        try:
            cross = (view.shift(1) < view_ma150.shift(1)) & (view >= view_ma150)
            cross_dates = view_idx[cross.values]
            if len(cross_dates) > 0:
                last_cross = cross_dates[-1]
                cross_price = view.loc[last_cross]
                ax.annotate(
                    f"Breakout\n${cross_price:.2f}",
                    xy=(last_cross, cross_price),
                    xytext=(15, 25), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#065f46",
                    arrowprops=dict(arrowstyle="->", color="#065f46", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#d1fae5", ec="#10b981")
                )
        except Exception:
            pass

        # Last close marker
        ax.scatter([view_idx[-1]], [view.iloc[-1]], color="#111827",
                   zorder=5, s=30)
        ax.text(view_idx[-1], view.iloc[-1],
                f"  ${view.iloc[-1]:.2f}", va="center", fontsize=9,
                fontweight="bold")

        ax.set_title(f"{ticker} — last 6 months", fontsize=11, fontweight="bold",
                     loc="left")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Price ($)", fontsize=9)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Volume bars, colored by up/down day
        diffs = view.diff()
        colors = np.where(diffs.values >= 0, "#10b981", "#ef4444")
        ax_v.bar(view_idx, view_vol.values, color=colors, alpha=0.65, width=1.0)
        ax_v.plot(view_idx, view_vol_avg.values, color="#374151", lw=1.0,
                  label="50d avg")
        ax_v.set_ylabel("Volume", fontsize=9)
        ax_v.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax_v.grid(True, alpha=0.25)
        ax_v.xaxis.set_major_locator(mdates.MonthLocator())
        ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        # Format volume axis in millions
        ax_v.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

        # constrained_layout=True at figure creation handles layout automatically.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"  chart error for {ticker}: {e}", file=sys.stderr)
        return None


# ----------------------------------------------------------------------------
# Per-ticker analysis
# ----------------------------------------------------------------------------
def analyze(ticker: str, bench_close: pd.Series, is_personal: bool = False,
            sector_strengths: dict | None = None) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=HISTORY_PERIOD, auto_adjust=True)
        if hist is None or len(hist) < 160:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        # Market cap filter — skip if we can confirm cap is below threshold
        # (personal watchlist tickers skip this filter)
        try:
            info = t.fast_info
            cap = info.get("market_cap") or info.get("marketCap")
        except Exception:
            cap = None
        if not is_personal and cap is not None and cap < MARKET_CAP_MIN:
            return None

        # Liquidity check: average daily $-volume over last 50 days.
        dollar_vol = (close * volume).iloc[-50:].mean()
        if not is_personal and dollar_vol < MIN_DOLLAR_VOLUME:
            return None  # too illiquid

        # Sector / industry / earnings — these come from t.info (slower).
        # Cache so we don't hit it twice.
        sector = None
        industry = None
        days_to_earnings = None
        earnings_growth = None
        revenue_growth = None
        try:
            full_info = t.info
            sector = full_info.get("sector")
            industry = full_info.get("industry")
            # YoY growth as fractions (e.g. 0.25 = +25%)
            earnings_growth = full_info.get("earningsGrowth") or full_info.get("earningsQuarterlyGrowth")
            revenue_growth = full_info.get("revenueGrowth")
        except Exception:
            full_info = {}
        try:
            cal = t.calendar
            if cal is not None and hasattr(cal, 'get'):
                earnings_date = cal.get('Earnings Date')
                if isinstance(earnings_date, list) and earnings_date:
                    earnings_date = earnings_date[0]
                if earnings_date:
                    days_to_earnings = (pd.Timestamp(earnings_date).tz_localize(None) -
                                        pd.Timestamp.now()).days
        except Exception:
            pass

        ma150 = sma(close, 150)
        ma50 = sma(close, 50)
        ma200 = sma(close, 200)
        rsi14 = rsi(close, 14)
        vol_avg50 = volume.rolling(50, min_periods=20).mean()
        atr14 = atr(hist, 14)
        adx14 = adx(hist, 14)
        last_adx = float(adx14.iloc[-1]) if not pd.isna(adx14.iloc[-1]) else None
        rs_raw = rs_raw_score(close)   # ranked into rs_rating (1-99) in main()
        weekly_ok, weekly_ma = weekly_above_ma(close, weeks=30)

        # 52-week high distance
        high_52w = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
        distance_from_52w = (float(close.iloc[-1]) / high_52w - 1) * 100  # 0 or negative
        near_52w_high = distance_from_52w >= -NEAR_52W_HIGH_PCT

        # Sector strength
        sector_strength = None
        if sector and sector_strengths:
            sector_strength = sector_strengths.get(sector)

        # Latest news headlines (best-effort)
        news_items = []
        try:
            raw_news = getattr(t, "news", None) or []
            for n in raw_news[:NEWS_PER_TICKER]:
                # Newer yfinance versions wrap items in {'content': {...}}
                src = n.get("content") if isinstance(n, dict) and "content" in n else n
                title = src.get("title") if isinstance(src, dict) else None
                publisher = None
                link = None
                ts = None
                if isinstance(src, dict):
                    publisher = (src.get("provider") or {}).get("displayName") if isinstance(src.get("provider"), dict) else src.get("publisher")
                    link = (src.get("canonicalUrl") or {}).get("url") if isinstance(src.get("canonicalUrl"), dict) else src.get("link")
                    ts = src.get("providerPublishTime") or src.get("pubDate")
                if title:
                    news_items.append({
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "time": ts,
                    })
        except Exception:
            pass

        last = close.iloc[-1]
        last_vol = volume.iloc[-1]
        last_ma150 = ma150.iloc[-1]
        last_ma50 = ma50.iloc[-1]
        last_ma200 = ma200.iloc[-1]
        last_rsi = rsi14.iloc[-1]
        last_vol_avg = vol_avg50.iloc[-1]

        # Pct distance from MA150
        pct_above_ma150 = (last / last_ma150 - 1) * 100 if last_ma150 else float("nan")
        pct_above_ma50 = (last / last_ma50 - 1) * 100 if last_ma50 else float("nan")
        vol_ratio = last_vol / last_vol_avg if last_vol_avg else float("nan")
        # Extended above MA50? (chasing risk)
        is_extended = bool(not pd.isna(pct_above_ma50) and pct_above_ma50 > EXTENDED_MA50_PCT)
        # Trend strength
        strong_trend = bool(last_adx is not None and last_adx >= ADX_TREND_MIN)
        # Fundamental growth
        strong_growth = bool(
            (earnings_growth is not None and earnings_growth >= STRONG_GROWTH_PCT) or
            (revenue_growth is not None and revenue_growth >= STRONG_GROWTH_PCT)
        )

        # Was there a recent cross above MA150 (within lookback window)?
        recent_above = (close.iloc[-BREAKOUT_LOOKBACK_DAYS:] > ma150.iloc[-BREAKOUT_LOOKBACK_DAYS:]).all()
        prior_below = (close.iloc[-BREAKOUT_LOOKBACK_DAYS - 5:-BREAKOUT_LOOKBACK_DAYS] <
                       ma150.iloc[-BREAKOUT_LOOKBACK_DAYS - 5:-BREAKOUT_LOOKBACK_DAYS]).any()
        is_recent_breakout = bool(recent_above and prior_below)

        # Stage 2 alignment: price > MA50 > MA150 > MA200 (or close to it)
        stage2 = bool(
            last > last_ma50
            and last_ma50 > last_ma150
            and last_ma150 > last_ma200
        )

        # Trend change indicators
        ma200_rising = bool(ma200.iloc[-1] > ma200.iloc[-21])  # rising over 1 month
        ma50_above_ma150 = bool(last_ma50 > last_ma150)
        golden_cross_recent = bool(
            (ma50.iloc[-21:] > ma150.iloc[-21:]).iloc[-1]
            and not (ma50.iloc[-42:-21] > ma150.iloc[-42:-21]).all()
        )

        rs = relative_strength(close, bench_close)

        # Composite score
        score = 0
        reasons = []
        if last > last_ma150:
            score += 20
            reasons.append("מעל MA150")
        if is_recent_breakout:
            score += 25
            reasons.append("פריצה טרייה של MA150")
        if vol_ratio and vol_ratio >= VOLUME_MULTIPLIER:
            score += 15
            reasons.append(f"נפח גבוה (×{vol_ratio:.1f})")
        if stage2:
            score += 15
            reasons.append("Stage 2 alignment")
        if RSI_LOW <= last_rsi <= RSI_HIGH:
            score += 10
            reasons.append(f"RSI={last_rsi:.0f}")
        elif last_rsi > RSI_HIGH:
            reasons.append(f"RSI גבוה ({last_rsi:.0f}) - אפשר אזור קניית יתר")
        if rs and rs > 1.0:
            score += 10
            reasons.append(f"RS={rs:.2f} מעל S&P")
        if golden_cross_recent:
            score += 5
            reasons.append("Golden cross MA50/MA150 טרי")
        if weekly_ok:
            score += 10
            reasons.append("אישור MA30 שבועי")
        # Earnings warning — important risk flag, no score impact but flagged
        if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_WARN_DAYS:
            reasons.append(f"⚠️ דוח בעוד {days_to_earnings} ימים")
            score -= 5  # small penalty for the risk
        if near_52w_high:
            score += 8
            reasons.append(f"🔥 קרוב לשיא 52 שבועות ({distance_from_52w:.1f}%)")
        if sector_strength:
            if sector_strength["strength"] == "strong":
                score += 8
                reasons.append(f"סקטור חזק ({sector_strength['emoji']})")
            elif sector_strength["strength"] == "weak":
                score -= 5
                reasons.append(f"⚠️ סקטור חלש")
        # ADX trend strength
        if strong_trend:
            score += 8
            reasons.append(f"מגמה חזקה (ADX {last_adx:.0f})")
        elif last_adx is not None and last_adx < 20:
            score -= 3
            reasons.append(f"⚠️ מגמה חלשה (ADX {last_adx:.0f})")
        # Extension above MA50 (chasing risk)
        if is_extended:
            score -= 6
            reasons.append(f"⚠️ מתוח (+{pct_above_ma50:.0f}% מעל MA50)")
        # Fundamental growth bonus
        if strong_growth:
            score += 8
            g = max([x for x in (earnings_growth, revenue_growth) if x is not None])
            reasons.append(f"📈 צמיחה חזקה ({g*100:.0f}%)")

        # Only include candidates that pass minimum bar
        passed = (
            last > last_ma150
            and (vol_ratio is None or vol_ratio >= 1.0)
            and last_rsi < 80
        )

        # ---- Trade levels: Stop-Loss & Targets via ATR ----
        last_atr = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else None
        stop_loss = target1 = target2 = risk_reward = None
        if last_atr:
            entry = float(last)
            stop_loss = round(entry - ATR_STOP_MULT * last_atr, 2)
            target1 = round(entry + ATR_TARGET1_MULT * last_atr, 2)
            target2 = round(entry + ATR_TARGET2_MULT * last_atr, 2)
            risk = entry - stop_loss
            reward = target1 - entry
            risk_reward = round(reward / risk, 2) if risk > 0 else None

        # ---- Build narrative summary in Hebrew ----
        summary_parts = []
        # 1. The MA150 setup
        if is_recent_breakout:
            summary_parts.append(
                f"המניה חצתה זה עתה את ממוצע 150 הימים מלמטה (כעת {pct_above_ma150:+.1f}% מעליו) — "
                "פריצה טרייה שמסמנת מעבר אפשרי ממגמה יורדת/מדשדשת לעולה."
            )
        elif last > last_ma150:
            summary_parts.append(
                f"המניה נסחרת {pct_above_ma150:+.1f}% מעל ממוצע 150 הימים — מגמה עולה אובייקטיבית "
                "לפי הגדרת ויינשטיין."
            )

        # 2. Volume confirmation
        if vol_ratio and vol_ratio >= VOLUME_MULTIPLIER:
            summary_parts.append(
                f"הנפח ביום הפריצה היה ×{vol_ratio:.1f} מהממוצע של 50 הימים — אישור חזק שגופים גדולים "
                "נכנסים לפוזיציה (ולא רק קונים בודדים)."
            )
        elif vol_ratio and vol_ratio >= 1.0:
            summary_parts.append(
                f"הנפח ×{vol_ratio:.1f} מהממוצע — נורמלי, לא ראינו עדיין כסף גדול נכנס. "
                "כדאי לחכות לאישור נפח לפני כניסה."
            )

        # 3. Stage 2 / trend alignment
        if stage2:
            summary_parts.append(
                f"מבנה Stage 2 קלאסי: מחיר (${last:.2f}) > MA50 (${last_ma50:.2f}) > "
                f"MA150 (${last_ma150:.2f}) > MA200 (${last_ma200:.2f}) — כל הממוצעים בסדר עולה, "
                "כלומר המגמה ארוכת הטווח עולה והקצרה מובילה אותה."
            )
        elif last > last_ma50 > last_ma150:
            summary_parts.append(
                "מבנה ממוצעים חיובי (מחיר > MA50 > MA150) אבל MA200 עוד לא תומך — "
                "מגמה עולה לטווח קצר/בינוני, ייתכן שלפני שינוי מגמה ארוך."
            )

        # 4. RSI / momentum
        if RSI_LOW <= last_rsi <= RSI_HIGH:
            summary_parts.append(
                f"RSI={last_rsi:.0f} — מומנטום בריא, באמצע התחום, לא מעיד על קניית יתר."
            )
        elif last_rsi > RSI_HIGH:
            summary_parts.append(
                f"⚠️ RSI={last_rsi:.0f} מעיד על אזור קניית יתר. ייתכן שהפריצה כבר הספיקה לרוץ — "
                "עדיף לחכות לרגיעה/משיכה אל הממוצע לפני כניסה במקום לרדוף."
            )
        elif last_rsi < RSI_LOW:
            summary_parts.append(
                f"RSI={last_rsi:.0f} עוד נמוך — המומנטום עוד לא התעורר, הפריצה זהירה."
            )

        # 5. Relative strength
        if rs and rs > 1.1:
            summary_parts.append(
                f"חוזק יחסי של {rs:.2f} מול S&P 500 — המניה עלתה משמעותית יותר מהמדד ב-3 חודשים "
                "האחרונים, מה שמסמן שמובילי שוק קונים אותה."
            )
        elif rs and rs > 1.0:
            summary_parts.append(
                f"חוזק יחסי {rs:.2f} מעל המדד — מתפקדת מעט טוב יותר מ-S&P."
            )
        elif rs and rs < 0.95:
            summary_parts.append(
                f"⚠️ חוזק יחסי {rs:.2f} מתחת ל-S&P — המניה מפגרת אחרי השוק. "
                "פריצה במניה חלשה לעומת השוק היא אות פחות אמין."
            )

        # 6. Golden cross bonus
        if golden_cross_recent:
            summary_parts.append(
                "🌟 חצייה טרייה של MA50 מעל MA150 (Golden Cross של מסחר בינוני) — "
                "אישור נוסף לשינוי מגמה."
            )

        # 6b. Weekly MA confirmation
        if weekly_ok:
            summary_parts.append(
                "📅 גם במסגרת זמן שבועית המחיר מעל MA30 (≈ MA150 יומי) — "
                "אישור במסגרת זמן גבוהה יותר, מקטין סיכון של false breakout."
            )

        # 6c. Earnings warning
        if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_WARN_DAYS:
            summary_parts.append(
                f"⚠️ **דוח כספי בעוד {days_to_earnings} ימים** — הימור על כיוון הדוח. "
                "אם נכנסים, להקטין גודל פוזיציה או לחכות לדוח."
            )

        # 6e. ADX trend strength
        if strong_trend:
            summary_parts.append(
                f"💪 ADX {last_adx:.0f} — מגמה חזקה ומבוססת (לא דשדוש). אות אמין יותר."
            )
        elif last_adx is not None and last_adx < 20:
            summary_parts.append(
                f"⚠️ ADX {last_adx:.0f} — מגמה חלשה/דשדוש. הפריצה פחות אמינה."
            )

        # 6f. Extension
        if is_extended:
            summary_parts.append(
                f"⚠️ המחיר מתוח +{pct_above_ma50:.0f}% מעל MA50 — ייתכן שכבר רץ. "
                "סיכון של רדיפה; עדיף להמתין למשיכה."
            )

        # 6g. Fundamental growth
        if strong_growth:
            bits = []
            if earnings_growth is not None:
                bits.append(f"רווחים {earnings_growth*100:+.0f}%")
            if revenue_growth is not None:
                bits.append(f"הכנסות {revenue_growth*100:+.0f}%")
            summary_parts.append(
                f"📈 צמיחה יסודית חזקה ({', '.join(bits)}) — לא רק טכני, גם העסק צומח."
            )

        # 6d. Trade levels
        if stop_loss and target1 and target2 and risk_reward:
            summary_parts.append(
                f"🎯 רמות מסחר מוצעות: כניסה ${last:.2f} · "
                f"Stop-Loss ${stop_loss} · "
                f"יעד 1 ${target1} (1:{risk_reward}) · "
                f"יעד 2 ${target2}."
            )

        # 7. Bottom-line entry rationale
        if score >= 70 and is_recent_breakout and stage2:
            verdict = "💎 הגדרה טכנית קלאסית: פריצה + Stage 2 + נפח. מועמדת חזקה לכניסה."
        elif score >= 70:
            verdict = "✅ מועמדת מעניינת לכניסה — מספר אישורים טכניים מצטלבים."
        elif score >= 60:
            verdict = "🔶 מועמדת על הגבול — חלק מהאישורים קיימים, חלק חסר. במעקב."
        elif is_recent_breakout:
            verdict = "🟡 פריצה ראשונית. עוקבים אחרי אישורי נפח/חוזק יחסי לפני כניסה."
        else:
            verdict = "📊 במגמה חיובית אך ללא אות כניסה ספציפי כעת."

        summary_parts.append(verdict)
        summary = " ".join(summary_parts)

        # Generate chart only for candidates that pass — keeps scan fast.
        chart_b64 = None
        chart_data = None
        if passed:
            chart_b64 = make_chart_b64(ticker, hist, ma50, ma150, ma200)
            chart_data = build_chart_data(hist, ma50, ma150, ma200, days=130)

        return {
            "ticker": ticker,
            "passed": passed,
            "score": score,
            "summary": summary,
            "chart_b64": chart_b64,
            "chart_data": chart_data,
            "price": round(float(last), 2),
            "ma150": round(float(last_ma150), 2) if not pd.isna(last_ma150) else None,
            "ma50": round(float(last_ma50), 2) if not pd.isna(last_ma50) else None,
            "ma200": round(float(last_ma200), 2) if not pd.isna(last_ma200) else None,
            "pct_above_ma150": round(float(pct_above_ma150), 2),
            "pct_above_ma50": round(float(pct_above_ma50), 2),
            "rsi": round(float(last_rsi), 1) if not pd.isna(last_rsi) else None,
            "vol_ratio": round(float(vol_ratio), 2) if vol_ratio else None,
            "rs_vs_spy": round(float(rs), 2) if not pd.isna(rs) else None,
            "is_recent_breakout": is_recent_breakout,
            "stage2": stage2,
            "ma200_rising": ma200_rising,
            "ma50_above_ma150": ma50_above_ma150,
            "golden_cross_recent": golden_cross_recent,
            "weekly_above_ma": weekly_ok,
            "market_cap": cap,
            "sector": sector,
            "industry": industry,
            "dollar_volume_avg": round(float(dollar_vol)) if dollar_vol else None,
            "days_to_earnings": days_to_earnings,
            "atr": round(last_atr, 2) if last_atr else None,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "risk_reward": risk_reward,
            "is_personal": is_personal,
            "high_52w": round(high_52w, 2),
            "distance_from_52w": round(distance_from_52w, 2),
            "near_52w_high": near_52w_high,
            "sector_strength": sector_strength,
            "adx": round(last_adx, 1) if last_adx is not None else None,
            "strong_trend": strong_trend,
            "is_extended": is_extended,
            "earnings_growth": round(earnings_growth, 3) if earnings_growth is not None else None,
            "revenue_growth": round(revenue_growth, 3) if revenue_growth is not None else None,
            "strong_growth": strong_growth,
            "rs_raw": round(rs_raw, 4) if rs_raw is not None else None,
            "rs_rating": None,  # filled in main() after ranking the whole universe
            "news": news_items,
            "reasons": reasons,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def build_journal_prices(payload: dict) -> None:
    """Inject a {ticker: price} map and date into journal_app.html so the
    interactive journal can show current prices and pre-fill buy prices."""
    import re
    journal_path = HERE / "journal_app.html"
    if not journal_path.exists():
        return
    # Build price map from every scanned result (broadest coverage).
    price_map = {}
    for bucket in ("breakouts", "watchlist", "personal", "all_results"):
        for r in payload.get(bucket, []):
            if r.get("ticker") and r.get("price") is not None:
                price_map[r["ticker"]] = r["price"]
    date_str = payload.get("generated_at", "")[:10]

    html = journal_path.read_text()
    html = re.sub(
        r"const PRICE_MAP = /\*__PRICE_MAP__\*/[^;]*;",
        "const PRICE_MAP = /*__PRICE_MAP__*/ " + json.dumps(price_map, ensure_ascii=False) + ";",
        html, count=1,
    )
    html = re.sub(
        r'const PRICE_MAP_DATE = /\*__PRICE_MAP_DATE__\*/[^;]*;',
        'const PRICE_MAP_DATE = /*__PRICE_MAP_DATE__*/ ' + json.dumps(date_str) + ";",
        html, count=1,
    )
    journal_path.write_text(html)
    print(f"Injected {len(price_map)} prices into journal_app.html")


def _rebuild_archive_index(archive_dir: Path) -> None:
    """Rebuild archive/index.html by listing all scan_*.json files."""
    entries = []
    for jf in sorted(archive_dir.glob("scan_*.json"), reverse=True):
        try:
            d = json.loads(jf.read_text())
            date_str = jf.stem.replace("scan_", "")
            top = ", ".join([r["ticker"] for r in d.get("breakouts", [])[:5]]) or "—"
            try:
                weekday = datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            except Exception:
                weekday = ""
            entries.append({
                "date": date_str, "weekday": weekday,
                "breakouts": len(d.get("breakouts", [])),
                "watchlist": len(d.get("watchlist", [])),
                "top_tickers": top,
            })
        except Exception:
            continue
    # Load shared theme CSS/JS
    try:
        sys.path.insert(0, str(HERE))
        from _theme import THEME_HEAD, THEME_TOGGLE_SCRIPT, THEME_TOGGLE_BUTTON
    except Exception:
        THEME_HEAD = THEME_TOGGLE_SCRIPT = ""
        THEME_TOGGLE_BUTTON = ""

    rows = []
    for e in entries:
        rows.append(f"""
        <tr onclick="location.href='dashboard_{e['date']}.html'" style="cursor:pointer">
          <td><strong>{e['date']}</strong> <span style="color:var(--text-muted);font-size:11px;">({e['weekday']})</span></td>
          <td style="text-align:center;color:{'var(--success)' if e['breakouts']>0 else 'var(--text-subtle)'};">{e['breakouts']}</td>
          <td style="text-align:center;color:{'var(--warning)' if e['watchlist']>0 else 'var(--text-subtle)'};">{e['watchlist']}</td>
          <td style="font-size:12px;color:var(--text-muted);">{e['top_tickers']}</td>
          <td><a href="dashboard_{e['date']}.html" style="color:var(--accent);">פתח →</a></td>
        </tr>""")
    html = f"""<!doctype html>
<html lang="he" dir="rtl"><head>
<meta charset="utf-8"><title>NASDAQ Archive</title>
{THEME_HEAD}
<style>
  body {{ font-family:-apple-system,'Segoe UI','Heebo',sans-serif; margin:0; padding:24px; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  .top-nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; gap:8px; flex-wrap:wrap; }}
  .nav-links {{ display:flex; gap:8px; align-items:center; }}
  .nav-btn {{ display:inline-flex; gap:6px; padding:8px 14px; border-radius:8px;
              font-size:13px; text-decoration:none; font-weight:500; }}
  h1 {{ margin:0 0 4px; }}
  .sub {{ color:var(--text-muted); font-size:14px; margin-bottom:24px; }}
  table {{ width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding:10px 14px; text-align:start; font-size:14px; }}
  th {{ font-weight:600; }}
  a {{ text-decoration:none; }}
</style></head><body><div class="wrap">
<div class="top-nav">
  <h1>📚 ארכיון סריקות NASDAQ</h1>
  <div class="nav-links">
    <a class="nav-btn" href="../dashboard_standalone.html">← חזרה לדשבורד</a>
    {THEME_TOGGLE_BUTTON}
  </div>
</div>
<div class="sub">{len(entries)} סריקות יומיות. לחץ על שורה לפתיחת הסריקה של אותו יום.</div>
<table><thead><tr><th>תאריך</th><th style="text-align:center">🚀 פריצות</th><th style="text-align:center">👀 מעקב</th><th>טיקרים מובילים</th><th></th></tr></thead><tbody>{"".join(rows)}</tbody></table>
</div>
{THEME_TOGGLE_SCRIPT}
</body></html>"""
    (archive_dir / "index.html").write_text(html)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting scan...")

    universe = json.loads(UNIVERSE_PATH.read_text())["tickers"]
    universe = sorted(set(universe))
    # Load personal watchlist if present
    personal = []
    if PERSONAL_PATH.exists():
        try:
            personal = json.loads(PERSONAL_PATH.read_text()).get("tickers", [])
            personal = sorted(set(personal))
            print(f"Personal watchlist: {len(personal)} tickers")
        except Exception as e:
            print(f"(personal watchlist parse error: {e})", file=sys.stderr)
    personal_set = set(personal)
    # Personal tickers not already in universe — add them so we still scan them
    combined = sorted(set(universe) | personal_set)
    print(f"Universe size: {len(universe)} tickers ({len(combined)} total with personal)")

    # Benchmark
    print(f"Fetching benchmark {BENCHMARK}...")
    bench_hist = yf.Ticker(BENCHMARK).history(period=HISTORY_PERIOD, auto_adjust=True)
    bench_close = bench_hist["Close"]

    # Sector ETFs — for sector-strength check
    print(f"Fetching sector ETFs...")
    sector_strengths = {}
    for sector_name, etf_sym in SECTOR_ETF_MAP.items():
        try:
            etf_hist = yf.Ticker(etf_sym).history(period="6mo", auto_adjust=True)
            s = compute_sector_strength(etf_hist)
            if s:
                s["etf"] = etf_sym
                sector_strengths[sector_name] = s
        except Exception as e:
            print(f"  sector fetch {etf_sym} failed: {e}", file=sys.stderr)
    print(f"  fetched {len(sector_strengths)} sector ETFs")

    results = []
    errors = []
    for i, ticker in enumerate(combined, 1):
        if i % 20 == 0:
            print(f"  ...{i}/{len(combined)}")
        is_personal = ticker in personal_set
        out = analyze(ticker, bench_close, is_personal=is_personal,
                       sector_strengths=sector_strengths)
        if out is None:
            continue
        if "error" in out:
            errors.append(out)
            continue
        results.append(out)
        # gentle pacing to avoid rate limits
        time.sleep(0.05)

    # ---- RS Rating: rank every scanned stock's raw RS into a 1-99 percentile ----
    rs_values = sorted(r["rs_raw"] for r in results if r.get("rs_raw") is not None)
    if rs_values:
        import bisect
        n = len(rs_values)
        for r in results:
            if r.get("rs_raw") is not None:
                # percentile rank (how many in the universe it beats)
                rank = bisect.bisect_right(rs_values, r["rs_raw"]) / n
                r["rs_rating"] = max(1, min(99, int(round(rank * 99))))
                # Bonus for top-tier relative strength
                if r["rs_rating"] >= 90:
                    r["score"] += 8
                    r["reasons"].append(f"RS Rating {r['rs_rating']} (טופ 10%)")
                elif r["rs_rating"] >= 80:
                    r["score"] += 4
                    r["reasons"].append(f"RS Rating {r['rs_rating']}")
        print(f"RS Rating computed across {n} stocks")

    # Sort: passed first, by score
    results.sort(key=lambda r: (not r["passed"], -r["score"]))

    # Top breakout candidates (passed + recent breakout, score>=50)
    breakouts = [r for r in results if r["passed"] and r["is_recent_breakout"] and r["score"] >= 50]
    watchlist = [r for r in results if r["passed"] and r["score"] >= 60 and r not in breakouts]
    # Personal watchlist results: anything from user's list (whether passed or not)
    personal_results = [r for r in results if r.get("is_personal")]

    # all_results table doesn't display charts — strip them to keep file small.
    all_results_lite = [
        {k: v for k, v in r.items() if k not in ("chart_b64", "chart_data")}
        for r in results
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "scanned": len(results),
        "errors": len(errors),
        "criteria": {
            "min_market_cap_usd": MARKET_CAP_MIN,
            "breakout_lookback_days": BREAKOUT_LOOKBACK_DAYS,
            "min_volume_multiplier": VOLUME_MULTIPLIER,
            "rsi_band": [RSI_LOW, RSI_HIGH],
        },
        "breakouts": breakouts,
        "watchlist": watchlist,
        "personal": personal_results,
        "all_results": all_results_lite,
        "error_tickers": [e["ticker"] for e in errors],
    }

    def _json_default(o):
        # Coerce numpy / pandas scalars to native Python types.
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        # Last resort
        try:
            return str(o)
        except Exception:
            return None

    RESULTS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    print(f"Wrote {RESULTS_JSON.name}: {len(breakouts)} breakouts, {len(watchlist)} watchlist")

    # Markdown summary for notifications
    lines = [f"# סריקת NASDAQ — {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines.append(f"נסרקו {len(results)} מניות. נמצאו **{len(breakouts)} פריצות** ו-**{len(watchlist)} מעקב**.\n")
    if breakouts:
        lines.append("## 🚀 פריצות מעניינות\n")
        for r in breakouts[:15]:
            lines.append(
                f"### {r['ticker']} — ציון {r['score']}\n"
                f"- מחיר: ${r['price']} | MA150: ${r['ma150']} (+{r['pct_above_ma150']}%) | "
                f"נפח ×{r['vol_ratio']} | RSI {r['rsi']} | RS {r['rs_vs_spy']}\n"
                f"- **למה:** {r.get('summary', ' · '.join(r['reasons']))}"
            )
    if watchlist:
        lines.append("\n## 👀 רשימת מעקב\n")
        for r in watchlist[:10]:
            lines.append(
                f"- **{r['ticker']}** — ${r['price']} | "
                f"+{r['pct_above_ma150']}% מעל MA150 | ציון {r['score']}"
            )
    if not breakouts and not watchlist:
        lines.append("_לא נמצאו מניות שעוברות את הסף היום._")

    RESULTS_MD.write_text("\n".join(lines))
    print(f"Wrote {RESULTS_MD.name}")

    # Generate the standalone dashboard (HTML with all data + charts inlined),
    # so the user can open it directly in any browser.
    try:
        tpl_path = HERE / "nasdaq_dashboard.html"
        if tpl_path.exists():
            tpl = tpl_path.read_text()
            inline = "const __INLINE_DATA__ = " + json.dumps(payload, ensure_ascii=False, default=_json_default) + ";"
            old_block = (
                'async function load() {\n'
                '  const errArea = document.getElementById("error-area");\n'
                '  errArea.innerHTML = "";\n'
                '  try {\n'
                '    const res = await window.cowork.callMcpTool("mcp__workspace__bash", {\n'
                '      command: `cat "${RESULTS_PATH}"`\n'
                '    });\n'
                '    // The MCP tool may return the raw text directly, or wrap it; handle both.\n'
                '    let text = typeof res === "string" ? res : (res?.content?.[0]?.text ?? res?.stdout ?? JSON.stringify(res));\n'
                '    // Strip any wrapping\n'
                '    text = text.trim();\n'
                '    const data = JSON.parse(text);'
            )
            new_block = (
                'async function load() {\n'
                '  const errArea = document.getElementById("error-area");\n'
                '  errArea.innerHTML = "";\n'
                '  try {\n'
                '    const data = __INLINE_DATA__;'
            )
            if old_block in tpl:
                tpl = tpl.replace(old_block, new_block)
                tpl = tpl.replace(
                    'const RESULTS_PATH = "/sessions/vigilant-zen-mendel/mnt/outputs/scan_results.json";',
                    inline + '\nconst RESULTS_PATH = "(inline)";'
                )
                (HERE / "dashboard_standalone.html").write_text(tpl)
                print(f"Wrote dashboard_standalone.html")
            else:
                print("(skipped standalone — template marker not found)")
    except Exception as e:
        print(f"(standalone build failed: {e})", file=sys.stderr)

    # Inject latest prices into the interactive journal app.
    try:
        build_journal_prices(payload)
    except Exception as e:
        print(f"(journal price injection failed: {e})", file=sys.stderr)

    # Archive: save a dated copy so we can build history over time.
    try:
        archive_dir = HERE / "archive"
        archive_dir.mkdir(exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        (archive_dir / f"scan_{today}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        )
        standalone = (HERE / "dashboard_standalone.html")
        if standalone.exists():
            (archive_dir / f"dashboard_{today}.html").write_text(standalone.read_text())
        # Rebuild archive index.html from existing files
        _rebuild_archive_index(archive_dir)
        print(f"Archived to archive/scan_{today}.json")
    except Exception as e:
        print(f"(archive save failed: {e})", file=sys.stderr)

    # Send email notification when there are breakouts.
    if breakouts:
        try:
            sent = send_email_notification(payload)
            if sent:
                print(f"Sent email notification ({len(breakouts)} breakouts).")
        except Exception as e:
            print(f"(email send failed: {e})", file=sys.stderr)
        # Telegram notification (independent of email)
        try:
            tg_sent = send_telegram_notification(payload)
            if tg_sent:
                print(f"Sent Telegram notification ({len(breakouts)} breakouts).")
        except Exception as e:
            print(f"(telegram send failed: {e})", file=sys.stderr)
    else:
        print("(no breakouts — skipping email/telegram)")

    print("Done.")
    return 0


def send_telegram_notification(payload: dict) -> bool:
    """Post the breakout summary to a Telegram chat via the Bot API.

    Reads credentials from ~/.nasdaq_scanner_telegram.json:
      {
        "bot_token": "123456:ABC-DEF...",   // from @BotFather
        "chat_id":   "123456789"            // your chat with the bot
      }

    To set up:
      1. Open @BotFather on Telegram, /newbot, pick a name & username, save the token.
      2. Open your new bot's chat, send /start.
      3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates and copy the
         "chat":{"id":...} value into chat_id.

    Returns True if sent, False if config missing.
    """
    if not TELEGRAM_CONFIG_PATH.exists():
        print(f"(telegram config not found at {TELEGRAM_CONFIG_PATH} — skipping)")
        return False
    cfg = json.loads(TELEGRAM_CONFIG_PATH.read_text())
    token = cfg.get("bot_token")
    chat_id = cfg.get("chat_id")
    if not token or not chat_id:
        print(f"(telegram config missing bot_token or chat_id)")
        return False

    breakouts = payload.get("breakouts", [])
    watchlist = payload.get("watchlist", [])
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [f"📈 *NASDAQ Scan — {today}*"]
    lines.append(f"_{len(breakouts)} פריצות · {len(watchlist)} מעקב_")
    lines.append("")
    if breakouts:
        lines.append("🚀 *פריצות:*")
        for r in breakouts[:10]:
            # Show ticker, score, price, %above MA150, link to TradingView
            tv = f"https://www.tradingview.com/symbols/NASDAQ-{r['ticker']}/"
            verdict_emoji = ""
            tr = r.get("historical_performance") or {}
            if tr.get("verdict") == "strong":
                verdict_emoji = "💎 "
            elif tr.get("verdict") == "good":
                verdict_emoji = "✅ "
            elif tr.get("verdict") == "weak":
                verdict_emoji = "❌ "
            stop = f" · stop ${r['stop_loss']}" if r.get("stop_loss") else ""
            t1 = f" · 🎯 ${r['target1']}" if r.get("target1") else ""
            lines.append(
                f"{verdict_emoji}[{r['ticker']}]({tv}) "
                f"score *{r['score']}* · ${r['price']} "
                f"(+{r['pct_above_ma150']}%){stop}{t1}"
            )
    if watchlist:
        lines.append("")
        lines.append("👀 *מעקב:* " + ", ".join(r["ticker"] for r in watchlist[:8]))

    text = "\n".join(lines)
    if len(text) > 3500:        # Telegram message limit is 4096; leave headroom
        text = text[:3500] + "\n…"

    import urllib.request
    import urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        result = json.loads(body)
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result}")
    return True


def send_email_notification(payload: dict) -> bool:
    """Send the breakout summary via Gmail SMTP.

    Reads credentials from ~/.nasdaq_scanner_email.json:
      {
        "smtp_host":  "smtp.gmail.com",
        "smtp_port":  587,
        "username":   "you@gmail.com",
        "app_password": "xxxxxxxxxxxxxxxx",
        "to":         "you@gmail.com",
        "from_name":  "NASDAQ Scanner"
      }

    The Gmail app password is created at https://myaccount.google.com/apppasswords
    (requires 2FA enabled on the Google account).

    Returns True if sent, False if config is missing.
    """
    if not EMAIL_CONFIG_PATH.exists():
        print(f"(email config not found at {EMAIL_CONFIG_PATH} — skipping)")
        return False

    cfg = json.loads(EMAIL_CONFIG_PATH.read_text())
    required = ["smtp_host", "smtp_port", "username", "app_password", "to"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"(email config missing fields: {missing})")
        return False

    import smtplib
    from email.message import EmailMessage
    from email.utils import formataddr, formatdate

    breakouts = payload.get("breakouts", [])
    watchlist = payload.get("watchlist", [])
    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"📈 NASDAQ Scan {today} — {len(breakouts)} פריצות, {len(watchlist)} במעקב"

    # Plain-text body
    text_lines = [
        f"סריקת NASDAQ — {today}",
        f"נסרקו {payload['scanned']} מניות מתוך {payload['universe_size']}.",
        f"נמצאו {len(breakouts)} פריצות ו-{len(watchlist)} במעקב.",
        "",
    ]
    if breakouts:
        text_lines.append("=" * 60)
        text_lines.append("🚀 פריצות מעניינות")
        text_lines.append("=" * 60)
        for r in breakouts:
            text_lines.append("")
            text_lines.append(f"  {r['ticker']} — ציון {r['score']} — ${r['price']}")
            text_lines.append(f"  +{r['pct_above_ma150']}% מעל MA150 | "
                              f"RSI {r['rsi']} | נפח ×{r.get('vol_ratio') or '—'} | "
                              f"RS {r.get('rs_vs_spy') or '—'}")
            text_lines.append("")
            text_lines.append(f"  {r.get('summary', '')}")
            text_lines.append("-" * 60)
    if watchlist:
        text_lines.append("")
        text_lines.append("=" * 60)
        text_lines.append("👀 רשימת מעקב")
        text_lines.append("=" * 60)
        for r in watchlist:
            text_lines.append(
                f"  {r['ticker']:6}  ${r['price']:>8.2f}  "
                f"+{r['pct_above_ma150']:>5.2f}% MA150  RSI {r['rsi']}  ציון {r['score']}"
            )
    text_lines.append("")
    text_lines.append("---")
    text_lines.append(f"דשבורד מקומי: file://{(HERE / 'dashboard_standalone.html')}")
    text_body = "\n".join(text_lines)

    # HTML body — same content, prettier
    def card_html(r, kind="breakout"):
        emoji = "🚀" if kind == "breakout" else "👀"
        chart_img = ""
        if r.get("chart_b64"):
            chart_img = (
                f'<img src="data:image/png;base64,{r["chart_b64"]}" '
                f'alt="{r["ticker"]} chart" style="width:100%;max-width:520px;'
                f'border:1px solid #e5e7eb;border-radius:6px;margin-top:8px;">'
            )
        return f"""
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin:12px 0;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <h3 style="margin:0;font-size:18px;color:#111;">{emoji} {r['ticker']}</h3>
            <span style="background:#4f46e5;color:white;border-radius:999px;padding:3px 12px;font-size:12px;font-weight:700;">ציון {r['score']}</span>
          </div>
          <div style="margin-top:6px;color:#374151;font-size:14px;">
            <strong>${r['price']}</strong>
            <span style="color:#6b7280;">  ·  +{r['pct_above_ma150']}% מעל MA150  ·  RSI {r['rsi']}  ·  נפח ×{r.get('vol_ratio') or '—'}  ·  RS {r.get('rs_vs_spy') or '—'}</span>
          </div>
          {chart_img}
          <div style="background:#f9fafb;padding:10px 12px;border-radius:8px;border-inline-start:3px solid #4f46e5;margin-top:10px;font-size:13px;line-height:1.6;color:#1f2937;">
            {r.get('summary', '')}
          </div>
        </div>
        """

    breakouts_html = "".join(card_html(r, "breakout") for r in breakouts)
    watchlist_html = ""
    if watchlist:
        rows = "".join(
            f"<tr><td><strong>{r['ticker']}</strong></td>"
            f"<td>${r['price']}</td>"
            f"<td>+{r['pct_above_ma150']}%</td>"
            f"<td>RSI {r['rsi']}</td>"
            f"<td>{r['score']}</td></tr>"
            for r in watchlist
        )
        watchlist_html = f"""
        <h2 style="margin-top:30px;">👀 רשימת מעקב</h2>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <thead style="background:#f3f4f6;">
            <tr><th align="start" style="padding:8px;">טיקר</th>
                <th align="start" style="padding:8px;">מחיר</th>
                <th align="start" style="padding:8px;">מעל MA150</th>
                <th align="start" style="padding:8px;">RSI</th>
                <th align="start" style="padding:8px;">ציון</th></tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """

    html_body = f"""<!doctype html>
    <html dir="rtl"><head><meta charset="utf-8"></head>
    <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Heebo',sans-serif;background:#fafafa;margin:0;padding:24px;color:#1a1a1a;">
      <div style="max-width:680px;margin:0 auto;">
        <h1 style="margin:0 0 4px;">📈 NASDAQ Scan — {today}</h1>
        <p style="color:#666;font-size:14px;">נסרקו {payload['scanned']} מניות מתוך {payload['universe_size']} · {len(breakouts)} פריצות · {len(watchlist)} במעקב</p>
        <h2>🚀 פריצות מעניינות</h2>
        {breakouts_html}
        {watchlist_html}
        <p style="color:#9ca3af;font-size:11px;margin-top:32px;text-align:center;">
          קריטריונים: שווי שוק > $5B · פריצה מעל MA150 ב-5 ימים אחרונים · נפח > ×1.5 ממוצע 50 ימים · RSI 50-75 · Stage 2 alignment
        </p>
      </div>
    </body></html>
    """

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((cfg.get("from_name", "NASDAQ Scanner"), cfg["username"]))
    msg["To"] = cfg["to"]
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as s:
        s.starttls()
        s.login(cfg["username"], cfg["app_password"])
        s.send_message(msg)
    return True


if __name__ == "__main__":
    sys.exit(main())
