#!/usr/bin/env python3
"""
NASDAQ Scanner — Backtest Archive Builder
==========================================
For each weekday in the past N days (default 30), reconstructs what the
breakout scanner would have flagged AS OF that date — using only data that
was available up to (and including) that date — and saves the results to:

    archive/scan_YYYY-MM-DD.json   — full structured archive entry
    archive/dashboard_YYYY-MM-DD.html  — standalone HTML view for that day
    archive/index.html              — chronological browser of all archived days

This is a one-shot run. Future daily runs (scan_nasdaq.py) automatically
append to the archive too.

Usage:
    python3 backtest_archive.py            # default: 90 days back, with charts
    python3 backtest_archive.py 30         # 30 days back
    python3 backtest_archive.py 90 nocharts # 90 days back, no charts (faster)
"""

from __future__ import annotations
import base64
import io
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared functions/config from the main scanner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_nasdaq as sc

HERE = Path(__file__).resolve().parent
ARCHIVE_DIR = HERE / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)


def analyze_at_cutoff(ticker: str, hist: pd.DataFrame, bench_hist: pd.DataFrame,
                      cutoff_idx: int, make_charts: bool = True) -> dict | None:
    """Run the same analysis as the live scanner but using only data up to
    `cutoff_idx` (inclusive). cutoff_idx is a positional index into hist."""
    try:
        sub = hist.iloc[: cutoff_idx + 1]
        if len(sub) < 160:
            return None
        close = sub["Close"]
        volume = sub["Volume"]

        ma150 = sc.sma(close, 150)
        ma50 = sc.sma(close, 50)
        ma200 = sc.sma(close, 200)
        rsi14 = sc.rsi(close, 14)
        vol_avg50 = volume.rolling(50, min_periods=20).mean()

        last = close.iloc[-1]
        last_vol = volume.iloc[-1]
        last_ma150 = ma150.iloc[-1]
        last_ma50 = ma50.iloc[-1]
        last_ma200 = ma200.iloc[-1]
        last_rsi = rsi14.iloc[-1]
        last_vol_avg = vol_avg50.iloc[-1]

        if pd.isna(last_ma150) or pd.isna(last_ma200):
            return None

        pct_above_ma150 = (last / last_ma150 - 1) * 100
        pct_above_ma50 = (last / last_ma50 - 1) * 100
        vol_ratio = last_vol / last_vol_avg if last_vol_avg else float("nan")

        recent_above = (close.iloc[-sc.BREAKOUT_LOOKBACK_DAYS:] >
                        ma150.iloc[-sc.BREAKOUT_LOOKBACK_DAYS:]).all()
        prior_below = (close.iloc[-sc.BREAKOUT_LOOKBACK_DAYS - 5: -sc.BREAKOUT_LOOKBACK_DAYS] <
                       ma150.iloc[-sc.BREAKOUT_LOOKBACK_DAYS - 5: -sc.BREAKOUT_LOOKBACK_DAYS]).any()
        is_recent_breakout = bool(recent_above and prior_below)

        stage2 = bool(last > last_ma50 and last_ma50 > last_ma150 and last_ma150 > last_ma200)
        ma200_rising = bool(ma200.iloc[-1] > ma200.iloc[-21]) if len(ma200) >= 21 else False
        ma50_above_ma150 = bool(last_ma50 > last_ma150)
        golden_cross_recent = bool(
            (ma50.iloc[-21:] > ma150.iloc[-21:]).iloc[-1]
            and not (ma50.iloc[-42:-21] > ma150.iloc[-42:-21]).all()
        ) if len(ma50) >= 42 else False

        bench_sub = bench_hist["Close"].loc[bench_hist.index <= sub.index[-1]]
        rs = sc.relative_strength(close, bench_sub)

        # Score (identical logic to scan_nasdaq)
        score, reasons = 0, []
        if last > last_ma150:
            score += 20
            reasons.append("מעל MA150")
        if is_recent_breakout:
            score += 25
            reasons.append("פריצה טרייה של MA150")
        if vol_ratio and vol_ratio >= sc.VOLUME_MULTIPLIER:
            score += 15
            reasons.append(f"נפח גבוה (×{vol_ratio:.1f})")
        if stage2:
            score += 15
            reasons.append("Stage 2 alignment")
        if sc.RSI_LOW <= last_rsi <= sc.RSI_HIGH:
            score += 10
            reasons.append(f"RSI={last_rsi:.0f}")
        elif last_rsi > sc.RSI_HIGH:
            reasons.append(f"RSI גבוה ({last_rsi:.0f})")
        if rs and rs > 1.0:
            score += 10
            reasons.append(f"RS={rs:.2f} מעל S&P")
        if golden_cross_recent:
            score += 5
            reasons.append("Golden cross")

        passed = (last > last_ma150 and (not vol_ratio or vol_ratio >= 1.0) and last_rsi < 80)

        # Reuse the main script's summary by temporarily mocking the locals
        # it depends on — quicker than duplicating 80 lines.
        # We just regenerate the simpler version here:
        summary_parts = []
        if is_recent_breakout:
            summary_parts.append(
                f"חצתה את MA150 לאחרונה (כעת {pct_above_ma150:+.1f}% מעליו)."
            )
        elif last > last_ma150:
            summary_parts.append(f"מעל MA150 ב-{pct_above_ma150:+.1f}%.")
        if vol_ratio and vol_ratio >= sc.VOLUME_MULTIPLIER:
            summary_parts.append(f"נפח ×{vol_ratio:.1f} מהממוצע — אישור חזק.")
        if stage2:
            summary_parts.append("Stage 2: מחיר>MA50>MA150>MA200.")
        if sc.RSI_LOW <= last_rsi <= sc.RSI_HIGH:
            summary_parts.append(f"RSI {last_rsi:.0f} בריא.")
        elif last_rsi > sc.RSI_HIGH:
            summary_parts.append(f"⚠️ RSI {last_rsi:.0f} - קניית יתר.")
        if rs and rs > 1.1:
            summary_parts.append(f"חוזק יחסי {rs:.2f} מעל S&P.")
        if score >= 70:
            summary_parts.append("✅ מועמדת לכניסה.")
        elif score >= 60:
            summary_parts.append("🔶 על הגבול.")
        elif is_recent_breakout:
            summary_parts.append("🟡 פריצה ראשונית.")

        chart_b64 = None
        if make_charts and passed:
            chart_b64 = sc.make_chart_b64(ticker, sub, ma50, ma150, ma200)

        return {
            "ticker": ticker,
            "passed": passed,
            "score": score,
            "summary": " ".join(summary_parts),
            "chart_b64": chart_b64,
            "price": round(float(last), 2),
            "ma150": round(float(last_ma150), 2),
            "ma50": round(float(last_ma50), 2),
            "ma200": round(float(last_ma200), 2),
            "pct_above_ma150": round(float(pct_above_ma150), 2),
            "pct_above_ma50": round(float(pct_above_ma50), 2),
            "rsi": round(float(last_rsi), 1),
            "vol_ratio": round(float(vol_ratio), 2) if vol_ratio else None,
            "rs_vs_spy": round(float(rs), 2) if rs and not pd.isna(rs) else None,
            "is_recent_breakout": is_recent_breakout,
            "stage2": stage2,
            "ma200_rising": ma200_rising,
            "ma50_above_ma150": ma50_above_ma150,
            "golden_cross_recent": golden_cross_recent,
            "reasons": reasons,
        }
    except Exception as e:
        return None


def build_dated_dashboard(payload: dict, date_str: str) -> str:
    """Generate a standalone HTML view for a given archive entry, using the
    template at nasdaq_dashboard.html with the data inlined."""
    tpl_path = HERE / "nasdaq_dashboard.html"
    tpl = tpl_path.read_text()
    inline = "const __INLINE_DATA__ = " + json.dumps(
        payload, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, 'item') else str(o)
    ) + ";"
    old = ('async function load() {\n'
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
           '    const data = JSON.parse(text);')
    new = ('async function load() {\n'
           '  const errArea = document.getElementById("error-area");\n'
           '  errArea.innerHTML = "";\n'
           '  try {\n'
           '    const data = __INLINE_DATA__;')
    tpl = tpl.replace(old, new)
    tpl = tpl.replace(
        'const RESULTS_PATH = "/sessions/vigilant-zen-mendel/mnt/outputs/scan_results.json";',
        inline + '\nconst RESULTS_PATH = "(inline)";'
    )
    tpl = tpl.replace("<title>NASDAQ Breakout Scanner</title>",
                      f"<title>NASDAQ Scan — {date_str}</title>")
    return tpl


def build_archive_index(entries: list[dict]) -> str:
    """Build an index.html that lists all archived scans chronologically."""
    try:
        from _theme import THEME_HEAD, THEME_TOGGLE_SCRIPT, THEME_TOGGLE_BUTTON
    except Exception:
        THEME_HEAD = THEME_TOGGLE_SCRIPT = ""
        THEME_TOGGLE_BUTTON = ""

    rows = []
    for e in sorted(entries, key=lambda x: x["date"], reverse=True):
        bcount = e["breakouts"]
        wcount = e["watchlist"]
        rows.append(f"""
        <tr onclick="location.href='dashboard_{e['date']}.html'" style="cursor:pointer">
          <td><strong>{e['date']}</strong> <span style="color:var(--text-muted);font-size:11px;">({e['weekday']})</span></td>
          <td style="text-align:center;color:{'var(--success)' if bcount>0 else 'var(--text-subtle)'};">{bcount}</td>
          <td style="text-align:center;color:{'var(--warning)' if wcount>0 else 'var(--text-subtle)'};">{wcount}</td>
          <td style="font-size:12px;color:var(--text-muted);">{e['top_tickers']}</td>
          <td><a href="dashboard_{e['date']}.html" style="color:var(--accent);">פתח →</a></td>
        </tr>""")
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head>
<meta charset="utf-8"><title>NASDAQ Archive</title>
{THEME_HEAD}
<style>
  body {{ font-family:-apple-system,'Segoe UI','Heebo',sans-serif; margin:0; padding:24px; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  .top-nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; gap:8px; flex-wrap:wrap; }}
  .nav-links {{ display:flex; gap:8px; align-items:center; }}
  .nav-btn {{ display:inline-flex; gap:6px; padding:8px 14px; border-radius:8px; font-size:13px; text-decoration:none; font-weight:500; }}
  h1 {{ margin:0 0 4px; }}
  .sub {{ color:var(--text-muted); font-size:14px; margin-bottom:24px; }}
  table {{ width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding:10px 14px; text-align:start; font-size:14px; }}
  a {{ text-decoration:none; }}
</style></head>
<body><div class="wrap">
<div class="top-nav">
  <h1>📚 ארכיון סריקות NASDAQ</h1>
  <div class="nav-links">
    <a class="nav-btn" href="../dashboard_standalone.html">← חזרה לדשבורד</a>
    {THEME_TOGGLE_BUTTON}
  </div>
</div>
<div class="sub">{len(entries)} סריקות יומיות. לחץ על שורה לפתיחת הסריקה של אותו יום.</div>
<table>
  <thead><tr><th>תאריך</th><th style="text-align:center">🚀 פריצות</th><th style="text-align:center">👀 מעקב</th><th>טיקרים מובילים</th><th></th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
{THEME_TOGGLE_SCRIPT}
</body></html>"""


def main() -> int:
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    make_charts = "nocharts" not in sys.argv[1:]
    print(f"Building backtest archive for last {days_back} days. Charts={make_charts}")

    universe = json.loads(sc.UNIVERSE_PATH.read_text())["tickers"]
    universe = sorted(set(universe))
    print(f"Universe: {len(universe)} tickers")

    import yfinance as yf

    # Fetch benchmark once — use 2y to ensure we have enough history when going
    # back 90 days (need 200 trading days before the oldest cutoff for MA200).
    fetch_period = "2y" if days_back > 30 else "1y"
    print(f"Fetching benchmark {sc.BENCHMARK} (period={fetch_period})...")
    bench_hist = yf.Ticker(sc.BENCHMARK).history(period=fetch_period, auto_adjust=True)

    # Determine cutoff trading dates: last `days_back` weekdays present in
    # the benchmark history (so we only use real trading days).
    available_dates = bench_hist.index.tolist()
    cutoff_dates = available_dates[-days_back:]
    print(f"Will archive {len(cutoff_dates)} trading days "
          f"({cutoff_dates[0].date()} to {cutoff_dates[-1].date()})")

    # Pre-fetch all ticker histories.
    histories = {}
    for i, ticker in enumerate(universe, 1):
        if i % 20 == 0:
            print(f"  Fetched {i}/{len(universe)}...")
        try:
            h = yf.Ticker(ticker).history(period=fetch_period, auto_adjust=True)
            if h is not None and len(h) >= 160:
                histories[ticker] = h
        except Exception as e:
            print(f"  skip {ticker}: {e}", file=sys.stderr)
        time.sleep(0.04)
    print(f"Fetched {len(histories)} usable ticker histories")

    # Build one archive entry per cutoff date
    index_entries = []
    for cutoff_ts in cutoff_dates:
        date_str = cutoff_ts.strftime("%Y-%m-%d")
        weekday_name = cutoff_ts.strftime("%a")
        print(f"\n=== Processing {date_str} ({weekday_name}) ===")

        results = []
        for ticker, h in histories.items():
            # Map cutoff date to positional index in this ticker's history
            try:
                idx_pos = h.index.get_indexer([cutoff_ts], method="nearest")[0]
                if idx_pos < 160:
                    continue
                r = analyze_at_cutoff(ticker, h, bench_hist, idx_pos, make_charts)
                if r is None:
                    continue
                results.append(r)
            except Exception as e:
                continue

        results.sort(key=lambda r: (not r["passed"], -r["score"]))
        breakouts = [r for r in results if r["passed"] and r["is_recent_breakout"] and r["score"] >= 50]
        watchlist = [r for r in results if r["passed"] and r["score"] >= 60 and r not in breakouts]

        all_lite = [{k: v for k, v in r.items() if k != "chart_b64"} for r in results]

        payload = {
            "generated_at": cutoff_ts.isoformat(),
            "as_of_date": date_str,
            "universe_size": len(universe),
            "scanned": len(results),
            "errors": 0,
            "criteria": {
                "min_market_cap_usd": sc.MARKET_CAP_MIN,
                "breakout_lookback_days": sc.BREAKOUT_LOOKBACK_DAYS,
                "min_volume_multiplier": sc.VOLUME_MULTIPLIER,
                "rsi_band": [sc.RSI_LOW, sc.RSI_HIGH],
            },
            "breakouts": breakouts,
            "watchlist": watchlist,
            "all_results": all_lite,
            "error_tickers": [],
        }

        # Save JSON
        json_path = ARCHIVE_DIR / f"scan_{date_str}.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                       default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
        # Save standalone HTML
        html_path = ARCHIVE_DIR / f"dashboard_{date_str}.html"
        html_path.write_text(build_dated_dashboard(payload, date_str))

        top_tickers = ", ".join([r["ticker"] for r in breakouts[:5]]) or "—"
        print(f"  {len(breakouts)} breakouts, {len(watchlist)} watchlist. Top: {top_tickers}")

        index_entries.append({
            "date": date_str,
            "weekday": weekday_name,
            "breakouts": len(breakouts),
            "watchlist": len(watchlist),
            "top_tickers": top_tickers,
        })

    # Build index
    (ARCHIVE_DIR / "index.html").write_text(build_archive_index(index_entries))
    print(f"\nWrote archive/index.html with {len(index_entries)} entries")
    print(f"Open: file://{(ARCHIVE_DIR / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
