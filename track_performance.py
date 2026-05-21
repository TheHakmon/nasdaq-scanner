#!/usr/bin/env python3
"""
Performance Tracker
====================
Goes through every archive entry, and for each breakout candidate that was
flagged, fetches what happened to the stock in the days that followed.
Computes returns at 1, 5, 20, and 60 trading days after the signal.

Outputs:
    performance_results.json  — full per-signal data
    performance_summary.html  — dashboard showing aggregate stats and
                                 hit-rate by score band

Use this to answer: "Is my scanner actually finding winners, or noise?"
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_nasdaq as sc

HERE = Path(__file__).resolve().parent
ARCHIVE_DIR = HERE / "archive"
RESULTS_JSON = HERE / "performance_results.json"
RESULTS_HTML = HERE / "performance_summary.html"

HORIZONS = [1, 5, 20, 60]   # trading days
GOOD_THRESHOLD = 5.0         # % gain at horizon 20d is the success criterion


def compute_returns(hist: pd.DataFrame, entry_date: pd.Timestamp) -> dict:
    """Return %change at each horizon, relative to entry close."""
    out = {}
    if hist is None or len(hist) == 0:
        return out
    try:
        # Normalize timezones — yfinance returns tz-aware; our entry_date is tz-naive.
        idx_dates = hist.index
        if getattr(idx_dates, "tz", None) is not None:
            idx_dates = idx_dates.tz_localize(None)
        # Find the first trading day >= entry_date
        mask = np.asarray(idx_dates >= entry_date)
        if not mask.any():
            return out
        idx = int(np.argmax(mask))  # first True
        if idx >= len(hist):
            return out
        entry_price = float(hist["Close"].iloc[idx])
        for h in HORIZONS:
            target_idx = idx + h
            if target_idx >= len(hist):
                continue
            exit_price = float(hist["Close"].iloc[target_idx])
            ret = (exit_price / entry_price - 1) * 100
            out[f"ret_{h}d"] = round(ret, 2)
        # Max drawdown / max gain within 20 days
        if idx + 20 < len(hist):
            window = hist["Close"].iloc[idx:idx + 20]
            out["max_drawdown_20d"] = round(float(window.min() / entry_price - 1) * 100, 2)
            out["max_gain_20d"] = round(float(window.max() / entry_price - 1) * 100, 2)
    except Exception as e:
        out["_error"] = str(e)
    return out


def main() -> int:
    import yfinance as yf

    if not ARCHIVE_DIR.exists():
        print(f"No archive at {ARCHIVE_DIR}. Run backtest_archive.py first.")
        return 1

    archives = sorted(ARCHIVE_DIR.glob("scan_*.json"))
    print(f"Found {len(archives)} archive entries")

    # Collect all (date, ticker, score) triples
    signals = []
    for jf in archives:
        try:
            d = json.loads(jf.read_text())
            date_str = jf.stem.replace("scan_", "")
            for r in d.get("breakouts", []):
                signals.append({
                    "date": date_str,
                    "ticker": r["ticker"],
                    "score": r["score"],
                    "price_at_entry": r["price"],
                    "pct_above_ma150": r.get("pct_above_ma150"),
                    "rsi": r.get("rsi"),
                    "vol_ratio": r.get("vol_ratio"),
                    "rs_vs_spy": r.get("rs_vs_spy"),
                    "is_recent_breakout": True,
                    "kind": "breakout",
                })
            for r in d.get("watchlist", []):
                signals.append({
                    "date": date_str,
                    "ticker": r["ticker"],
                    "score": r["score"],
                    "price_at_entry": r["price"],
                    "pct_above_ma150": r.get("pct_above_ma150"),
                    "rsi": r.get("rsi"),
                    "vol_ratio": r.get("vol_ratio"),
                    "rs_vs_spy": r.get("rs_vs_spy"),
                    "is_recent_breakout": False,
                    "kind": "watchlist",
                })
        except Exception as e:
            print(f"  skip {jf.name}: {e}", file=sys.stderr)

    print(f"Total signals to evaluate: {len(signals)}")

    # Group by ticker to minimize yfinance calls
    tickers = sorted(set(s["ticker"] for s in signals))
    print(f"Unique tickers: {len(tickers)}")

    histories = {}
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0:
            print(f"  Fetched {i}/{len(tickers)}")
        try:
            h = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
            if h is not None and len(h) > 0:
                histories[ticker] = h
        except Exception:
            pass

    # Compute returns for each signal
    for s in signals:
        h = histories.get(s["ticker"])
        if h is None:
            continue
        entry_date = pd.Timestamp(s["date"])
        s.update(compute_returns(h, entry_date))

    # --- Aggregate stats ---
    df = pd.DataFrame(signals)
    summary = {"total_signals": len(df), "by_horizon": {}}
    for h in HORIZONS:
        col = f"ret_{h}d"
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        summary["by_horizon"][f"{h}d"] = {
            "n": int(len(valid)),
            "win_rate": round((valid > 0).mean() * 100, 1),
            "avg_return": round(valid.mean(), 2),
            "median_return": round(valid.median(), 2),
            "best": round(valid.max(), 2),
            "worst": round(valid.min(), 2),
            "stdev": round(valid.std(), 2),
        }

    # Hit-rate by score band (using 20d return)
    by_score = {}
    if "ret_20d" in df.columns:
        for band, (low, high) in [("50-59", (50, 60)), ("60-69", (60, 70)),
                                   ("70-79", (70, 80)), ("80+", (80, 200))]:
            sub = df[(df["score"] >= low) & (df["score"] < high) & df["ret_20d"].notna()]
            if len(sub) == 0:
                continue
            by_score[band] = {
                "n": int(len(sub)),
                "win_rate": round((sub["ret_20d"] > 0).mean() * 100, 1),
                "good_rate": round((sub["ret_20d"] >= GOOD_THRESHOLD).mean() * 100, 1),
                "avg_return_20d": round(sub["ret_20d"].mean(), 2),
            }
    summary["by_score_band"] = by_score

    # Top winners and losers
    if "ret_20d" in df.columns:
        winners = df.dropna(subset=["ret_20d"]).nlargest(10, "ret_20d")
        losers = df.dropna(subset=["ret_20d"]).nsmallest(10, "ret_20d")
        summary["top_winners"] = winners[["date", "ticker", "score", "price_at_entry", "ret_20d"]].to_dict(orient="records")
        summary["top_losers"] = losers[["date", "ticker", "score", "price_at_entry", "ret_20d"]].to_dict(orient="records")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signals": signals,
        "summary": summary,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                      default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
    print(f"\nWrote {RESULTS_JSON.name}")

    # Build summary HTML
    horizon_rows = ""
    for hkey, st in summary["by_horizon"].items():
        win_color = "#059669" if st["win_rate"] >= 50 else "#dc2626"
        avg_color = "#059669" if st["avg_return"] > 0 else "#dc2626"
        horizon_rows += f"""
        <tr>
          <td><strong>{hkey}</strong></td>
          <td style="color:{win_color};font-weight:600">{st['win_rate']}%</td>
          <td style="color:{avg_color};font-weight:600">{st['avg_return']:+.2f}%</td>
          <td>{st['median_return']:+.2f}%</td>
          <td>{st['best']:+.2f}%</td>
          <td>{st['worst']:+.2f}%</td>
          <td>{st['n']}</td>
        </tr>
        """

    score_rows = ""
    for band, st in by_score.items():
        win_color = "#059669" if st["win_rate"] >= 50 else "#dc2626"
        good_color = "#059669" if st["good_rate"] >= 30 else "#9ca3af"
        score_rows += f"""
        <tr>
          <td><strong>{band}</strong></td>
          <td style="color:{win_color};font-weight:600">{st['win_rate']}%</td>
          <td style="color:{good_color}">{st['good_rate']}%</td>
          <td>{st['avg_return_20d']:+.2f}%</td>
          <td>{st['n']}</td>
        </tr>
        """

    def stock_rows(items, color):
        rows = ""
        for r in items:
            rows += f"""
            <tr>
              <td>{r['date']}</td>
              <td><strong>{r['ticker']}</strong></td>
              <td>{r['score']}</td>
              <td>${r['price_at_entry']}</td>
              <td style="color:{color};font-weight:600">{r['ret_20d']:+.2f}%</td>
            </tr>
            """
        return rows

    winners_html = stock_rows(summary.get("top_winners", []), "#059669")
    losers_html = stock_rows(summary.get("top_losers", []), "#dc2626")

    # Pre-compute summary stats safely (avoid dict-in-f-string trickery)
    h20 = summary.get("by_horizon", {}).get("20d", {})
    win20 = h20.get("win_rate")
    avg20 = h20.get("avg_return")
    win20_color = "#059669" if isinstance(win20, (int, float)) and win20 >= 50 else "#dc2626"
    avg20_color = "#059669" if isinstance(avg20, (int, float)) and avg20 > 0 else "#dc2626"
    win20_disp = f"{win20}%" if win20 is not None else "—"
    avg20_disp = f"{avg20}%" if avg20 is not None else "—"

    try:
        from _theme import THEME_HEAD, THEME_TOGGLE_SCRIPT, THEME_TOGGLE_BUTTON
    except Exception:
        THEME_HEAD = THEME_TOGGLE_SCRIPT = ""
        THEME_TOGGLE_BUTTON = ""
    html = f"""<!doctype html>
<html lang="he" dir="rtl"><head>
<meta charset="utf-8"><title>ניתוח ביצועי הסקנר</title>
{THEME_HEAD}
<style>
  body {{ font-family:-apple-system,'Segoe UI','Heebo',sans-serif; margin:0; padding:24px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  .top-nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; gap:8px; flex-wrap:wrap; }}
  .nav-links {{ display:flex; gap:8px; align-items:center; }}
  .nav-btn {{ display:inline-flex; gap:6px; padding:8px 14px; border-radius:8px; font-size:13px;
              text-decoration:none; font-weight:500; }}
  h1 {{ margin:0 0 4px; }}
  .sub {{ color:var(--text-muted); font-size:14px; margin-bottom:24px; }}
  h2 {{ margin-top:32px; font-size:18px; }}
  table {{ width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-top:12px; }}
  th, td {{ padding:10px 14px; text-align:start; font-size:14px; }}
  .stat {{ border-radius:10px; padding:14px 16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:16px; }}
  .label {{ font-size:12px; color:var(--text-muted); }}
  .value {{ font-size:24px; font-weight:700; margin-top:4px; }}
  .note {{ padding:12px 14px; border-radius:8px; font-size:13px; margin-top:16px; }}
</style></head>
<body><div class="wrap">
<div class="top-nav">
  <h1>📊 ניתוח ביצועי הסקנר</h1>
  <div class="nav-links">
    <a class="nav-btn" href="dashboard_standalone.html">← חזרה לדשבורד</a>
    {THEME_TOGGLE_BUTTON}
  </div>
</div>
<div class="sub">סטטיסטיקה אמיתית: מה קרה למניות שהסקנר זיהה, בפועל</div>

<div class="grid">
  <div class="stat"><div class="label">סך כל ההמלצות</div><div class="value">{summary['total_signals']}</div></div>
  <div class="stat"><div class="label">% מנצחות (20 יום)</div><div class="value" style="color:{win20_color}">{win20_disp}</div></div>
  <div class="stat"><div class="label">תשואה ממוצעת (20 יום)</div><div class="value" style="color:{avg20_color}">{avg20_disp}</div></div>
</div>

<h2>ביצועים לפי מסגרת זמן</h2>
<table>
  <thead><tr><th>אופק</th><th>% מנצחות</th><th>תשואה ממוצעת</th><th>תשואה חציונית</th><th>הכי טובה</th><th>הכי גרועה</th><th>n</th></tr></thead>
  <tbody>{horizon_rows}</tbody>
</table>

<h2>ביצועים לפי ציון הסקנר (אופק 20 יום)</h2>
<table>
  <thead><tr><th>ציון</th><th>% רווח</th><th>% רווח של 5%+</th><th>תשואה ממוצעת</th><th>n</th></tr></thead>
  <tbody>{score_rows}</tbody>
</table>
<div class="note">
  אם רואים שמניות עם ציון 70+ יש להן win-rate הרבה יותר גבוה - זה אומר שכדאי להתמקד רק בפריצות עם ציון גבוה.
</div>

<h2>🏆 הטובות ביותר (אופק 20 יום)</h2>
<table>
  <thead><tr><th>תאריך</th><th>טיקר</th><th>ציון</th><th>מחיר כניסה</th><th>תשואה</th></tr></thead>
  <tbody>{winners_html}</tbody>
</table>

<h2>📉 הגרועות ביותר (אופק 20 יום)</h2>
<table>
  <thead><tr><th>תאריך</th><th>טיקר</th><th>ציון</th><th>מחיר כניסה</th><th>תשואה</th></tr></thead>
  <tbody>{losers_html}</tbody>
</table>

</div>
{THEME_TOGGLE_SCRIPT}
</body></html>
    """
    RESULTS_HTML.write_text(html)
    print(f"Wrote {RESULTS_HTML.name}")

    # ---- Augment today's scan_results.json with historical track-record  ----
    augment_scan_results_with_history(signals)

    return 0


def _classify_track_record(stats: dict) -> dict:
    """Return a verdict label and color based on aggregate historical stats."""
    appearances = stats["appearances"]
    win_rate = stats.get("win_rate_20d")
    avg_return = stats.get("avg_return_20d")

    if appearances < 2 or win_rate is None or avg_return is None:
        return {"verdict": "unknown", "label": "ראשונה",
                "color": "#9ca3af", "emoji": "🆕"}
    if win_rate >= 75 and avg_return >= 10:
        return {"verdict": "strong", "label": "טרק רקורד חזק",
                "color": "#059669", "emoji": "💎"}
    if win_rate >= 60 and avg_return >= 5:
        return {"verdict": "good", "label": "ביצועים טובים",
                "color": "#10b981", "emoji": "✅"}
    if win_rate >= 50 and avg_return >= 0:
        return {"verdict": "moderate", "label": "מעורב",
                "color": "#f59e0b", "emoji": "⚠️"}
    return {"verdict": "weak", "label": "ביצועים חלשים",
            "color": "#dc2626", "emoji": "❌"}


def _build_track_record(ticker_signals: list) -> dict:
    """Aggregate one ticker's historical performance across all archive
    appearances. Only counts signals that have a completed 20-day return."""
    completed = [s for s in ticker_signals if "ret_20d" in s and s["ret_20d"] is not None]
    out = {
        "appearances": len(ticker_signals),
        "appearances_with_returns": len(completed),
    }
    if completed:
        rets = [s["ret_20d"] for s in completed]
        out["wins_20d"] = sum(1 for r in rets if r > 0)
        out["win_rate_20d"] = round(out["wins_20d"] / len(rets) * 100, 1)
        out["avg_return_20d"] = round(sum(rets) / len(rets), 2)
        out["best_return"] = round(max(rets), 2)
        out["worst_return"] = round(min(rets), 2)
        # List the most recent 3 appearances with their returns
        sorted_completed = sorted(completed, key=lambda x: x["date"], reverse=True)
        out["history"] = [
            {"date": s["date"], "score": s["score"], "ret_20d": s["ret_20d"]}
            for s in sorted_completed[:3]
        ]
    out.update(_classify_track_record(out))
    return out


def augment_scan_results_with_history(all_signals: list) -> None:
    """Add a `historical_performance` field to each result in today's
    scan_results.json based on the archive of past signals."""
    scan_path = HERE / "scan_results.json"
    if not scan_path.exists():
        print("(no scan_results.json — skipping history augmentation)")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    # Group signals by ticker, EXCLUDING today's signals (so we measure
    # past performance only).
    by_ticker = defaultdict(list)
    for s in all_signals:
        if s["date"] >= today:
            continue
        by_ticker[s["ticker"]].append(s)

    data = json.loads(scan_path.read_text())
    augmented = 0
    for bucket in ("breakouts", "watchlist", "personal", "all_results"):
        for r in data.get(bucket, []):
            tr = _build_track_record(by_ticker.get(r["ticker"], []))
            r["historical_performance"] = tr
            if tr["appearances"] > 0:
                augmented += 1

    scan_path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
    print(f"Augmented {augmented} results with historical track-record.")

    # Rebuild the standalone dashboard to bake in the new data.
    rebuild_standalone_dashboard(data)


def rebuild_standalone_dashboard(payload: dict) -> None:
    """Regenerate dashboard_standalone.html using the updated scan_results."""
    tpl_path = HERE / "nasdaq_dashboard.html"
    if not tpl_path.exists():
        return
    tpl = tpl_path.read_text()
    inline = "const __INLINE_DATA__ = " + json.dumps(
        payload, ensure_ascii=False,
        default=lambda o: float(o) if hasattr(o, 'item') else str(o)
    ) + ";"
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
    if old_block not in tpl:
        print("(template marker missing — couldn't rebuild standalone)")
        return
    tpl = tpl.replace(old_block, new_block)
    tpl = tpl.replace(
        'const RESULTS_PATH = "/sessions/vigilant-zen-mendel/mnt/outputs/scan_results.json";',
        inline + '\nconst RESULTS_PATH = "(inline)";'
    )
    (HERE / "dashboard_standalone.html").write_text(tpl)
    print("Rebuilt dashboard_standalone.html with historical badges.")


if __name__ == "__main__":
    sys.exit(main())
