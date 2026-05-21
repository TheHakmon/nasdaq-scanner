#!/usr/bin/env python3
"""
Trade Journal & Paper Trading Tracker
======================================
Reads trade_journal.json, fetches current prices for each open position,
auto-closes positions where Stop-Loss or Target was hit, and produces a
status report (HTML + console).

Workflow:
  1. Open trade_journal.json and add trades (paper or real) using the schema.
  2. Run `python3 trade_journal.py` (also runs automatically every night).
  3. Open trade_journal.html to see open positions, P&L, and closed history.

Schema for each trade:
  {
    "id": "unique-id",
    "ticker": "AFRM",
    "kind": "paper" | "real",
    "status": "open" | "closed",
    "entry_date": "2026-05-08",
    "entry_price": 64.01,
    "shares": 47,
    "stop_loss": 58.20,
    "target1": 73.50,
    "target2": 80.10,
    "notes": "...",

    // Auto-filled by this script:
    "exit_date": "...", "exit_price": ..., "exit_reason": "stop"|"target1"|"target2",
    "current_price": ..., "current_pnl": ..., "max_pnl_seen": ..., "min_pnl_seen": ...
  }
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOURNAL_JSON = HERE / "trade_journal.json"
JOURNAL_HTML = HERE / "trade_journal.html"


def _fetch_price_history(ticker: str, period: str = "3mo"):
    import yfinance as yf
    try:
        return yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return None


def _check_exit(trade: dict, hist) -> dict | None:
    """Walk the days since entry. If any day's range hit stop or target,
    record the exit. Returns updated trade dict or None if still open."""
    if hist is None or len(hist) == 0:
        return None
    import pandas as pd
    entry = pd.Timestamp(trade["entry_date"])
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx_local = idx.tz_localize(None)
    else:
        idx_local = idx
    post = hist[idx_local >= entry]
    if len(post) == 0:
        return None
    stop = trade.get("stop_loss")
    t1 = trade.get("target1")
    t2 = trade.get("target2")
    for ts, row in post.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        date_str = ts.strftime("%Y-%m-%d")
        # Stop hit first if both stop and target touched on same day (conservative)
        if stop and low <= stop:
            return {**trade, "status": "closed", "exit_date": date_str,
                    "exit_price": stop, "exit_reason": "stop"}
        if t2 and high >= t2:
            return {**trade, "status": "closed", "exit_date": date_str,
                    "exit_price": t2, "exit_reason": "target2"}
        if t1 and high >= t1:
            return {**trade, "status": "closed", "exit_date": date_str,
                    "exit_price": t1, "exit_reason": "target1"}
    return None


def _compute_running_pnl(trade: dict, hist) -> dict:
    """For an open trade, attach current price and running P&L plus max/min seen."""
    if hist is None or len(hist) == 0:
        return trade
    import pandas as pd
    entry = pd.Timestamp(trade["entry_date"])
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx_local = idx.tz_localize(None)
    else:
        idx_local = idx
    post = hist[idx_local >= entry]
    if len(post) == 0:
        return trade
    last_close = float(post["Close"].iloc[-1])
    last_date = post.index[-1].strftime("%Y-%m-%d")
    entry_price = trade["entry_price"]
    shares = trade["shares"]
    current_pnl_dollars = round((last_close - entry_price) * shares, 2)
    current_pnl_pct = round((last_close / entry_price - 1) * 100, 2)
    max_seen = float(post["High"].max())
    min_seen = float(post["Low"].min())
    return {
        **trade,
        "current_price": round(last_close, 2),
        "current_date": last_date,
        "current_pnl": current_pnl_dollars,
        "current_pnl_pct": current_pnl_pct,
        "max_pnl_pct": round((max_seen / entry_price - 1) * 100, 2),
        "min_pnl_pct": round((min_seen / entry_price - 1) * 100, 2),
    }


def update_all(trades: list) -> list:
    """Iterate every open trade, fetch its price history, and update."""
    out = []
    for t in trades:
        if t.get("status") == "closed":
            out.append(t)
            continue
        hist = _fetch_price_history(t["ticker"])
        # Did stop/target hit?
        closed = _check_exit(t, hist)
        if closed is not None:
            # Fill running P&L of the exit
            entry_price = closed["entry_price"]
            exit_price = closed["exit_price"]
            shares = closed["shares"]
            closed["final_pnl"] = round((exit_price - entry_price) * shares, 2)
            closed["final_pnl_pct"] = round((exit_price / entry_price - 1) * 100, 2)
            out.append(closed)
        else:
            out.append(_compute_running_pnl(t, hist))
    return out


def aggregate(trades: list) -> dict:
    open_trades = [t for t in trades if t.get("status") == "open"]
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    wins = [t for t in closed_trades if t.get("final_pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("final_pnl", 0) <= 0]
    total_realized = sum(t.get("final_pnl", 0) for t in closed_trades)
    total_unrealized = sum(t.get("current_pnl", 0) for t in open_trades)
    win_rate = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else None
    avg_win = round(sum(t["final_pnl_pct"] for t in wins) / len(wins), 2) if wins else None
    avg_loss = round(sum(t["final_pnl_pct"] for t in losses) / len(losses), 2) if losses else None
    return {
        "n_open": len(open_trades),
        "n_closed": len(closed_trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": win_rate,
        "total_realized_pnl": round(total_realized, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_pnl": round(total_realized + total_unrealized, 2),
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
    }


def render_html(trades: list, agg: dict) -> str:
    def row_open(t):
        pnl = t.get("current_pnl", 0)
        pnl_pct = t.get("current_pnl_pct", 0)
        color = "#059669" if pnl > 0 else "#dc2626" if pnl < 0 else "#6b7280"
        kind_badge = "📄 Paper" if t.get("kind") == "paper" else "💰 Real"
        return f"""
        <tr>
          <td><strong>{t['ticker']}</strong> <span style="font-size:11px;color:#6b7280">{kind_badge}</span></td>
          <td>{t['entry_date']}</td>
          <td>${t['entry_price']:.2f} × {t['shares']}</td>
          <td>${t.get('current_price','—')}</td>
          <td style="color:{color};font-weight:600">${pnl:+,.0f}</td>
          <td style="color:{color};font-weight:600">{pnl_pct:+.1f}%</td>
          <td>${t.get('stop_loss','—')}</td>
          <td>${t.get('target1','—')}</td>
          <td>{t.get('notes','')}</td>
        </tr>"""

    def row_closed(t):
        pnl = t.get("final_pnl", 0)
        pnl_pct = t.get("final_pnl_pct", 0)
        color = "#059669" if pnl > 0 else "#dc2626"
        reason_emoji = {"stop": "🛑 Stop", "target1": "🎯 T1", "target2": "🎯 T2"}.get(t.get("exit_reason", ""), t.get("exit_reason", ""))
        return f"""
        <tr>
          <td><strong>{t['ticker']}</strong></td>
          <td>{t['entry_date']} → {t.get('exit_date','—')}</td>
          <td>${t['entry_price']:.2f} → ${t.get('exit_price','—')}</td>
          <td>{t['shares']}</td>
          <td>{reason_emoji}</td>
          <td style="color:{color};font-weight:600">${pnl:+,.0f}</td>
          <td style="color:{color};font-weight:600">{pnl_pct:+.1f}%</td>
        </tr>"""

    open_rows = "".join(row_open(t) for t in trades if t.get("status") == "open")
    closed_rows = "".join(row_closed(t) for t in sorted([t for t in trades if t.get("status") == "closed"], key=lambda x: x.get("exit_date", ""), reverse=True))

    total_pnl_color = "#059669" if agg["total_pnl"] > 0 else "#dc2626" if agg["total_pnl"] < 0 else "#6b7280"

    try:
        from _theme import THEME_HEAD, THEME_TOGGLE_SCRIPT, THEME_TOGGLE_BUTTON
    except Exception:
        THEME_HEAD = THEME_TOGGLE_SCRIPT = ""
        THEME_TOGGLE_BUTTON = ""
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head>
<meta charset="utf-8"><title>יומן מסחר</title>
{THEME_HEAD}
<style>
  body {{ font-family:-apple-system,'Segoe UI','Heebo',sans-serif; margin:0; padding:24px; }}
  .wrap {{ max-width:1200px; margin:0 auto; }}
  .top-nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px; }}
  .nav-links {{ display:flex; gap:8px; align-items:center; }}
  .nav-btn {{ display:inline-flex; gap:6px; padding:8px 14px; border-radius:8px; font-size:13px;
              text-decoration:none; font-weight:500; }}
  h1 {{ margin:0; }}
  .sub {{ color:var(--text-muted); font-size:14px; margin-bottom:24px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:24px; }}
  .stat {{ border-radius:10px; padding:14px 16px; }}
  .stat .label {{ font-size:12px; color:var(--text-muted); }}
  .stat .value {{ font-size:22px; font-weight:700; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:24px; }}
  th, td {{ padding:10px 14px; text-align:start; font-size:14px; }}
  h2 {{ font-size:18px; margin-top:32px; }}
  .empty {{ padding:24px; border-radius:10px; text-align:center;
            border:1px dashed var(--border-strong); }}
  .help {{ padding:12px 14px; border-radius:8px; font-size:13px; margin-top:16px; }}
  code {{ padding:1px 6px; border-radius:4px; font-size:12px; }}
</style></head>
<body><div class="wrap">
<div class="top-nav">
  <h1>📓 יומן מסחר</h1>
  <div class="nav-links">
    <a class="nav-btn" href="dashboard_standalone.html">← חזרה לדשבורד</a>
    {THEME_TOGGLE_BUTTON}
  </div>
</div>
<div class="sub">מעקב אחר עסקאות paper/real, חישוב P&amp;L אוטומטי לפי המחירים הנוכחיים</div>

<div class="stats">
  <div class="stat"><div class="label">פתוחות / סגורות</div><div class="value">{agg['n_open']} / {agg['n_closed']}</div></div>
  <div class="stat"><div class="label">% הצלחה</div><div class="value">{agg['win_rate_pct'] if agg['win_rate_pct'] is not None else '—'}%</div></div>
  <div class="stat"><div class="label">P&amp;L ממומש</div><div class="value" style="color:{'#059669' if agg['total_realized_pnl']>0 else '#dc2626' if agg['total_realized_pnl']<0 else '#6b7280'}">${agg['total_realized_pnl']:+,.0f}</div></div>
  <div class="stat"><div class="label">P&amp;L לא ממומש</div><div class="value" style="color:{'#059669' if agg['total_unrealized_pnl']>0 else '#dc2626' if agg['total_unrealized_pnl']<0 else '#6b7280'}">${agg['total_unrealized_pnl']:+,.0f}</div></div>
  <div class="stat"><div class="label">P&amp;L כולל</div><div class="value" style="color:{total_pnl_color}">${agg['total_pnl']:+,.0f}</div></div>
</div>

<h2>📂 פוזיציות פתוחות</h2>
{f'<table><thead><tr><th>טיקר</th><th>תאריך כניסה</th><th>כניסה</th><th>מחיר נוכחי</th><th>P&L $</th><th>P&L %</th><th>Stop</th><th>Target 1</th><th>הערות</th></tr></thead><tbody>{open_rows}</tbody></table>' if open_rows else '<div class="empty">אין פוזיציות פתוחות. ערוך את trade_journal.json להוסיף.</div>'}

<h2>📜 פוזיציות סגורות</h2>
{f'<table><thead><tr><th>טיקר</th><th>תקופה</th><th>מחיר</th><th>מניות</th><th>סיבת יציאה</th><th>P&L $</th><th>P&L %</th></tr></thead><tbody>{closed_rows}</tbody></table>' if closed_rows else '<div class="empty">אין פוזיציות סגורות עדיין.</div>'}

<div class="help">
  💡 <strong>איך להוסיף עסקה:</strong> ערוך את <code>trade_journal.json</code> והוסף אובייקט תחת <code>trades</code> עם השדות:
  <code>ticker, kind, status:"open", entry_date, entry_price, shares, stop_loss, target1, target2, notes</code>.
  שמור ובהרצה הבאה של <code>trade_journal.py</code> (אוטומטית כל לילה) - היומן יעודכן.
  <br><br>
  ⚠️ הסקריפט סוגר אוטומטית עסקה כשמחיר הנמוך של היום ירד מתחת ל-stop, או הגבוה עלה מעל target. בחישוב שמרני, אם שניהם נפגעו ביום אחד - מקדימים את ה-stop.
</div>

</div>
{THEME_TOGGLE_SCRIPT}
</body></html>"""


def main() -> int:
    if not JOURNAL_JSON.exists():
        print(f"trade_journal.json not found at {JOURNAL_JSON}")
        return 1
    data = json.loads(JOURNAL_JSON.read_text())
    trades = data.get("trades", [])
    print(f"Loaded {len(trades)} trades")
    if not trades:
        print("No trades to track. Add some to trade_journal.json under 'trades'.")
        # Still generate empty HTML so the page exists.
    updated = update_all(trades)

    # Persist updated trades back to JSON
    data["trades"] = updated
    data["_last_update"] = datetime.now().isoformat(timespec="seconds")
    JOURNAL_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Generate HTML report
    agg = aggregate(updated)
    JOURNAL_HTML.write_text(render_html(updated, agg))
    print(f"Wrote {JOURNAL_HTML.name}")
    print(f"  Open: {agg['n_open']}  Closed: {agg['n_closed']}")
    if agg["n_closed"] > 0:
        print(f"  Win-rate: {agg['win_rate_pct']}%  Realized P&L: ${agg['total_realized_pnl']:+,.0f}")
    if agg["n_open"] > 0:
        print(f"  Unrealized: ${agg['total_unrealized_pnl']:+,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
