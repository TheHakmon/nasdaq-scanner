#!/usr/bin/env python3
"""
NASDAQ Scanner — Rule-based backtest of archived signals
=========================================================
track_performance.py answers "what was the return after N days?".
This answers "what if I had actually traded every breakout by fixed rules?".

For every breakout in archive/scan_*.json:
  - entry at the next trading day's open
  - one open trade per ticker at a time (repeat signals while in a trade are skipped)
Three exit styles, each simulated separately:
  A "scanner levels"  : stop = close − 2×ATR, target = close + 4×ATR (the dashboard's levels)
  B "MA150 (Micha)"   : exit next open after a close below MA150; protective stop 3% under MA150
  C "ATR stop + MA150": stop = close − 2×ATR, exit on close below MA150, no fixed target
Trades still open today are marked to market (flagged "open").
Splits: score band, earnings within 7 days (when known), market status at signal.

Outputs: backtest_signals.json, backtest_signals.html
Usage:   python3 backtest_signals.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "archive"
OUT_JSON = HERE / "backtest_signals.json"
OUT_HTML = HERE / "backtest_signals.html"

ATR_STOP_MULT, ATR_TARGET_MULT = 2.0, 4.0     # same as scan_nasdaq.py
MA_LEN, MA_STOP_PCT = 150, 0.03
EARNINGS_DAYS = 7
COST = 0.001                                  # fee + slippage per side
STYLES = {"A": "רמות הסקנר (סטופ 2ATR, יעד 4ATR)",
          "B": "ממוצע 150 של מיכה",
          "C": "סטופ 2ATR + יציאה מתחת ל-MA150"}


def load_signals() -> pd.DataFrame:
    rows = []
    for f in sorted(ARCHIVE.glob("scan_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        # The file name is the day the scan RAN (often after midnight UTC), not the trading day it
        # describes. Keep the run time; the signal bar is the last bar completed before it.
        gen = d.get("generated_at")
        ran = pd.Timestamp(gen) if gen else pd.Timestamp(f.stem.replace("scan_", ""))
        if ran.tzinfo is not None:
            ran = ran.tz_convert("UTC").tz_localize(None)
        for r in d.get("breakouts") or []:
            rows.append({"ran": ran, "ticker": r["ticker"], "score": r.get("score"),
                         "days_to_earnings": r.get("days_to_earnings")})
    return pd.DataFrame(rows).sort_values(["ran", "ticker"]).reset_index(drop=True)


def signal_bar(index: pd.DatetimeIndex, ran: pd.Timestamp) -> int:
    """Position of the last daily bar that had closed (≈20:00 UTC) when the scan ran."""
    return int(index.searchsorted(ran - pd.Timedelta(hours=20), side="right")) - 1


def fetch_prices(tickers, start) -> dict[str, pd.DataFrame]:
    df = yf.download(sorted(set(tickers) | {"SPY"}), start=start, auto_adjust=True,
                     group_by="ticker", threads=True, progress=False)
    out = {}
    for t in set(tickers) | {"SPY"}:
        if t in df.columns.get_level_values(0):
            h = df[t].dropna(subset=["Close"])
            if len(h) > MA_LEN:
                prev = h["Close"].shift(1)
                tr = pd.concat([h["High"] - h["Low"], (h["High"] - prev).abs(), (h["Low"] - prev).abs()], axis=1).max(axis=1)
                h = h.assign(atr=tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(),
                             ma=h["Close"].rolling(MA_LEN).mean())
                out[t] = h
    return out


def simulate(sig, h, style):
    """Return one trade dict for a signal, or None if it can't be entered."""
    idx = h.index
    k = signal_bar(idx, sig.ran) + 1                   # first bar after the signal day
    if k >= len(idx) or k <= 0:
        return None
    s = h.iloc[k - 1]                                  # signal-day bar
    entry = float(h["Open"].iloc[k]) * (1 + COST)
    if style in ("A", "C"):
        stop = float(s["Close"] - ATR_STOP_MULT * s["atr"])
    else:
        stop = float(s["ma"] * (1 - MA_STOP_PCT))
    target = float(s["Close"] + ATR_TARGET_MULT * s["atr"]) if style == "A" else None
    risk = entry - stop
    if not risk > 0:
        return None
    exit_px = exit_i = None
    reason = "open"
    pending_exit = False
    for i in range(k, len(idx)):
        o, hi, lo, c, ma = (float(h[x].iloc[i]) for x in ("Open", "High", "Low", "Close", "ma"))
        if pending_exit:
            exit_px, exit_i, reason = o, i, "close<MA150"
            break
        if lo <= stop:                                 # stop first when both hit (conservative)
            exit_px, exit_i, reason = min(o, stop), i, "stop"   # gap below stop fills at the open
            break
        if target and hi >= target:
            exit_px, exit_i, reason = max(o, target), i, "target"
            break
        if style == "B":
            stop = max(stop, ma * (1 - MA_STOP_PCT))   # never moves down
        if style in ("B", "C") and c < ma:
            pending_exit = True
    if exit_px is None:
        exit_i, exit_px = len(idx) - 1, float(h["Close"].iloc[-1])
    else:
        exit_px *= (1 - COST)
    return {"entry_date": idx[k], "exit_date": idx[exit_i], "entry": entry, "exit": exit_px,
            "ret": exit_px / entry - 1, "r": (exit_px - entry) / risk, "reason": reason,
            "days": int(exit_i - k)}


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    closed = tr[tr.reason != "open"]
    w, l = tr[tr.ret > 0], tr[tr.ret <= 0]
    return {"n": len(tr), "closed": len(closed), "win_rate": round(len(w) / len(tr) * 100, 1),
            "avg_ret": round(tr.ret.mean() * 100, 2), "median_ret": round(tr.ret.median() * 100, 2),
            "avg_r": round(tr.r.mean(), 2), "avg_win": round(w.ret.mean() * 100, 2) if len(w) else None,
            "avg_loss": round(l.ret.mean() * 100, 2) if len(l) else None,
            "profit_factor": round(w.ret.sum() / -l.ret.sum(), 2) if len(l) and l.ret.sum() < 0 else None,
            "avg_days": round(tr.days.mean(), 1), "spy_avg_ret": round(tr.spy_ret.mean() * 100, 2),
            "edge_vs_spy": round((tr.ret - tr.spy_ret).mean() * 100, 2)}


def market_state(spy: pd.DataFrame, ran) -> str:
    k = signal_bar(spy.index, ran)
    if k < 20:
        return "unknown"
    c, ma, ma20 = spy["Close"].iloc[k], spy["ma"].iloc[k], spy["ma"].iloc[k - 20]
    above, rising = c > ma, (ma / ma20 - 1) * 100 >= -0.5
    return "positive" if above and rising else ("negative" if not above and not rising else "mixed")


def main() -> int:
    sig = load_signals()
    if sig.empty:
        print("no archived breakouts")
        return 0
    start = (sig.ran.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    print(f"{len(sig)} breakout signals, fetching prices...")
    px = fetch_prices(sig.ticker.unique().tolist(), start)
    spy = px["SPY"]
    sig["date"] = [spy.index[signal_bar(spy.index, r)] for r in sig.ran]   # the trading day the signal describes
    sig["market"] = [market_state(spy, r) for r in sig.ran]
    print(f"  signal days {sig.date.min().date()} → {sig.date.max().date()}")
    results = {}
    trades_out = {}
    for style in STYLES:
        busy_until: dict[str, pd.Timestamp] = {}
        trades = []
        for s in sig.itertuples():
            h = px.get(s.ticker)
            if h is None or s.date <= busy_until.get(s.ticker, pd.Timestamp.min):
                continue
            t = simulate(s, h, style)
            if t is None:
                continue
            busy_until[s.ticker] = t["exit_date"] if t["reason"] != "open" else pd.Timestamp.max
            a = spy["Open"].asof(t["entry_date"]); b = spy["Close"].asof(t["exit_date"])
            t.update(ticker=s.ticker, signal_date=s.date, score=s.score, market=s.market,
                     days_to_earnings=s.days_to_earnings, spy_ret=b / a - 1)
            trades.append(t)
        tr = pd.DataFrame(trades)
        de = pd.to_numeric(tr.days_to_earnings, errors="coerce")
        res = {"all": summarize(tr), "by_score": {}, "by_earnings": {}, "by_market": {}}
        for name, m in {"80+": tr.score >= 80, "70-79": (tr.score >= 70) & (tr.score < 80),
                        "50-69": tr.score < 70}.items():
            res["by_score"][name] = summarize(tr[m])
        res["by_earnings"]["report within 7 days"] = summarize(tr[(de >= 0) & (de <= EARNINGS_DAYS)])
        res["by_earnings"]["no report soon"] = summarize(tr[(de > EARNINGS_DAYS) | (de < 0)])
        res["by_earnings"]["unknown"] = summarize(tr[de.isna()])
        for st in ["positive", "mixed", "negative"]:
            res["by_market"][st] = summarize(tr[tr.market == st])
        results[style] = res
        trades_out[style] = tr.sort_values("r").assign(
            signal_date=lambda x: x.signal_date.dt.strftime("%Y-%m-%d"),
            entry_date=lambda x: x.entry_date.dt.strftime("%Y-%m-%d"),
            exit_date=lambda x: x.exit_date.dt.strftime("%Y-%m-%d")).round(4).to_dict("records")
        a = res["all"]
        print(f"{style} {STYLES[style]}: n={a['n']} win={a['win_rate']}% avgR={a['avg_r']} "
              f"avg={a['avg_ret']}% vs SPY {a['spy_avg_ret']}% PF={a['profit_factor']}")
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "period": [str(sig.date.min().date()), str(sig.date.max().date())],
               "signals": len(sig), "styles": STYLES, "results": results, "trades": trades_out}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    write_html(payload)
    print(f"Wrote {OUT_JSON.name}, {OUT_HTML.name}")
    return 0


def _fmt(v, suffix=""):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v}{suffix}"


def _row(label, s):
    if not s or s.get("n", 0) == 0:
        return f"<tr><td>{label}</td><td colspan='7'>אין עסקאות</td></tr>"
    col = "var(--success)" if (s.get("avg_r") or 0) > 0 else "var(--danger)"
    return (f"<tr><td>{label}</td><td>{s['n']}</td><td>{s['win_rate']}%</td>"
            f"<td style='color:{col};font-weight:700'>{s['avg_r']}</td><td>{s['avg_ret']}%</td>"
            f"<td>{s['spy_avg_ret']}%</td><td>{_fmt(s['profit_factor'])}</td><td>{s['avg_days']}</td></tr>")


def write_html(p):
    try:
        from _theme import THEME_HEAD, THEME_TOGGLE_SCRIPT, THEME_TOGGLE_BUTTON
    except Exception:
        THEME_HEAD = THEME_TOGGLE_SCRIPT = THEME_TOGGLE_BUTTON = ""
    head = ("<tr><th>קבוצה</th><th>עסקאות</th><th>% מנצחות</th><th>R ממוצע</th><th>תשואה ממוצעת</th>"
            "<th>SPY באותו זמן</th><th>Profit Factor</th><th>ימים בממוצע</th></tr>")
    R = p["results"]
    main_rows = "".join(_row(f"{k} · {v}", R[k]["all"]) for k, v in p["styles"].items())
    sections = ""
    for k, v in p["styles"].items():
        r = R[k]
        rows = "".join(_row(f"ציון {n}", s) for n, s in r["by_score"].items())
        rows += "".join(_row({"report within 7 days": "🚫 דוח בתוך 7 ימים", "no report soon": "בלי דוח קרוב",
                              "unknown": "תאריך דוח לא ידוע"}[n], s) for n, s in r["by_earnings"].items())
        rows += "".join(_row({"positive": "🟢 שוק בריא", "mixed": "🟡 שוק מעורב", "negative": "🔴 שוק חלש"}[n], s)
                        for n, s in r["by_market"].items())
        worst = p["trades"][k][:5]
        best = p["trades"][k][-5:][::-1]
        tl = lambda ts: ", ".join(f"{t['ticker']} ({t['r']:+.1f}R, {t['signal_date']})" for t in ts)
        sections += f"""<h2>{k} · {v}</h2><table><thead>{head}</thead><tbody>{rows}</tbody></table>
<div class="note">הכי טובות: {tl(best)}<br>הכי גרועות: {tl(worst)}</div>"""
    html = f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>בדיקת הסיגנלים</title>
{THEME_HEAD}
<style>
  body {{ font-family:-apple-system,'Segoe UI','Heebo',sans-serif; margin:0; padding:24px 16px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  .top-nav {{ display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }}
  .nav-links {{ display:flex; gap:8px; align-items:center; }}
  .nav-btn {{ padding:8px 14px; border-radius:8px; font-size:13px; text-decoration:none; }}
  .sub {{ color:var(--text-muted); font-size:14px; margin-bottom:16px; }}
  h2 {{ margin-top:28px; font-size:17px; }}
  .tbl {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden; margin-top:10px; }}
  th, td {{ padding:8px 10px; text-align:start; font-size:13px; white-space:nowrap; }}
  .note {{ padding:10px 12px; border-radius:8px; font-size:13px; margin-top:10px; line-height:1.6; }}
</style></head>
<body><div class="wrap">
<div class="top-nav"><h1>🧪 בדיקת הסיגנלים לפי כללים</h1>
  <div class="nav-links"><a class="nav-btn" href="dashboard_standalone.html">← חזרה לדשבורד</a>{THEME_TOGGLE_BUTTON}</div></div>
<div class="sub">כל פריצה בארכיון ({p['signals']} סיגנלים, {p['period'][0]} עד {p['period'][1]}) כאילו נקנתה בפתיחה למחרת
ונוהלה לפי כלל יציאה קבוע. R = רווח או הפסד ביחס לסיכון המקורי (1R = הפסד מלא עד הסטופ).</div>
<div class="tbl"><table><thead>{head}</thead><tbody>{main_rows}</tbody></table></div>
<div class="note">⚠️ תקופה קצרה (פחות משנה) ושוק אחד. עסקאות שעדיין פתוחות מחושבות לפי מחיר הסגירה האחרון.
עלות של 0.1% לכל צד. זו בדיקה של נתוני עבר, לא המלצה.</div>
<div class="tbl">{sections}</div>
<div class="sub" style="margin-top:24px">עודכן: {p['generated_at']}</div>
</div>{THEME_TOGGLE_SCRIPT}</body></html>"""
    OUT_HTML.write_text(html)


if __name__ == "__main__":
    raise SystemExit(main())
