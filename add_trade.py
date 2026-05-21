#!/usr/bin/env python3
"""
Add a trade to the journal — one command, no manual JSON editing.

Usage:
    python3 add_trade.py AFRM
        → pulls AFRM's price/stop/targets from the latest scan_results.json,
          computes shares from your account/risk settings, and appends an
          open paper trade to trade_journal.json. Then refreshes the journal.

    python3 add_trade.py AFRM --real          # mark as a real trade
    python3 add_trade.py AFRM --shares 50     # override share count
    python3 add_trade.py AFRM --account 50000 --risk 1   # override sizing
    python3 add_trade.py AFRM --price 64.5 --stop 60 --t1 72 --t2 80  # manual

After adding, run nothing else — this script also rebuilds trade_journal.html.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCAN_JSON = HERE / "scan_results.json"
JOURNAL_JSON = HERE / "trade_journal.json"
SETTINGS_JSON = HERE / ".trade_settings.json"   # optional: {"account":..., "risk":...}


def _find_in_scan(ticker: str) -> dict | None:
    """Look up a ticker across breakouts/watchlist/personal/all_results."""
    if not SCAN_JSON.exists():
        return None
    data = json.loads(SCAN_JSON.read_text())
    for bucket in ("breakouts", "watchlist", "personal", "all_results"):
        for r in data.get(bucket, []):
            if r.get("ticker", "").upper() == ticker.upper():
                return r
    return None


def _load_settings() -> dict:
    if SETTINGS_JSON.exists():
        try:
            return json.loads(SETTINGS_JSON.read_text())
        except Exception:
            pass
    return {"account": 10000, "risk": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Add a trade to the journal")
    ap.add_argument("ticker", help="Stock symbol, e.g. AFRM")
    ap.add_argument("--real", action="store_true", help="Mark as real (default: paper)")
    ap.add_argument("--shares", type=int, help="Override share count")
    ap.add_argument("--account", type=float, help="Account size for sizing")
    ap.add_argument("--risk", type=float, help="Risk %% per trade for sizing")
    ap.add_argument("--price", type=float, help="Manual entry price")
    ap.add_argument("--stop", type=float, help="Manual stop-loss")
    ap.add_argument("--t1", type=float, help="Manual target 1")
    ap.add_argument("--t2", type=float, help="Manual target 2")
    ap.add_argument("--notes", default="", help="Free-text note")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    scan = _find_in_scan(ticker)

    # Resolve trade levels: manual flags override scan data.
    price = args.price or (scan or {}).get("price")
    stop = args.stop or (scan or {}).get("stop_loss")
    t1 = args.t1 or (scan or {}).get("target1")
    t2 = args.t2 or (scan or {}).get("target2")

    if price is None:
        print(f"❌ Couldn't find {ticker} in scan_results.json and no --price given.")
        print("   Provide manually:  python3 add_trade.py {0} --price X --stop Y --t1 Z --t2 W".format(ticker))
        return 1
    if stop is None:
        print(f"❌ No stop-loss for {ticker}. Provide --stop.")
        return 1

    # Position sizing
    settings = _load_settings()
    account = args.account or settings.get("account", 10000)
    risk = args.risk or settings.get("risk", 1.0)
    if args.shares:
        shares = args.shares
    else:
        risk_dollars = account * (risk / 100)
        risk_per_share = price - stop
        if risk_per_share <= 0:
            print(f"❌ Stop (${stop}) must be below entry (${price}).")
            return 1
        shares = max(1, int(risk_dollars / risk_per_share))

    today = datetime.now().strftime("%Y-%m-%d")
    notes = args.notes or (f"Score {scan.get('score')}, {scan.get('sector') or ''}".strip(", ") if scan else "")
    trade = {
        "id": f"{ticker}-{today}-{datetime.now().strftime('%H%M%S')}",
        "ticker": ticker,
        "kind": "real" if args.real else "paper",
        "status": "open",
        "entry_date": today,
        "entry_price": round(float(price), 2),
        "shares": int(shares),
        "stop_loss": round(float(stop), 2) if stop else None,
        "target1": round(float(t1), 2) if t1 else None,
        "target2": round(float(t2), 2) if t2 else None,
        "notes": notes,
    }

    # Append to journal
    if JOURNAL_JSON.exists():
        data = json.loads(JOURNAL_JSON.read_text())
    else:
        data = {"trades": []}
    data.setdefault("trades", []).append(trade)
    JOURNAL_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"✅ Added {trade['kind']} trade:")
    print(f"   {ticker}: {shares} shares @ ${trade['entry_price']}  "
          f"(stop ${trade['stop_loss']}, T1 ${trade['target1']}, T2 ${trade['target2']})")
    cost = shares * trade["entry_price"]
    max_loss = shares * (trade["entry_price"] - trade["stop_loss"]) if trade["stop_loss"] else 0
    print(f"   Cost ${cost:,.0f} · Max loss ${max_loss:,.0f}")

    # Refresh the journal HTML
    try:
        import subprocess
        subprocess.run([sys.executable, str(HERE / "trade_journal.py")],
                       capture_output=True, timeout=60)
        print("   📓 trade_journal.html refreshed.")
    except Exception as e:
        print(f"   (couldn't refresh journal HTML: {e} — run trade_journal.py manually)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
