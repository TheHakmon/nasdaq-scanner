"""Write a tiny public summary of the latest scan for yossihakmon.com.

Reads scan_results.json (~3 MB) and writes site_summary.json (~1 KB) with only
neutral facts: scan date, counts, and the top 3 tickers by score.
No buy/sell wording, entry, stop-loss or targets — the site is not investment advice.
Usage: python3 make_site_summary.py [output_path]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _clean_name(name):
    """The scanner cuts names at ~31 chars; drop a half-cut last word."""
    if not name or len(name) < 30 or " " not in name:
        return name
    return name.rsplit(" ", 1)[0].rstrip(",")


def build() -> dict:
    d = json.loads((ROOT / "scan_results.json").read_text(encoding="utf-8"))
    breakouts = d.get("breakouts") or []
    source = breakouts if breakouts else (d.get("watchlist") or [])
    top = sorted(source, key=lambda r: r.get("score") or 0, reverse=True)[:3]
    return {
        "generated_at": d.get("generated_at"),
        "scanned": d.get("scanned"),
        "breakouts": len(breakouts),
        "watchlist": len(d.get("watchlist") or []),
        "top_kind": "breakouts" if breakouts else "watchlist",
        "top": [
            {
                "ticker": r.get("ticker"),
                "name": _clean_name(r.get("company_name")),
                "score": r.get("score"),
                "pct_above_ma150": r.get("pct_above_ma150"),
            }
            for r in top
        ],
        "dashboard": "https://thehakmon.github.io/nasdaq-scanner/",
    }


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
