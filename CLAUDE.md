# NASDAQ Breakout Scanner — Project Memory

**Owner:** Yossi Hakmon (imyossihakmon@gmail.com, GitHub: TheHakmon)
**Live site:** https://thehakmon.github.io/nasdaq-scanner/
**Repo:** https://github.com/TheHakmon/nasdaq-scanner (public)
**Local path:** `/Users/yossihakmon/ClaudeProjects/nasdaq-scanner`

## What this project is

A fully-automated NASDAQ technical breakout scanner + interactive trade journal, running daily in the cloud (GitHub Actions) and published as a mobile-installable PWA on GitHub Pages. Built from May–Sep 2026 across many iterations.

Scans ~200 NASDAQ stocks with market cap > $5B daily, applying Minervini/Weinstein-style breakout criteria: cross above MA150 with volume confirmation, Stage 2 alignment, ADX trend strength, RS Rating 1-99, sector strength, growth fundamentals. Produces a scored recommendation with clear buy/wait/no headline in Hebrew for each candidate.

## Architecture

- **Runtime:** GitHub Actions (Ubuntu, Python 3.11, yfinance). Cron: `35 20 * * 1-5` UTC (~23:35 IDT summer, 30 min after NASDAQ close).
- **Publishing:** GitHub Pages, deployed from `_site/` via `actions/deploy-pages@v4`.
- **Workflow commits back:** Fresh scan results + archive entries are pushed to the repo, then deployed.
- **Journal sync:** Browser localStorage by default; optional cross-device sync via JSONBin.io (user enters BIN ID + Master Key in the ☁️ סנכרון modal — creds stay client-side, never in the repo).
- **Legacy local launchd** on user's Mac was **disabled** — the cloud is authoritative. If it's ever re-enabled it will cause repo conflicts (this happened once, resolved).

## Key files

| File | Purpose |
|---|---|
| `scan_nasdaq.py` | Main scanner. All indicators, scoring, headline recommendation, chart generation, notifications, PWA support, archive save, journal price injection. Runs first. |
| `track_performance.py` | Post-processing. Reads archive, computes returns at 1/5/20/60 days per historical signal, generates `performance_summary.html`, augments `scan_results.json` with per-ticker track record. |
| `backtest_archive.py` | One-shot backtest generator (default 90 days). Reconstructs "what would have been flagged on date X". Not part of daily run. |
| `_theme.py` | Shared CSS + JS for dark-mode theme across all auxiliary HTML pages. |
| `nasdaq_dashboard.html` | Dashboard template — `scan_nasdaq.py` inlines data into it to produce `dashboard_standalone.html` (which is what Pages serves as `index.html`). |
| `journal_app.html` | Fully-interactive trade journal (add/sell/partial-sell), localStorage + optional JSONBin sync. Price map injected daily by `scan_nasdaq.build_journal_prices`. |
| `nasdaq_universe.json` | Curated ~200 NASDAQ tickers > $5B. Editable. |
| `personal_watchlist.json` | User's personal tickers to also scan (bypasses market-cap & liquidity filters). |
| `.github/workflows/daily-scan.yml` | The Actions workflow. |
| `requirements.txt` | yfinance, pandas, numpy, matplotlib. |
| `manifest.webmanifest`, `icon_*.png` | PWA — makes the site installable as an app on iPhone/Android. |
| `archive/scan_YYYY-MM-DD.{json,html}` | Every daily snapshot preserved. Currently ~365 files. |
| `archive/index.html` | Chronological archive browser. |

## Notifications (optional, set via GitHub repo Secrets)

- **Email** (Gmail SMTP + App Password): secrets `EMAIL_USERNAME`, `EMAIL_APP_PASSWORD`, `EMAIL_TO`.
- **Telegram**: secrets `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Config path when running locally: `~/.nasdaq_scanner_telegram.json`.
- Only sent when there are breakouts. Both wired in `scan_nasdaq.py`.

## What's implemented (features summary)

Technical indicators: MA50/150/200, RSI(14), ATR(14), ADX(14), Volume vs 50-day avg, Relative Strength vs SPY, weekly MA30, 52-week high distance, sector ETF strength (XLK/XLV/XLF/…), earnings proximity from `t.calendar`, EPS/revenue growth from `t.info`, extension above MA50, RS Rating 1-99 (ranked across universe).

Composite scoring model + a **plain-Hebrew headline recommendation** (`_build_headline_recommendation`) that answers "buy now / wait for pullback / wait for earnings / no" in one sentence at the top of each card. Full technical breakdown follows below.

Trade management: ATR-based Stop-Loss and 2 targets (R/R 1:2, 1:3), position-size calculator in the dashboard (persisted in localStorage), interactive journal with add / sell / partial-sell / edit / delete, portfolio P&L stats, per-ticker aggregation.

Historical: 90+ day archive, performance summary with win-rate by score band, per-ticker historical track record badge on each dashboard card ("💎 טרק רקורד חזק" etc).

Charts: matplotlib PNG (offline fallback) + TradingView Lightweight Charts (interactive, dark-theme aware) with candlesticks, MA overlays, breakout marker, horizontal Entry/Stop/Target lines. Deep-link to TradingView per ticker.

News: 3 latest headlines per candidate via yfinance `t.news`.

UI: Dark mode by default with light toggle (persisted). Mobile-responsive (one card per row < 700px, larger text, iOS zoom prevention). PWA — "Add to Home Screen" gives an app icon with fullscreen mode.

## Development workflow

Edits are made to the template files (`nasdaq_dashboard.html`, `scan_nasdaq.py` etc.), pushed to GitHub. The next workflow run regenerates all derived files (`dashboard_standalone.html`, archive HTMLs, `journal_app.html` price injection) and deploys.

To push:
```bash
cd ~/ClaudeProjects/nasdaq-scanner
git pull --rebase origin main    # ALWAYS pull first — cloud workflow commits daily
git add -A && git commit -m "…" && git push
```

Then trigger manually at github.com/TheHakmon/nasdaq-scanner/actions → Daily NASDAQ Scan → Run workflow (or just wait for next scheduled run).

## Recurring gotchas (learned the hard way)

- **Auth**: GitHub personal access token (classic) with **both `repo` AND `workflow`** scopes is required to push. Token belongs to `TheHakmon` (not `imyossihakmon-ai`, which is a separate account owned by the user but not used).
- **Conflicts** happen if the local git repo is behind the cloud workflow. Always `git pull --rebase origin main` before pushing. If archive files conflict, `git checkout --ours` (cloud version) — data files are always regenerated so cloud's version is authoritative.
- **Editor prompts**: `git rebase --continue` may open nano. Save & exit with `Ctrl+X`, `Y`, `Enter`. Do NOT `Ctrl+Z` (suspends the editor instead of quitting).
- **iOS PWA cache**: Safari aggressively caches. After a big change: delete home-screen icon, close all tabs, re-add.
- **Bash quoting** in commands with apostrophes (e.g. "they'll") — strip comments before pasting, or bash gets stuck at `quote>`.
- **matplotlib tight_layout warning** already silenced via `warnings.filterwarnings` and `constrained_layout=True`.

## Recommended next steps (from the last discussion, not yet built)

Ranked by value if the user wants to keep extending:
1. **Full backtest engine** — simulate the strategy P&L historically ("if I followed score ≥ 70 with 2×ATR stops, where would I be?"). Concrete validation.
2. **VCP (Volatility Contraction Pattern)** — Minervini's classic signature; complements the current breakout detector by identifying setups *before* the breakout.
3. **Market regime indicator** (VIX + market breadth) — protects against buying into weak markets; suggests reducing position size when risky.
4. Dashboard filtering (by sector, score, RS Rating), pre-market gappers scan, MACD/OBV/Bollinger squeeze, trailing stops.

**Honest advice given to the user**: the system is already more sophisticated than most retail traders use. The highest-value next step is not more features — it's actually using it with paper trades in the journal, then extending based on real-world need.

## Environment quirks

- Cowork sandbox blocks direct network access to yahoo.com (proxy 403), so `scan_nasdaq.py` cannot run in this environment. All runs happen either in GitHub Actions (cloud) or on the user's Mac (has full network). We do development/testing here and push to GitHub for execution.
- The old `/Users/yossihakmon/Library/Application Support/Claude/local-agent-mode-sessions/.../outputs/` was the working folder during development; the project has now been consolidated into `~/ClaudeProjects/nasdaq-scanner/`.
