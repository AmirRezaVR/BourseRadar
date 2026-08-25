# BourseRadar

A small Python tool I built to stop eyeballing charts every morning.

You give it a Tehran Stock Exchange symbol, it pulls the real price history, runs it through a handful of technical factors, and hands you back a plain answer: strong buy, buy, hold, sell, or strong sell, with a confidence percentage attached — plus a suggested entry price, stop-loss, and target. It also explains itself: every factor it checked, what it found, and how much weight it carried, so you can see whether the logic makes sense before acting on it.

It's not a trading bot, it doesn't place orders, and it has no idea what a company's earnings look like or what's in the news. Just price, volume, and a few well-known indicators, weighed against each other.

## What it's like to use

You type a symbol — `فملی`, `فولاد`, whatever — and it looks it up for you, no ticker code memorized. Want to check a few at once? Space them out and it'll run through all of them, then give you a summary table.

Behind the scenes, it weighs five things against each other — RSI, MACD, moving-average trend alignment, volume, and where price sits relative to recent support and resistance — instead of leaning on any single indicator. Volatility works differently: it doesn't push the verdict one way or the other, but it knocks the confidence number down when things get unusually choppy.

First run, it asks whether you want English or Persian, and that's what you get from then on. It keeps running until you tell it to stop (`exit`, `quit`, `q`, `خروج`, `پایان`), so you can work through a watchlist in one sitting.

## What it doesn't do

No fundamentals, no news, no sentiment. No divergence detection or order-book data yet, though that's on the list. No charts — plain text in a terminal for now. No portfolio tracking, so it won't tell you how your overall holdings are doing, only what it thinks of one symbol at a time. No real-time streaming — daily data, checked whenever you ask.

## Getting started

```bash
git clone https://github.com/<your-username>/bourseradar.git
cd bourseradar

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

It'll ask which language you want, then just start typing symbols.

## A couple of honest notes

TSETMC doesn't publish an official API, so this talks to endpoints I found by inspecting real responses, not documentation — because there isn't any. If a symbol suddenly stops working, that's probably why.

The recommendation is a calculation, not a prediction. It weighs trend, momentum, volume, and volatility and tells you what that combination adds up to — nothing more. It's blind to earnings reports, sanctions news, or anything else happening outside the price chart. Treat it as one opinion in the room, not the final word.

## Project layout

```
bourseradar/
├── data/                    # your local SQLite database (not tracked in git)
├── src/
│   ├── database.py          # saves and reads price history
│   ├── tsetmc_fetcher.py    # talks to TSETMC
│   ├── indicators.py        # SMA, EMA, RSI, MACD, ATR
│   └── signal_engine.py     # weighs the indicators into a scored recommendation
├── main.py                  # run this
└── requirements.txt
```

## Where this is headed

Divergence detection and some TSE-specific signals (order-book, buyer/seller money flow) are next. After that, backtesting against real historical data, so the confidence numbers mean something proven, not just internally consistent.

## License

MIT — see [LICENSE](LICENSE).
