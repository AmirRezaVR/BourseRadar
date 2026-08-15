# BourseRadar

A small Python tool I built to stop eyeballing charts every morning.

It pulls real price data for Tehran Stock Exchange stocks, runs a handful of standard technical indicators on them, and gives me a plain answer — buy, wait, or skip — along with a suggested entry price, a stop-loss, and a target. It explains its reasoning in both English and Persian, so it's easy to double-check that the logic actually makes sense before acting on it.

That's it. It's not a trading bot, it doesn't place orders, and it doesn't know anything about a company's fundamentals or the news — just price, volume, and a few well-known indicators.

## What it does

- You type a symbol (`فملی`, `فولاد`, whatever) — no need to look up a code, it searches TSETMC for you
- It fetches the price history and saves it locally, so you're not re-downloading the same data every time
- It calculates SMA, EMA, RSI, MACD, and ATR
- It turns those numbers into a short-term swing-trade read: buy zone, stop-loss, target, and a plain-language explanation of why — in English and Persian
- It keeps running so you can check several symbols in one sitting, and exits cleanly when you tell it to

## What it doesn't do (yet, or maybe ever)

- No fundamentals, no news, no sentiment
- No charts — right now it's just text in your terminal
- No portfolio tracking — it looks at one symbol at a time, not "how is my overall position doing"
- No real-time streaming — it's daily data, checked when you ask for it

Some of these are planned. Some might just stay out of scope on purpose, since the goal was a fast, honest second opinion — not a full trading platform.

## Getting started

```bash
git clone https://github.com/<your-username>/bourseradar.git
cd bourseradar

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

Then just type a symbol when it asks. Type `exit` or `خروج` when you're done.

## An honest note on the data

TSETMC doesn't publish an official API. This project talks to endpoints that were found and reverse-engineered by inspecting real responses — not from documentation, because there isn't any. That means things can break without warning if TSETMC changes something on their end. If a symbol suddenly stops working, that's usually why.

## And an honest note on the advice part

The buy/wait/skip suggestion is a rule-based calculation, not a prediction. It looks at trend, momentum, and volatility, and tells you what those specific numbers imply — nothing more. It can be wrong. It doesn't know about earnings reports, sanctions news, or anything happening outside the price chart. Treat it as one input, not a verdict.

## Project layout

```
bourseradar/
├── data/                    # your local SQLite database (not tracked in git)
├── src/
│   ├── database.py          # saves and reads price history
│   ├── tsetmc_fetcher.py    # talks to TSETMC
│   ├── indicators.py        # SMA, EMA, RSI, MACD, ATR
│   └── trade_advisor.py     # turns indicators into a buy/wait/skip read
├── main.py                  # run this
└── requirements.txt
```

## Where this is headed

Still fairly early. Next up is probably a proper stock screener and volume/order-book analysis — the rest is in the [issues](../../issues) if you're curious or want to help.

## License

MIT — see [LICENSE](LICENSE).
