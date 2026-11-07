# BourseRadar

BourseRadar analyzes stocks on the Tehran Stock Exchange using public market data from TSETMC. For each symbol it gives a verdict, a signal strength, and price levels. It is available as a desktop window and as a terminal program.

> **Disclaimer:** This tool is for information and education only. It is not investment advice and does not guarantee any result. All decisions and risks are yours.

## How it works

For each symbol, the program downloads daily history and calculates a set of technical indicators (trend, momentum, volume, volatility, and market flow). Their scores are combined into one composite score, which produces the verdict.

## Install and run

**Windows:** Download `BourseRadar-windows.zip` from the Releases page, extract it to a folder you can write to (not Program Files), and run `BourseRadar.exe`.

**From source:**

```bash
pip install -r requirements.txt
python app_gui.py    # desktop version
python main.py       # terminal version
```

## Usage

- **Desktop:** Type one or more symbols separated by spaces, then press Enter or click the analyze button. Each symbol appears as a row in the table.
- **Terminal:** Choose a language, type one or more symbols separated by spaces, and type `exit` to quit.

Use short symbol names as they appear on the exchange. Full company names are not supported.

## Reading the output

- **Verdict:** A summary of the analysis, from strong buy to strong sell.
- **Signal strength:** A number from 0 to 100. It reflects how strong the analysis is, how many active indicators agree, and how actively the stock trades. It is not the probability that the verdict is correct.
- **Price levels:** Entry, stop loss, target, and risk/reward are estimates based on recent prices and volatility. They are not orders. A dash (—) means no value applies for that verdict or that the data is not sufficient.
- **Warnings:** Shown when the data has issues, such as large price jumps that may come from corporate actions.

## Limitations

- Indicator weights and thresholds are set manually and have not been optimized.
- Broker commissions and taxes are not included in the results.
- Suspended, adjusted, or limit-price stocks can produce misleading signals.
- TSETMC data is not an official API and may change without notice.
