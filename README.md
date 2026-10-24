# BourseRadar

BourseRadar is a desktop and terminal tool for analyzing stocks listed on the Tehran Stock Exchange (TSETMC). It collects public market data, runs technical and market-flow analyses, and shows a short summary for each symbol: a verdict, a confidence score, and suggested price levels.

> **Disclaimer:** BourseRadar is for information and education only. It is not financial advice, it does not guarantee any result, and it does not place orders. Every decision and its risks are yours.

## Features

- Analyze one or several symbols at once
- Desktop window (Windows) with a results table
- Terminal version for command-line users
- Persian interface; the terminal version also supports English
- Adjustable font size in the desktop window, remembered between runs
- Portable Windows build that needs no installation

## Download (Windows)

1. Open the [Releases](https://github.com/AmirRezaVR/BourseRadar/releases) page.
2. Download the latest `BourseRadar-windows.zip`.
3. Extract it to a folder you can write to, such as Desktop or Documents. Avoid Program Files, because the app saves its settings next to the program.
4. Run `BourseRadar.exe`.

Windows may show a SmartScreen warning because the file is not digitally signed. Click **More info**, then **Run anyway**.

## Run from source

Requirements: Python 3.10 or newer and an internet connection.

```bash
git clone https://github.com/AmirRezaVR/BourseRadar.git
cd BourseRadar
pip install -r requirements.txt
```

Desktop version:

```bash
python app_gui.py
```

Terminal version:

```bash
python main.py
```

## How to use

**Desktop:** Type one or more symbols separated by spaces, then press **Enter** or click **Analyze**. Each symbol appears as a row in the table. Use the font size selector to make the text larger.

**Terminal:** Choose a language, enter one or more symbols, and type `exit` to quit.

## Understanding the results

- **Verdict:** A summary of the analysis, ranging from strong buy to strong sell, with a neutral (hold) option.
- **Confidence:** How strongly the available signals agree with the verdict. It is not the probability that the verdict will be correct.
- **Entry range, stop loss, and target:** Price levels calculated from recent volatility. They are estimates, not instructions.
- **Last market update:** The time of the latest data received from the exchange.

Additional columns show the estimated risk and reward.

## Data source

BourseRadar reads public data from TSETMC. These endpoints are not official APIs and may change or stop working without notice. If a symbol returns no data, check your internet connection or try again later.

## Limitations

- Results depend on the available data and on the methods used. Past patterns do not guarantee future results.
- The confidence score has not been independently validated.
- Data for the current trading day may be incomplete during market hours.
- Commissions, taxes, and price slippage are not included in the calculations.

## Build the Windows executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name BourseRadar app_gui.py
```

The executable is created in the `dist` folder.

**Automatic releases:** Pushing a version tag such as `v2.0.0` triggers a GitHub Actions workflow that builds the Windows package and publishes it on the Releases page.

## Project structure

- `src/`: data retrieval and analysis
- `gui/`: desktop interface
- `main.py`: terminal interface
- `app_gui.py`: desktop entry point

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
