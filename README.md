# BourseRadar

BourseRadar is a command-line tool for reviewing Tehran Stock Exchange instruments. It retrieves market data, evaluates price and trading activity, and presents an indicative summary for each requested symbol.

## Capabilities

- Analyze one symbol or a space- or comma-separated list of symbols.
- Display a directional assessment, a score-based confidence measure, and indicative price levels.
- Evaluate multiple market signals together rather than relying on a single measure.
- Provide an English or Persian interface.

## Requirements

- Python 3.9 or later
- Internet access to retrieve market data

## Installation

```bash
git clone https://github.com/AmirRezaVR/BourseRadar.git
cd BourseRadar
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies and start the application:

```bash
pip install -r requirements.txt
python main.py
```

Choose a language when prompted, then enter one or more Tehran Stock Exchange symbols. Enter `q`, `quit`, or `exit` to finish.

## Data and limitations

Market data is retrieved from TSETMC services and depends on their availability and coverage. The analysis is based on available market data; it does not incorporate company fundamentals, news, or an investor's personal circumstances. The confidence measure is derived from the analysis score and is not a probability of future performance.

BourseRadar provides analysis only. It does not place orders or guarantee results. Any displayed assessment or price level is informational, not investment advice.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
