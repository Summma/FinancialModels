import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1) Your optimal weights
# weights = {
#     "AVGO": 0.1167,
#     "KO": 0.0216,
#     "NVDA": 0.0520,
#     "PLTR": 0.0722,
#     "ROOT": 0.0217,
#     "T": 0.2201,
#     "WMT": 0.3321,
#     "XOM": 0.1636
# }
weights = {
    "AVGO": 0.3367,
    "MRK": 0.4098,
    "UBER": 0.0152,
    "XOM": 0.2383,
}

tickers = list(weights.keys())

# 2) Backtest window
start_date = "2024-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
end_date = "2024-04-04"

# 3) Download adjusted close prices
prices = (
    yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )["Adj Close"]
    .dropna(how="all")
    .ffill()
    .dropna(how="any")
)

# 4) Compute daily simple returns
rets = prices.pct_change().dropna()

# 5) Portfolio daily returns
w = pd.Series(weights)
port_rets = rets.dot(w)

# 6) Equity curve (normalized to 1.0 at t=0)
equity = (1 + port_rets).cumprod()

# 7) Performance metrics
initial_capital = 100_000
final_value = equity.iloc[-1] * initial_capital
annual_return = port_rets.mean() * 252
annual_vol = port_rets.std() * np.sqrt(252)
expost_sharpe = annual_return / annual_vol

# 8) Print results
print(f"Backtest period: {start_date} → {end_date}")
print(f"Starting capital: ${initial_capital:,.0f}")
print(f"Final portfolio value: ${final_value:,.0f}")
print(f"Annualized return: {annual_return:.2%}")
print(f"Annualized volatility: {annual_vol:.2%}")
print(f"Ex-post Sharpe ratio: {expost_sharpe:.2f}")
