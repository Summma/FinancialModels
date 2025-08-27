import yfinance as yf
import pandas as pd
import numpy as np

stocks = (
    "NVDA",
    "MSFT",
    "AMD",
    "ASML",
    "INTC",
    "GOOGL",
    "AMZN",
    "SMCI",
    "MU",
    "LIN",
    "EQIX",
    "DLR",
    "NEE",
    "AAPL",
    "EXC",
    "LRCX",
    "KLAC",
)

if __name__ == "__main__":
    # df = yf.download(stocks, start="2010-01-01")
    # df = df.fillna(0)
    # df.to_csv("finance_data.csv", index=False)

    tickers = yf.Tickers(" ".join(stocks))
    df_1m = tickers.download(
        start="2025-05-01",
        end="2025-05-08",
        interval="1m"
    )
    print(df_1m[1:2])
