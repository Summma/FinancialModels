# opt_portfolio_bt.py

import backtrader as bt
import yfinance as yf
from datetime import datetime


class OptimalPortfolio(bt.Strategy):
    params = dict(
        # your optimized weights
        weights={
            "PLTR": 0.1059,
            "WMT": 0.4611,
            "AVGO": 0.1225,
            "XOM": 0.1371,
            "T": 0.1072,
            "NVDA": 0.0661,
        }
    )

    def __init__(self):
        self.placed = False

    def next(self):
        if not self.placed:
            cash = self.broker.getcash()
            for data in self.datas:
                ticker = data._name
                w = self.p.weights.get(ticker, 0.0)
                if w > 0:
                    alloc = cash * w
                    price = data.close[0]
                    size = int(alloc / price)
                    if size > 0:
                        self.buy(data=data, size=size)
            self.placed = True


class BuyHoldAll(bt.Strategy):
    """
    Buy and hold an equal-dollar allocation to each ticker on the first bar,
    then do nothing for the remainder of the backtest.
    """

    def __init__(self):
        self.placed = False  # ensure we only buy once

    def next(self):
        if not self.placed:
            cash = self.broker.getcash()
            n = len(self.datas)
            # allocate equal dollars to each asset
            alloc_per_asset = cash / n

            for data in self.datas:
                price = data.close[0]
                size = int(alloc_per_asset / price)
                if size > 0:
                    self.buy(data=data, size=size)

            self.placed = True


if __name__ == "__main__":
    cerebro = bt.Cerebro(stdstats=False)
    # cerebro.addstrategy(OptimalPortfolio)
    cerebro.addstrategy(BuyHoldAll)

    # Your tickers and date range
    tickers = ["PLTR", "WMT", "AVGO", "XOM", "T", "NVDA"]
    start = "2022-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    # Download & feed each ticker
    for ticker in tickers:
        df = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=False
        )
        # keep only the OHLCV columns and lowercase them
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]

        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=datetime.strptime(start, "%Y-%m-%d"),
            todate=datetime.strptime(end, "%Y-%m-%d"),
            name=ticker,
        )
        cerebro.adddata(data)

    # Broker setup
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.0)  # no commission

    # Run
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value:   {cerebro.broker.getvalue():.2f}")

    # Plot
    cerebro.plot()
