import yfinance as yf
import pandas as pd

import datetime
import math
from typing import Iterable

class Stocks:
    def __init__(self, stocks: Iterable[str]):
        self.stocks = stocks

    def get_stock_data(self, start_date: str, duration: int, increment: str) -> pd.DataFrame:
        end_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=duration)

        stocks = yf.Tickers(" ".join(self.stocks))
        data = stocks.download(start=start_date, end=end_date, interval=increment)

        if data is None:
            raise ValueError("No data found for the given date range and increment")

        return data

    def multi_sample_range(self, days_before_today: int, reps: int, increment: str) -> pd.DataFrame:
        if reps < 1:
            raise ValueError("Number of repetitions must be at least 1")

        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_before_today)

        duration = int(days_before_today // reps)

        data = self.get_stock_data(start_date.strftime('%Y-%m-%d'), duration, increment)

        for _ in range(reps - 1):
            start_date += datetime.timedelta(days=duration)
            data = pd.concat([data, self.get_stock_data(start_date.strftime('%Y-%m-%d'), duration, increment)])

        if data is None:
            raise ValueError("No data found for the given date range and increment")

        return data


class Market:
    def __init__(self, stocks: tuple[str, ...], trade_fee: float):
        self.stocks = stocks
        self.trade_fee = trade_fee

        self.current_date = None
        self.start_date = None
        self.increment = None
        self.index = 0
        self.balance = None
        self.profits = 0

        self.stock_data = Stocks(stocks)
        self.data = None

        self.started_sim = True

        self.portfolio = {stock: 0.0 for stock in stocks}

    def start_market_sim(self, start_date: str, duration: int, increment: str, balance: float) -> None:
        self.current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.start_date = start_date
        self.increment = increment

        self.balance = balance
        self.profits = 0

        if increment != "1m":
            self.data = self.stock_data.get_stock_data(start_date, duration, increment)
        else:
            if duration > 28:
                raise ValueError("Duration must be less than or equal to 28 days for 1-minute increments")

            self.data = self.stock_data.multi_sample_range(duration, math.ceil(duration / 7), "1m")

        self.started_sim = True

    def get_data_point(self, history: int, horizon: int) -> tuple[pd.DataFrame, float]:
        if not self.started_sim:
            raise ValueError("Market simulation not started")

        if self.data is None:
            raise ValueError("No data found in the current simulation for the given date range and increment")

        data = self.data[self.index: self.index + history]
        future_data = self.data[self.index + history + horizon: self.index + history + horizon + 1]

        return data, future_data

    def purchase_stock(self, stock: str, quantity: float, time: str="Close") -> None:
        if stock not in self.portfolio:
            raise ValueError(f"Invalid stock symbol: {stock}")

        data, _ = self.get_data_point(1, 1)
        price = data[time][stock].to_numpy()[0]

        if self.balance < price * quantity:
            raise ValueError("Insufficient balance to purchase stock")

        if not isinstance(price, float):
            raise ValueError("Price must be a float")

        if not isinstance(self.balance, float):
            raise ValueError(f"Balance must be a float, not a {type(self.balance).__name__}")

        if quantity >= 0:
            self.balance -= price * quantity - self.trade_fee
            self.portfolio[stock] += quantity
        elif quantity == -1:
            self.portfolio[stock] += self.balance / price
            self.balance = 0.0
        else:
            raise ValueError("Quantity must be positive or -1")

    def sell_stock(self, stock: str, quantity: float, time: str="Close") -> None:
        if stock not in self.portfolio:
            raise ValueError(f"Invalid stock symbol: {stock}")

        data, _ = self.get_data_point(1, 1)
        price = data[time][stock].to_numpy()[0]

        if self.portfolio[stock] < quantity:
            raise ValueError("Insufficient quantity of stock to sell")

        if not isinstance(price, float):
            raise ValueError("Price must be a float")

        if not isinstance(self.balance, float):
            raise ValueError(f"Balance must be a float, not a {type(self.balance).__name__}")

        if quantity >= 0:
            self.balance += price * quantity - self.trade_fee
            self.portfolio[stock] -= quantity
        elif quantity == -1:
            self.balance += price * self.portfolio[stock] - self.trade_fee
            self.portfolio[stock] = 0.0
        else:
            raise ValueError("Quantity must be positive or -1")

    def sell_portfolio(self):
        for stock, quantity in self.portfolio.items():
            self.sell_stock(stock, quantity)

    def step(self, step_size: int=1) -> bool:
        if step_size <= 0:
            raise ValueError("Step size must be a positive integer")

        if self.index + step_size >= self.data.shape[0] and step_size > 1:
            raise ValueError(f"Step size exceeds available data: {self.index + step_size + 1} > {self.data.shape[0]}")

        if self.index + step_size >= self.data.shape[0]:
            return False
        else:
            self.index += step_size
            return True

    def get_portfolio(self):
        out = f"Balance: {self.balance:.2f}\n"
        for stock, quantity in self.portfolio.items():
            out += f"{stock}: {quantity}\n"
        return out

if __name__ == "__main__":
    market = Market(("AAPL", "GOOGL", "MSFT", "NVDA"), 0.0)
    market.start_market_sim("2010-04-27", 365 * 15, "1d", 10000.0)
    data, future_data = market.get_data_point(1, 1)
    market.purchase_stock("NVDA", -1)
    print(market.get_portfolio())
    market.step(3770)
    market.sell_stock("NVDA", -1)
    print(market.get_portfolio())
