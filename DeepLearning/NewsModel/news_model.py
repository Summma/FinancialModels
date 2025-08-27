from trading_endpoint import Stocks

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pandas as pd
from gdeltdoc import GdeltDoc, Filters

import os
from typing import Sequence
from time import time

load_dotenv()

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

stock_list = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "SNOW": "Snowflake",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "NFLX": "Netflix",
}


class NewsData(Dataset):
    def __init__(
        self, stocks: Sequence[str], start_date: str, duration: int, increment: str
    ):
        return
        self.stocks = Stocks(stocks=stocks)
        self.start_date = start_date
        self.duration = duration
        self.increment = increment
        self.n_stocks: int = len(stocks)

        self.df: pd.DataFrame = self.stocks.get_stock_data(
            start_date=start_date, duration=duration, increment=increment
        )
        self.df = self.df.stack(level=1).reset_index()

    def __len__(self):
        return len(self.df) - self.n_stocks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        next_row = self.df.iloc[idx + self.n_stocks]

        X = news.get_everything(q=stock_list[row["Ticker"]], from_param=row["Date"])
        y = torch.tensor(next_row["Close"].values)

        return X, y


test = NewsData(list(stock_list.keys()), "2022-01-01", 7, "1d")

gd = GdeltDoc()
start = time()

for i in range(2, 29):
    f = Filters(
        keyword="Apple",
        start_date="2025-05-01",
        end_date=f"2025-05-{str(i).zfill(2)}",
        language="english",
        country="US",
        # domain = ["reuters.com", "forbes.com", "nytimes.com"],
        num_records=250,
    )

    articles = gd.article_search(f)
    print(articles)

print(time() - start)
