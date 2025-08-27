from DeepLearning.NewsModel.constants import STOCK_LIST
from trading_endpoint import Stocks

from gdeltdoc import GdeltDoc, Filters
from tqdm import tqdm

from datetime import datetime, timedelta
from typing import Generator, Sequence, Any
from time import sleep, time
from math import ceil


def date_range(start_date: datetime, days: int) -> Generator[datetime, None, None]:
    for i in range(days):
        yield start_date + timedelta(days=i)


def generate_data(start_date: str, end_date: str, increment: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    days = int((end - start).days)

    stocks = Stocks(list(STOCK_LIST.keys()))

    df = stocks.get_stock_data(
        start_date=start_date, duration=days, increment=increment
    )
    df = df.stack(level=1, future_stack=True).reset_index()

    dates = set(df["Date"].unique())

    gd = GdeltDoc()

    news_titles: list[str | None] = []
    for date in tqdm(date_range(start, days), total=len(dates)):
        if date not in dates:
            continue

        for i in range(0, ceil(len(STOCK_LIST) / 7), 7):
            companies = list(STOCK_LIST.values())[i : i + 7]

            f = Filters(
                keyword=companies,
                start_date=date - timedelta(days=1),
                end_date=date,
                language="english",
                country="US",
                domain=["reuters.com", "forbes.com"],
                num_records=250,
                tone_absolute=">2",
            )

            start = time()
            articles = gd.article_search(f)
            for company in companies:
                print(f"Company: {company}")
                print(articles["title"].str.contains(company))
            print(articles.columns)
            return
            if articles.shape != (0, 0):
                if isinstance(title := articles["title"][0], str):
                    news_titles.append(title)
            else:
                news_titles.append(None)

            end = time()

            sleep(5 - (end - start))

    df["news_titles"] = news_titles
    print(df)
    df.to_csv("DeepLearning/NewsModel/data.csv")


if __name__ == "__main__":
    generate_data("2020-01-01", "2025-01-01", "1d")
    # generate_data("2020-01-01", "2020-01-04", "1d")
