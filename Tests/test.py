# optimize_sharpe_fixed.py

import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, expected_returns, risk_models
from datetime import datetime


def download_data(tickers, start_date, end_date):
    """Download adjusted close prices and clean the DataFrame."""
    # Explicitly set auto_adjust=False to keep 'Adj Close' column
    df = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )["Close"]
    df = df.dropna(how="all").ffill().dropna(how="any")
    return df


def greedy_sharpe_selection(mu, S, tickers):
    """
    Greedy forward‐selection: start with empty portfolio,
    add the ticker that yields the highest Sharpe until no improvement.
    """
    selected = []
    remaining = tickers.copy()
    best_sharpe = -np.inf

    while remaining:
        sharpe_candidates = {}
        for t in remaining:
            universe = selected + [t]
            mu_sub = mu[universe]
            S_sub = S.loc[universe, universe]
            ef = EfficientFrontier(mu_sub, S_sub)
            try:
                # force risk-free rate to zero
                ef.max_sharpe(risk_free_rate=0.0)
                _, _, sharpe = ef.portfolio_performance(risk_free_rate=0.0)
            except ValueError:
                sharpe = -np.inf
            sharpe_candidates[t] = sharpe

        t_best, sharpe_best = max(sharpe_candidates.items(), key=lambda x: x[1])
        if sharpe_best > best_sharpe:
            selected.append(t_best)
            remaining.remove(t_best)
            best_sharpe = sharpe_best
        else:
            break

    return selected, best_sharpe


def optimize_portfolio(tickers, start_date, end_date):
    # 1. Download price data
    df = download_data(tickers, start_date, end_date)

    # 2. Estimate expected returns & sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # 3. Greedy selection for max Sharpe
    selected, achieved_sharpe = greedy_sharpe_selection(mu, S, tickers)

    # 4. Final mean-variance optimization on selected universe
    mu_sub = mu[selected]
    S_sub = S.loc[selected, selected]
    ef = EfficientFrontier(mu_sub, S_sub)
    # again force zero risk-free for consistency
    ef.max_sharpe(risk_free_rate=0.0)
    weights = ef.clean_weights()

    return selected, weights, achieved_sharpe


if __name__ == "__main__":
    # 40 popular and 40 upcoming tickers
    popular_stocks = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "JPM",
        "V",
        "JNJ",
        "PG",
        "MA",
        "UNH",
        "HD",
        "DIS",
        "BAC",
        "XOM",
        "PFE",
        "CVX",
        "KO",
        "PEP",
        "CSCO",
        "CMCSA",
        "ADBE",
        "NFLX",
        "CRM",
        "ORCL",
        "INTC",
        "AVGO",
        "TXN",
        "T",
        "VZ",
        "QCOM",
        "NKE",
        "MRK",
        "ABT",
        "MDT",
        "MCD",
        "COST",
        "WMT",
    ]
    upcoming_stocks = [
        # 2020 IPOs
        "SNOW",  # Snowflake (Sep ’20)
        "ABNB",  # Airbnb    (Dec ’20)
        "DASH",  # DoorDash  (Dec ’20)
        "PLTR",  # Palantir  (Sep ’20)
        "LMND",  # Lemonade  (Jul ’20)
        "ASAN",  # Asana     (Sep ’20)
        "AI",  # C3.ai     (Dec ’20)
        "U",  # Unity Software (Sep ’20)
        "ROOT",  # Root, Inc.   (Aug ’20)
        "GDRX",  # GoodRx    (Sep ’20)
        # 2021 IPOs
        "COIN",  # Coinbase       (Apr ’21)
        "RBLX",  # Roblox         (Mar ’21)
        "AFRM",  # Affirm         (Jan ’21)
        "BMBL",  # Bumble         (Feb ’21)
        "SOFI",  # SoFi           (Jan ’21)
        "RIVN",  # Rivian         (Nov ’21)
        "HOOD",  # Robinhood      (Jul ’21)
        "OTLY",  # Oatly          (May ’21)
        "PATH",  # UiPath         (Apr ’21)
        "GRAB",  # Grab Holdings  (Dec ’21)
        "MQ",  # Marqeta        (Jun ’21)
        "CPNG",  # Coupang        (Mar ’21)
        "BIRD",  # Allbirds       (Nov ’21)
        "CHPT",  # ChargePoint    (Mar ’21)
        "ML",  # MoneyLion      (Jun ’21)
        "CLOV",  # Clover Health  (Jan ’21)
        "TOST",  # Toast, Inc.    (Sep ’21)
        # 2019 IPOs (still “new” going into 2022)
        "LYFT",  # Lyft           (Mar ’19)
        "UBER",  # Uber           (May ’19)
        "ZM",  # Zoom Video     (Apr ’19)
        "NET",  # Cloudflare     (Sep ’19)
        "BILL",  # Bill.com       (Dec ’19)
        "CRWD",  # CrowdStrike    (Jun ’19)
        "FAST",  # Fastly         (May ’19)
        "DOCU",  # DocuSign       (Apr ’18)
        "PTON",  # Peloton        (Sep ’19)
        "PINS",  # Pinterest      (Apr ’19)
    ]
    tickers = popular_stocks + upcoming_stocks

    # Define your backtest window
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Run the optimizer
    selected, weights, sharpe = optimize_portfolio(tickers, start_date, end_date)

    # Display results
    print("Selected tickers for maximum Sharpe:")
    for t, w in weights.items():
        if w > 0:
            print(f"  {t}: {w:.4f}")
    print(f"\nAchieved Sharpe Ratio: {sharpe:.2f}")
