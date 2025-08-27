# simulated_annealing_sharpe_filtered.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def download_price_data(tickers, start_date, end_date):
    """
    Download daily adjusted close prices for given tickers,
    forward-fill and drop any remaining NaNs.
    """
    df = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )["Adj Close"]
    return df.dropna(how="all").ffill().dropna(how="any")


def calculate_mu_cov(returns):
    """Calculate expected daily returns (mu) and covariance matrix (cov)."""
    mu = returns.mean()
    cov = returns.cov()
    return mu, cov


def sharpe_ratio(weights, mu, cov, periods_per_year=252):
    """
    Compute annualized Sharpe ratio for a given weight vector,
    assuming risk-free rate = 0.
    """
    port_return = np.dot(mu, weights) * periods_per_year
    port_vol = np.sqrt(weights.T @ cov @ weights * periods_per_year)
    return port_return / port_vol if port_vol > 0 else -np.inf


def neighbor(weights, step=0.05):
    """
    Generate a neighboring portfolio by shifting a small random
    amount between two randomly chosen assets, enforce non-negativity
    and normalization to sum to 1.
    """
    w = weights.copy()
    n = len(w)
    i, j = np.random.choice(n, 2, replace=False)
    delta = np.random.uniform(-step, step)
    w[i] += delta
    w[j] -= delta
    # enforce non-negativity
    w = np.maximum(w, 0)
    return w / w.sum()


def simulated_annealing(mu, cov, n_iter=10000, T0=1.0, alpha=0.995, step=0.05):
    """
    Simulated annealing to maximize Sharpe ratio.
    - mu: pd.Series of expected daily returns
    - cov: pd.DataFrame covariance of daily returns
    """
    n = len(mu)
    # start with equal weights
    current_w = np.ones(n) / n
    current_score = sharpe_ratio(current_w, mu, cov)
    best_w, best_score = current_w.copy(), current_score

    T = T0
    for k in range(n_iter):
        new_w = neighbor(current_w, step=step)
        new_score = sharpe_ratio(new_w, mu, cov)
        # accept if better, or with probability exp((new_score - current_score)/T)
        if new_score > current_score or np.random.rand() < np.exp(
            (new_score - current_score) / T
        ):
            current_w, current_score = new_w, new_score
            if new_score > best_score:
                best_w, best_score = new_w.copy(), new_score
        # cool down
        T *= alpha

    return best_w, best_score


if __name__ == "__main__":
    # Full universe
    tickers = [
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
        # 2020 IPOs
        "SNOW",
        "ABNB",
        "DASH",
        "PLTR",
        "LMND",
        "ASAN",
        "AI",
        "U",
        "ROOT",
        "GDRX",
        # 2021 IPOs
        "COIN",
        "RBLX",
        "AFRM",
        "BMBL",
        "SOFI",
        "RIVN",
        "HOOD",
        "OTLY",
        "PATH",
        "GRAB",
        "MQ",
        "CPNG",
        "BIRD",
        "CHPT",
        "ML",
        "CLOV",
        "TOST",
        # 2019+ IPOs
        "LYFT",
        "UBER",
        "ZM",
        "NET",
        "BILL",
        "CRWD",
        "FAST",
        "DOCU",
        "PTON",
        "PINS",
    ]

    # start = "2022-01-01"
    # end   = datetime.today().strftime("%Y-%m-%d")
    start = "2021-01-01"
    end = "2024-01-01"

    # 1) Download price data
    prices = download_price_data(tickers, start, end)

    # 2) Compute daily returns
    returns = prices.pct_change().dropna()

    # 3) Filter out tickers with non-positive expected return
    mu_all = returns.mean()
    positive = mu_all[mu_all > 0].index.tolist()
    returns = returns[positive]
    mu, cov = calculate_mu_cov(returns)
    tickers = positive

    print(f"Optimizing over {len(tickers)} tickers (positive μ only).")

    # 4) Run simulated annealing
    best_weights, best_sharpe = simulated_annealing(
        mu, cov, n_iter=200000, T0=1.0, alpha=0.995, step=0.05
    )

    # 5) Compute realized volatility
    port_rets = returns.dot(best_weights)
    annualized_vol = port_rets.std() * np.sqrt(252)

    # 6) Display results
    print(f"\nAchieved Sharpe Ratio:      {best_sharpe:.4f}")
    print(f"Realized Annual Volatility: {annualized_vol:.2%}\n")
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, best_weights):
        if weight > 1e-4:
            print(f"  {ticker}: {weight:.4f}")
