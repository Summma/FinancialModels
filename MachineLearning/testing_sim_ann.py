import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt


# download and prepare ohlcv data for each ticker with retry and backoff
def download_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    max_retries: int = 5,
    backoff_seconds: float = 2.0,
    min_rows: int = 50,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of per ticker DataFrames with columns:
    ['open','high','low','close','volume'] in lowercase.
    Skips tickers that fail or have too little data.
    """
    data_dict: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        attempt = 0
        while attempt <= max_retries:
            try:
                df = yf.download(
                    t,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,  # safer for rate limits when looping
                )
                if df is None or df.empty:
                    raise ValueError("Empty dataframe")
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.dropna(how="any", inplace=True)
                if len(df) < min_rows:
                    raise ValueError("Too few rows")
                data_dict[t] = df
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"Skip {t}: {e}")
                    break
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_for)

    return data_dict


# sharpe helper with simple annualization
def sharpe_ratio(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, trading_days: int = 252
) -> float:
    r = float(np.dot(mu, weights)) * trading_days
    v = float(np.sqrt(weights @ cov @ weights * trading_days))
    if v <= 0:
        return -np.inf
    return r / v


# neighbor generator for simulated annealing
def neighbor(
    weights: np.ndarray, step: float = 0.05, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = len(weights)
    if n < 2:
        return weights.copy()
    w2 = weights.copy()
    i, j = rng.choice(n, size=2, replace=False)
    delta = float(rng.uniform(-step, step))
    w2[i] += delta
    w2[j] -= delta
    w2 = np.maximum(w2, 0.0)
    s = w2.sum()
    if s == 0:
        return weights.copy()
    return w2 / s


# simulated annealing optimizer for portfolio weights
def simulated_annealing(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_iter: int = 10000,
    t0: float = 1.0,
    alpha: float = 0.995,
    step: float = 0.05,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    if len(mu) < 2:
        raise ValueError("Need at least two assets with valid returns for optimization")

    order = list(mu.index)
    mu_vec = mu.loc[order].to_numpy(dtype=float)
    cov_mat = cov.loc[order, order].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n = len(mu_vec)

    w_cur = np.ones(n, dtype=float) / n
    s_cur = sharpe_ratio(w_cur, mu_vec, cov_mat)
    w_best, s_best = w_cur.copy(), s_cur
    temp = t0

    for _ in range(max(n_iter, 1)):
        w_new = neighbor(w_cur, step=step, rng=rng)
        s_new = sharpe_ratio(w_new, mu_vec, cov_mat)
        if s_new > s_cur or rng.random() < np.exp((s_new - s_cur) / max(temp, 1e-12)):
            w_cur, s_cur = w_new, s_new
            if s_new > s_best:
                w_best, s_best = w_new.copy(), s_new
        temp *= alpha

    return w_best, s_best


# simple backtrader strategy that buys once using provided weights
class SAWeightsStrategy(bt.Strategy):
    params = dict(weights={})

    def __init__(self):
        self.placed = False

    def next(self):
        if self.placed:
            return
        cash = self.broker.getcash()
        for data in self.datas:
            w = float(self.p.weights.get(data._name, 0.0))
            if w <= 0:
                continue
            price = float(data.close[0])
            if price <= 0:
                continue
            size = int((cash * w) / price)
            if size > 0:
                self.buy(data=data, size=size)
        self.placed = True


if __name__ == "__main__":
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
        "CLOV",
        "TOST",
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

    start = "2022-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    # download data with retries and backoff to reduce rate limit issues
    data_dict = download_ohlcv(tickers, start, end)

    # ensure there is enough data to proceed
    if not data_dict:
        raise RuntimeError(
            "No data downloaded. Try reducing the ticker set or widening the date range."
        )

    # build returns dataframe from available tickers
    rets = pd.concat(
        [df["close"].pct_change().rename(t) for t, df in data_dict.items()],
        axis=1,
    ).dropna(how="any")

    # filter to assets with positive mean return
    mu_all = rets.mean()
    positive = mu_all[mu_all > 0].index.tolist()
    if len(positive) < 2:
        raise RuntimeError(
            "Not enough assets with positive mean returns. Adjust filters or tickers."
        )
    rets = rets[positive]
    mu = rets.mean()
    cov = rets.cov()
    opt_universe = positive

    # run simulated annealing optimizer
    w_best, s_best = simulated_annealing(mu, cov, n_iter=10000, seed=42)
    weights = dict(zip(opt_universe, w_best))

    print(f"Simulated Annealing Sharpe: {s_best:.4f}")
    print("Weights:")
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 1e-4:
            print(f"  {t}: {w:.4f}")

    # set up backtrader with the weighted assets
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(SAWeightsStrategy, weights=weights)

    for t, w in weights.items():
        if w <= 1e-4:
            continue
        df = data_dict.get(t)
        if df is None or df.empty:
            continue
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=pd.to_datetime(start),
            todate=pd.to_datetime(end),
            name=t,
        )
        cerebro.adddata(data)

    if len(cerebro.datas) < 1:
        raise RuntimeError(
            "No data feeds added to the engine. Check data availability and filters."
        )

    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    print(f"\nStarting Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final   Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()
