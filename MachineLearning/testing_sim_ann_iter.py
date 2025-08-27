import yfinance as yf
import numpy as np
import pandas as pd
import backtrader as bt
from datetime import datetime
from dateutil.relativedelta import relativedelta


def sharpe_ratio(w, mu, cov, ann=252):
    r = np.dot(mu, w) * ann
    v = np.sqrt(w @ cov @ w * ann)
    return r / v if v > 0 else -np.inf


def neighbor(w, step=0.05):
    w2 = w.copy()
    if len(w2) < 2:
        return w2
    i, j = np.random.choice(len(w2), 2, replace=False)
    delta = np.random.uniform(-step, step)
    w2[i] += delta
    w2[j] -= delta
    w2 = np.maximum(w2, 0)
    s = w2.sum()
    return w2 / s if s > 0 else w


def simulated_annealing(mu, cov, n_iter=20000, T0=1.0, alpha=0.995, step=0.05):
    n = len(mu)
    if n < 2:
        return np.ones(n), -np.inf
    w_cur = np.ones(n) / n
    s_cur = sharpe_ratio(w_cur, mu, cov)
    w_best, s_best = w_cur.copy(), s_cur
    T = T0
    for _ in range(n_iter):
        w_new = neighbor(w_cur, step)
        s_new = sharpe_ratio(w_new, mu, cov)
        if s_new > s_cur or np.random.rand() < np.exp((s_new - s_cur) / max(T, 1e-12)):
            w_cur, s_cur = w_new, s_new
            if s_new > s_best:
                w_best, s_best = w_new.copy(), s_new
        T *= alpha
    return w_best, s_best

class SimAnnMonthlyRebalance(bt.Strategy):
    params = dict(
        tickers=[],  # list of tickers to consider
        lookback_years=2,  # rolling window length
        n_iter=20000,  # SA iterations
        T0=1.0,  # SA initial temperature
        alpha=0.995,  # SA cooling rate
        step=0.05,  # SA neighbor step-size
    )

    def __init__(self):
        self.last_month = None
        self.weights = {t: 0.0 for t in self.p.tickers}
        self.data_map = {data._name: data for data in self.datas}

    def next(self):
        dt = self.datas[0].datetime.date(0)
        # first trading day of a new month
        if dt.month != self.last_month and dt.day == 1:
            executed = self.rebalance(dt)  # only mark month if we actually rebalanced
            if executed:
                self.last_month = dt.month

    def _history_panel(self, cutoff_date):
        """
        Build a wide dataframe of closes for all feeds using only bars before cutoff_date.
        """
        series = []
        for data in self.datas:
            n = len(data)
            if n < 2:
                continue
            # full series up to current sim time
            idx = [bt.num2date(ts) for ts in data.datetime.get(size=n)]
            s = pd.Series(
                data.close.get(size=n), index=pd.to_datetime(idx), name=data._name
            )
            # strictly prior to cutoff_date to avoid peeking at current bar
            s = s[s.index.date < cutoff_date]
            if len(s):
                series.append(s)
        if not series:
            return pd.DataFrame()
        df = pd.concat(series, axis=1).sort_index().ffill().dropna(how="any")
        return df

    def rebalance(self, as_of_date):
        # define rolling window
        end = as_of_date
        start = end - relativedelta(years=self.p.lookback_years)

        # build adjusted close panel from existing feeds only (no web calls)
        panel = self._history_panel(end)
        if panel.empty:
            return False

        # trim to rolling window
        panel = panel.loc[(panel.index.date >= start) & (panel.index.date < end)]
        if panel.shape[0] < 2 or panel.shape[1] < 2:
            return False

        # compute returns
        returns = panel.pct_change().dropna(how="any")
        if returns.empty:
            return False

        # filter by positive mean
        mu_all = returns.mean()
        positive = mu_all[mu_all > 0].index.tolist()
        if len(positive) < 2:
            return False

        returns = returns[positive]
        mu = returns.mean().to_numpy()
        cov = returns.cov().to_numpy()

        # run SA
        w_vec, s_val = simulated_annealing(
            mu,
            cov,
            n_iter=self.p.n_iter,
            T0=self.p.T0,
            alpha=self.p.alpha,
            step=self.p.step,
        )

        # build new weights dict
        new_weights = {t: 0.0 for t in self.p.tickers}
        for t, w in zip(positive, w_vec):
            new_weights[t] = float(w)

        # liquidate existing positions
        for t, data in self.data_map.items():
            if self.getposition(data).size:
                self.close(data=data)

        # deploy new weights using current bar price for sizing; fills on next bar
        cash = float(self.broker.getcash())
        for t, w in new_weights.items():
            if w <= 0:
                continue
            data = self.data_map.get(t)
            if data is None:
                continue
            price = float(data.close[0])
            if price <= 0:
                continue
            size = int((cash * w) / price)
            if size > 0:
                self.buy(data=data, size=size)

        self.weights = new_weights
        print(f"\nRebalanced on {as_of_date} to Sharpe ~ {s_val:.4f}")
        for t, w in new_weights.items():
            if w > 1e-3:
                print(f"  {t}: {w:.3f}")
        return True


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

    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(
        SimAnnMonthlyRebalance,
        tickers=tickers,
        lookback_years=2,
        n_iter=10000,
        T0=1.0,
        alpha=0.995,
        step=0.05,
    )

    # download once
    raw = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )

    # add feeds
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        names = [c for c in set(raw.columns.get_level_values(1)) if c in tickers]
        for t in names:
            try:
                df = raw.xs(t, axis=1, level=1)[fields].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.dropna(how="any", inplace=True)
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                data = bt.feeds.PandasData(
                    dataname=df,
                    fromdate=pd.to_datetime(start_date),
                    todate=pd.to_datetime(end_date),
                    name=t,
                )
                cerebro.adddata(data)
            except Exception:
                pass
    else:
        df = raw[fields].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.dropna(how="any", inplace=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=pd.to_datetime(start_date),
            todate=pd.to_datetime(end_date),
            name=tickers[0],
        )
        cerebro.adddata(data)

    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    print(f"\n=== Starting Backtest: {start_date} → {end_date} ===")
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value:    {cerebro.broker.getvalue():.2f}")
    # cerebro.plot()
