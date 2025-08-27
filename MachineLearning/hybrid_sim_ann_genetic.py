# hybrid_ga_sa.py

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime


def download_price_data(tickers, start_date, end_date):
    df = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )["Adj Close"]
    return df.dropna(how="all").ffill().dropna(how="any")


def calculate_mu_cov(returns):
    mu = returns.mean()
    cov = returns.cov()
    return mu, cov


def sharpe_ratio(w, mu, cov, ann=252):
    r = np.dot(mu, w) * ann
    v = np.sqrt(w.T @ cov @ w * ann)
    return r / v if v > 0 else 0.0


def generate_initial_population(pop_size, n):
    return [np.random.dirichlet(np.ones(n)) for _ in range(pop_size)]


def tournament_selection(pop, fits, k=3):
    new = []
    N = len(pop)
    for _ in range(N):
        idx = np.random.choice(N, k, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        new.append(pop[best].copy())
    return new


def blend_crossover(p1, p2):
    α = np.random.rand()
    c = α * p1 + (1 - α) * p2
    c = np.maximum(c, 0)
    return c / c.sum()


def mutate(w, rate=0.2, strength=0.05):
    if np.random.rand() < rate:
        noise = np.random.normal(0, strength, size=w.shape)
        w = np.maximum(w + noise, 0)
        return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return w


def prune(w, rate=0.5):
    mask = np.random.rand(len(w)) < rate
    w[mask] = 0
    return (w / w.sum()) if w.sum() > 0 else np.ones_like(w) / len(w)


def neighbor(w, step=0.02):
    w2 = w.copy()
    i, j = np.random.choice(len(w), 2, False)
    δ = np.random.uniform(-step, step)
    w2[i] += δ
    w2[j] -= δ
    w2 = np.maximum(w2, 0)
    return w2 / w2.sum()


def simulated_annealing_refine(w_init, mu, cov, n_iter=200, T0=0.5, α=0.9, step=0.02):
    w = w_init.copy()
    score = sharpe_ratio(w, mu, cov)
    best_w, best_s = w.copy(), score
    T = T0
    for _ in range(n_iter):
        w_new = neighbor(w, step)
        s_new = sharpe_ratio(w_new, mu, cov)
        if s_new > score or np.random.rand() < np.exp((s_new - score) / T):
            w, score = w_new, s_new
            if s_new > best_s:
                best_w, best_s = w_new.copy(), s_new
        T *= α
    return best_w, best_s


def hybrid_ga_sa(
    mu,
    cov,
    tickers,
    pop_size=100,
    gens=100,
    tour_k=3,
    cross_rate=0.8,
    mut_rate=0.2,
    mut_str=0.05,
    prune_rate=0.5,
    sa_iter=200,
    sa_T0=0.5,
    sa_alpha=0.9,
    sa_step=0.02,
):
    n = len(mu)
    pop = generate_initial_population(pop_size, n)
    fits = [sharpe_ratio(w, mu, cov) for w in pop]
    best_idx = np.argmax(fits)
    best_w, best_s = pop[best_idx].copy(), fits[best_idx]

    for gen in range(1, gens + 1):
        # 1) Selection
        sel = tournament_selection(pop, fits, k=tour_k)
        # 2) Crossover → Mutation → Prune → SA-refine
        new_pop = []
        for i in range(0, pop_size, 2):
            p1, p2 = sel[i], sel[(i + 1) % pop_size]
            if np.random.rand() < cross_rate:
                c1 = blend_crossover(p1, p2)
                c2 = blend_crossover(p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                c = mutate(c, mut_rate, mut_str)
                c = prune(c, prune_rate)
                c, sc = simulated_annealing_refine(
                    c, mu, cov, n_iter=sa_iter, T0=sa_T0, α=sa_alpha, step=sa_step
                )
                new_pop.append(c)
        pop = new_pop[:pop_size]
        fits = [sharpe_ratio(w, mu, cov) for w in pop]
        # track best
        idx = np.argmax(fits)
        if fits[idx] > best_s:
            best_s = fits[idx]
            best_w = pop[idx].copy()
        if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen}/{gens}  Best Sharpe: {best_s:.4f}")

    # map back to tickers
    return dict(zip(tickers, best_w)), best_s


if __name__ == "__main__":
    # your tickers

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

    start = "2022-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    # download & prep
    df = download_price_data(tickers, start, end)
    rets = df.pct_change().dropna()
    mu, cov = calculate_mu_cov(rets)

    # run hybrid
    weights, sharpe = hybrid_ga_sa(
        mu,
        cov,
        tickers,
        pop_size=100,
        gens=100,
        tour_k=3,
        cross_rate=0.8,
        mut_rate=0.2,
        mut_str=0.05,
        prune_rate=0.5,
        sa_iter=100,
        sa_T0=0.5,
        sa_alpha=0.9,
        sa_step=0.02,
    )

    print("\nOptimal weights:")
    for t, w in weights.items():
        if w > 0:
            print(f"  {t}: {w:.4f}")
    print(f"\nAchieved Sharpe Ratio: {sharpe:.4f}")
