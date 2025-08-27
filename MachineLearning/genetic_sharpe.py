import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def download_price_data(tickers, start_date, end_date):
    df = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
    )["Adj Close"]
    df = df.dropna(how="all").ffill().dropna(how="any")
    return df


def calculate_mu_cov(returns):
    mu = returns.mean()
    cov = returns.cov()
    return mu, cov


def sharpe_ratio(weights, mu, cov, periods_per_year=252):
    port_return = np.dot(mu, weights) * periods_per_year
    port_vol = np.sqrt(weights.T @ cov @ weights * periods_per_year)
    return port_return / port_vol if port_vol > 0 else 0.0


def generate_initial_population(pop_size, n_assets):
    return [np.random.dirichlet(np.ones(n_assets)) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=3):
    pop_size = len(population)
    new_pop = []
    for _ in range(pop_size):
        contestants = np.random.choice(pop_size, k, replace=False)
        best = contestants[np.argmax([fitnesses[i] for i in contestants])]
        new_pop.append(population[best].copy())
    return new_pop


def blend_crossover(p1, p2):
    alpha = np.random.rand()
    child = alpha * p1 + (1 - alpha) * p2
    child = np.maximum(child, 0)
    return child / child.sum()


def mutate(weights, mutation_rate=0.2, mutation_strength=0.05):
    if np.random.rand() < mutation_rate:
        noise = np.random.normal(0, mutation_strength, size=weights.shape)
        w = weights + noise
        w = np.maximum(w, 0)
        if w.sum() == 0:
            return np.ones_like(w) / len(w)
        return w / w.sum()
    return weights


def prune(weights, prune_rate=0.5):
    """
    With high probability prune (zero out) each weight.
    Then renormalize to sum to 1, or return equal weights if all zero.
    """
    w = weights.copy()
    mask = np.random.rand(len(w)) < prune_rate
    w[mask] = 0
    if w.sum() == 0:
        return np.ones_like(w) / len(w)
    return w / w.sum()


def genetic_algorithm(
    mu,
    cov,
    pop_size=100,
    generations=200,
    tournament_k=3,
    crossover_rate=0.8,
    mutation_rate=0.2,
    mutation_strength=0.05,
    prune_rate=0.5,
):
    """
    GA to maximize Sharpe, with a high chance to zero out any ticker.
    """
    n = len(mu)
    population = generate_initial_population(pop_size, n)
    fitnesses = [sharpe_ratio(w, mu, cov) for w in population]
    best_idx = np.argmax(fitnesses)
    best_w, best_sharpe = population[best_idx].copy(), fitnesses[best_idx]

    for gen in range(1, generations + 1):
        # Selection
        selected = tournament_selection(population, fitnesses, k=tournament_k)

        # Crossover, Mutation, Prune
        next_pop = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % pop_size]
            if np.random.rand() < crossover_rate:
                c1 = blend_crossover(p1, p2)
                c2 = blend_crossover(p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutate then prune
            c1 = prune(mutate(c1, mutation_rate, mutation_strength), prune_rate)
            c2 = prune(mutate(c2, mutation_rate, mutation_strength), prune_rate)
            next_pop.extend([c1, c2])

        population = next_pop[:pop_size]
        fitnesses = [sharpe_ratio(w, mu, cov) for w in population]

        # Update best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_sharpe = fitnesses[gen_best_idx]
        if gen_best_sharpe > best_sharpe:
            best_sharpe = gen_best_sharpe
            best_w = population[gen_best_idx].copy()

        if gen % 20 == 0 or gen == 1:
            print(f"Gen {gen}/{generations}  Best Sharpe: {best_sharpe:.4f}")

    return best_w, best_sharpe


if __name__ == "__main__":
    # Example usage: replace with your tickers

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

    # 1) Download price data
    prices = download_price_data(tickers, start, end)

    # 2) Compute daily returns, mu, cov
    returns = prices.pct_change().dropna()
    mu, cov = calculate_mu_cov(returns)

    # 3) Run the genetic algorithm
    best_weights, best_sharpe = genetic_algorithm(
        mu,
        cov,
        pop_size=100,
        generations=1000,
        tournament_k=3,
        crossover_rate=0.6,
        mutation_rate=0.9,
        mutation_strength=0.15,
        prune_rate=0.8,
    )

    # 4) Display results
    print(f"\nOptimal Sharpe Ratio: {best_sharpe:.4f}")
    print("Optimal weights:")
    for ticker, w in zip(tickers, best_weights):
        if w > 0:
            print(f"  {ticker}: {w:.4f}")
