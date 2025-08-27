#!/usr/bin/env python3
# nvda_price_news_xgb_full.py
# --------------------------------------------------------------
#  Full-featured Price + News + Macro + Tech Indicators 3-class
#  direction classifier for NVDA
#
#  • yfinance closes & volumes for NVDA + 16 peers + SP500 + VIX
#  • Technical indicators: RSI, MACD, ATR, Bollinger Bands, volume spikes
#  • Cross-section return & volatility ranks
#  • Calendar features: day-of-week dummies, month cyclic
#  • News: count + sentence-transformer embeddings per day
#  • Macro: SP500 & VIX returns
#  • τ auto via quantile or override, robust to NaNs
#  • Tunable class-weight exponent
#  • Hyperparameter tuning via RandomizedSearchCV + TimeSeriesSplit
#  • Early stopping on validation log-loss
#  • Save model + plots + feature importances
# --------------------------------------------------------------
#  pip install yfinance pandas numpy scikit-learn xgboost
#  pip install sentence-transformers matplotlib seaborn ta
# --------------------------------------------------------------

import argparse
import datetime as dt
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "NVDA",
    "MSFT",
    "AMD",
    "ASML",
    "INTC",
    "GOOGL",
    "AMZN",
    "SMCI",
    "MU",
    "LIN",
    "EQIX",
    "DLR",
    "NEE",
    "CEG",
    "EXC",
    "LRCX",
    "KLAC",
]
MACRO_TICKERS = ["^GSPC", "^VIX"]
ALL_TICKERS = TICKERS + MACRO_TICKERS

# ───────── data fetch ─────────


def fetch_market_data(start):
    """Download adjusted Close and Volume for all tickers from start date."""
    df = yf.download(ALL_TICKERS, start=start, auto_adjust=True, progress=False)
    close = df["Close"].ffill().bfill()
    vol = df["Volume"].ffill().bfill()
    return close, vol


def fetch_news(start):
    """Aggregate Yahoo Finance headlines per day for tickers."""
    start_ts = int(pd.Timestamp(start).timestamp())
    news_map = {}
    for tk in TICKERS:
        for item in getattr(yf.Ticker(tk), "news", []):
            ts = item.get("providerPublishTime")
            ttl = item.get("title")
            if ts is None or ttl is None or ts < start_ts:
                continue
            d = dt.datetime.utcfromtimestamp(ts).date()
            news_map.setdefault(d, []).append(ttl)
    return news_map


# ───────── feature engineering ─────────


def compute_technical_indicators(close, vol):
    """Compute RSI, MACD, ATR, Bollinger Bands, and volume spike features for NVDA."""
    df = pd.DataFrame(index=close.index)
    price = close["NVDA"]
    # RSI(14)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["NVDA_RSI14"] = 100 - 100 / (1 + rs)
    # MACD & signal
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["NVDA_MACD"] = macd
    df["NVDA_MACD_signal"] = macd.ewm(span=9, adjust=False).mean()
    # ATR(14)
    high = yf.download("NVDA", start=close.index[0], auto_adjust=False, progress=False)[
        "High"
    ]
    low = yf.download("NVDA", start=close.index[0], auto_adjust=False, progress=False)[
        "Low"
    ]
    tr1 = high - low
    tr2 = (high - price.shift()).abs()
    tr3 = (low - price.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["NVDA_ATR14"] = tr.rolling(14).mean()
    # Bollinger Bands (20,2)
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    df["NVDA_BB_width"] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20
    # Volume spike: today's volume / 20-day avg volume
    df["NVDA_vol_spike"] = vol["NVDA"] / vol["NVDA"].rolling(20).mean()
    return df.fillna(0)


def compute_price_features(close):
    """Compute log returns, rolling vol, SMA gap and cross-section ranks."""
    log = np.log(close)
    feats = {}
    for tk in TICKERS:
        feats[f"{tk}_ret1"] = log[tk].diff()
        feats[f"{tk}_ret5"] = log[tk].diff(5)
        feats[f"{tk}_ret20"] = log[tk].diff(20)
        feats[f"{tk}_vol20"] = log[tk].diff().rolling(20).std() * np.sqrt(252)
        feats[f"{tk}_smaGap"] = (close[tk] - close[tk].rolling(5).mean()) / close[tk]
    pf = pd.DataFrame(feats).fillna(0)
    # Cross-section ranks for NVDA returns and vol
    for h in [1, 5, 20]:
        col = f"NVDA_ret{h}"
        cs = pf[[f"{tk}_ret{h}" for tk in TICKERS]].rank(axis=1, pct=True)
        pf[f"NVDA_ret{h}_csrank"] = cs[col]
    csv = pf[[f"{tk}_vol20" for tk in TICKERS]].rank(axis=1, pct=True)
    pf["NVDA_vol20_csrank"] = csv["NVDA_vol20"]
    return pf


def compute_macro_features(close):
    """Compute SP500 & VIX returns as macro indicators."""
    df = pd.DataFrame(index=close.index)
    for mk in MACRO_TICKERS:
        df[f"{mk}_ret1"] = np.log(close[mk]).diff()
        df[f"{mk}_ret5"] = np.log(close[mk]).diff(5)
        df[f"{mk}_vol20"] = np.log(close[mk]).diff().rolling(20).std() * np.sqrt(252)
    return df.fillna(0)


def compute_calendar_features(dates):
    """Day-of-week dummies and month as cyclic features."""
    df = pd.DataFrame(index=dates)
    df["dow"] = dates.weekday
    ohe = OneHotEncoder(categories="auto", drop="first", sparse_output=False)
    dow_dummies = ohe.fit_transform(df[["dow"]])
    for i, cat in enumerate(ohe.categories_[0][1:], start=1):
        df[f"dow_{cat}"] = dow_dummies[:, i - 1]
    month = dates.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df.fillna(0)


def embed_news(news_map, model):
    """Compute mean Transformer embedding + news count per day."""
    dim = model.get_sentence_embedding_dimension()
    zero = np.zeros(dim, np.float32)
    em = {}
    for d, titles in news_map.items():
        vecs = model.encode(titles, convert_to_numpy=True, show_progress_bar=False)
        em[pd.Timestamp(d)] = vecs.mean(axis=0).astype(np.float32)
    return em, zero


def build_feature_matrix(close, vol, news_map, args):
    """Assemble full feature DataFrame."""
    tech = compute_technical_indicators(close, vol)
    price = compute_price_features(close)
    macro = compute_macro_features(close)
    cal = compute_calendar_features(close.index)

    # embed news
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_map, emb_zero = embed_news(news_map, embedder)
    emb_mat = np.vstack([emb_map.get(d.date(), emb_zero) for d in close.index])
    emb_df = pd.DataFrame(
        emb_mat,
        index=close.index,
        columns=[f"news_emb_{i}" for i in range(emb_mat.shape[1])],
    )
    # news count
    cnt = pd.Series({pd.Timestamp(d): len(titles) for d, titles in news_map.items()})
    cnt = cnt.reindex(close.index.date, fill_value=0)
    cnt.index = close.index
    news_cnt = cnt.rename("news_count")

    # combine
    X = pd.concat([price, tech, macro, cal, emb_df, news_cnt], axis=1).fillna(0)

    # align so we can label
    X = X.iloc[: -args.horizon]
    return X


def make_labels(close, horizon, tau):
    """3-class labels: 0=Sell,1=Neutral,2=Buy."""
    log = np.log(close["NVDA"])
    fwd = log.shift(-horizon) - log
    return np.where(fwd < -tau, 0, np.where(fwd > tau, 2, 1)).astype(np.int8)[
        : len(log) - horizon
    ]


# ───────── main ─────────


def main(args):
    # 1) Fetch data
    print("Fetching market data…")
    close, vol = fetch_market_data(args.start)
    print("Fetching news…")
    news_map = fetch_news(args.start)

    # 2) Build features
    print("Building feature matrix…")
    X = build_feature_matrix(close, vol, news_map, args)

    # 3) Train/val/test split
    N = len(X)
    tr = int(0.7 * N)
    val = int(0.85 * N)

    # 4) Compute τ
    abs_r = np.abs(np.log(close["NVDA"]).diff(args.horizon)).dropna().iloc[:tr]
    if args.tau is None:
        tau = abs_r.quantile(args.tau_quantile)
    else:
        tau = args.tau
    if np.isnan(tau):
        raise RuntimeError("Computed τ is NaN – adjust horizon or data.")
    print(f"τ = {tau:.4f}")

    # 5) Labels
    y = make_labels(close, args.horizon, tau)
    # Sanity check
    counts = pd.Series(y[:tr]).value_counts().sort_index()
    print("Train label distribution:", dict(zip(["Sell", "Neutral", "Buy"], counts)))

    # 6) Scaling
    scaler = StandardScaler().fit(X.iloc[:tr])
    X_tr, X_val, X_te = (
        scaler.transform(df) for df in [X.iloc[:tr], X.iloc[tr:val], X.iloc[val:]]
    )
    y_tr, y_val, y_te = y[:tr], y[tr:val], y[val:]

    # 7) Class weights
    freq = np.bincount(y_tr, minlength=3) + 1e-8
    weights = (freq.max() / freq) ** args.weight_exponent
    sample_w = weights[y_tr]

    # 8) Model training / tuning
    if args.tune:
        print("Hyperparameter tuning…")
        param_dist = {
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
            "n_estimators": [200, 400, 800],
        }
        base = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            use_label_encoder=False,
            verbosity=0,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            base,
            param_dist,
            n_iter=20,
            cv=tscv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            refit=True,
        )
        search.fit(X_tr, y_tr, sample_weight=sample_w)
        clf = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            tree_method="hist",
            use_label_encoder=False,
            verbosity=0,
        )
        print("Training XGBoost with early stopping…")
        clf.fit(
            X_tr, y_tr, sample_weight=sample_w, eval_set=[(X_val, y_val)], verbose=False
        )

    # 9) Evaluation & plots
    y_val_hat = clf.predict(X_val)
    y_te_hat = clf.predict(X_te)
    bal_val = balanced_accuracy_score(y_val, y_val_hat)
    bal_te = balanced_accuracy_score(y_te, y_te_hat)
    print(f"Validation balanced acc: {bal_val:.3f}")
    print(f"Test       balanced acc: {bal_te:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_te_hat, labels=[0, 1, 2])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sell", "Neutral", "Buy"],
        yticklabels=["Sell", "Neutral", "Buy"],
    )
    plt.title(f"Confusion Matrix (bal {bal_te:.2f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved confusion_matrix.png")

    # Validation log-loss
    if hasattr(clf, "evals_result"):
        res = clf.evals_result().get("validation_0", {})
        ll = res.get("mlogloss")
        if ll:
            plt.plot(ll)
            plt.title("Validation Log-Loss")
            plt.tight_layout()
            plt.savefig("xgb_logloss.png")
            plt.close()
            print("Saved xgb_logloss.png")

    # Feature importances
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    imp.nlargest(20).sort_values().plot.barh()
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.close()
    print("Saved feature_importances.png")

    # 10) Save model
    out = f"xgb_nvda_priceNews_full_{int(time.time())}.json"
    clf.save_model(out)
    print("Model saved to", out)


# ─ CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NVDA direction classifier: price, tech, macro, news"
    )
    p.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for data (YYYY-MM-DD)",
    )
    p.add_argument(
        "--horizon", type=int, default=30, help="Forecast horizon in trading days"
    )
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Absolute τ override (skip quantile if set)",
    )
    p.add_argument(
        "--tau-quantile", type=float, default=0.33, help="Quantile for auto τ selection"
    )
    p.add_argument(
        "--weight-exponent",
        type=float,
        default=2.0,
        help="Exponent for class-weighting: (max/freq)^exp",
    )
    p.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    # default model params
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=2.0)
    args = p.parse_args()
    main(args)
