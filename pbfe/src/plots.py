# ! --- INFO ---

'''
run with python pbfe/src/plots.py
Plots ONLY — no CSVs saved.
plots the raw close, z score, pct change, and log return
saves them to pbfe/data/plots/
raw_close.png, norm_hist_zscore.png, norm_hist_pct.png, norm_hist_log.png
this is used to visualise the data before it is used to train the LSTM and GRU models
'''

# ! --- IMPORTS ---

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

# ! --- CONFIG ---

PROCESSED_DIR = "pbfe/data/processed"
DATA4PLOTS_DIR = "pbfe/data/plots"
FIG_DIR = "pbfe/data/plots"

CLEAN_FILE = os.path.join(PROCESSED_DIR, "clean_ohlc.csv")

FIG_RAW      = os.path.join(FIG_DIR, "raw_close.png")
FIG_HIST_ZS  = os.path.join(FIG_DIR, "norm_hist_zscore.png")
FIG_HIST_PCT = os.path.join(FIG_DIR, "norm_hist_pct.png")
FIG_HIST_LOG = os.path.join(FIG_DIR, "norm_hist_log.png")

# ! --- FUNCTIONS ---

def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA4PLOTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def load_clean():
    if not os.path.exists(CLEAN_FILE):
        raise FileNotFoundError(f"Processed file not found: {CLEAN_FILE}. Run process_data.py first.")
    df = pd.read_csv(CLEAN_FILE, parse_dates=True, index_col=0)
    return df[["Open", "High", "Low", "Close"]].copy()

def calc_returns(df: pd.DataFrame):
    close = df["Close"].astype(float)
    pct_returns = close.pct_change().dropna()
    pct_returns.name = "PctReturn"
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns.name = "LogReturn"
    return pct_returns, log_returns

def calc_zscore_close(df: pd.DataFrame):
    close = df["Close"].astype(float)
    mu = close.mean()
    sd = close.std(ddof=0) or 1.0
    z = (close - mu) / sd
    z.name = "ZScoreClose"
    return z

def _plot_hist(series: pd.Series, title: str, outfile: str, bins: int = 100):
    plt.figure()
    plt.hist(series.dropna().values, bins=bins, density=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()

def plot_all(df: pd.DataFrame, pct_returns: pd.Series, log_returns: pd.Series, zscore_close: pd.Series):

    _plot_hist(df["Close"].astype(float), "PDF of Raw Close (Price Levels)", FIG_RAW)

    _plot_hist(zscore_close, "PDF of Z-Score (Price Levels)", FIG_HIST_ZS)

    _plot_hist(pct_returns, "PDF of Percent-Change Returns", FIG_HIST_PCT)

    _plot_hist(log_returns, "PDF of Log Returns", FIG_HIST_LOG)

#! --- MAIN ---

def main():
    ensure_dirs()
    df = load_clean()
    pct_returns, log_returns = calc_returns(df)
    zscore_close = calc_zscore_close(df)

    plot_all(df, pct_returns, log_returns, zscore_close)
    print("Plots saved to pbfe/data/plots/")

if __name__ == "__main__":
    main()
