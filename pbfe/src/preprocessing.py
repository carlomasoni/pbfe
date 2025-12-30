#! --- INFO ---

'''
run with python pbfe/src/preprocessing.py

takes clean_ohlc.csv and normalises it using pct change, log return, and z score
then saves it to pbfe/data/normalised/
norm_pct.csv, norm_log.csv, norm_zscore.csv
this is used to create the input data for the LSTM and GRU models

'''

#! --- IMPORTS ---

import os
import pandas as pd
import numpy as np

#! --- CONFIG ---

CLEAN_FILE = os.path.join("pbfe", "data", "processed", "clean_ohlc.csv")

NORMALISED_DIR = os.path.join("pbfe", "data", "normalised")
OUT_PCT = os.path.join(NORMALISED_DIR, "norm_pct.csv")
OUT_LOG = os.path.join(NORMALISED_DIR, "norm_log.csv")
OUT_ZS  = os.path.join(NORMALISED_DIR, "norm_zscore.csv")

#! --- FUNCTIONS ---

def ensure_dirs():
    os.makedirs(NORMALISED_DIR, exist_ok=True)

def load_clean():
    if not os.path.exists(CLEAN_FILE):
        raise FileNotFoundError(f"Processed file not found: {CLEAN_FILE}. Run process_data.py first.")
    df = pd.read_csv(CLEAN_FILE, parse_dates=True, index_col=0)
    return df[["Open", "High", "Low", "Close"]].copy()

def _pct_change_all(df: pd.DataFrame) -> pd.DataFrame:
    pct = df.pct_change()
    pct = pct.fillna(0.0)
    return pct

def _log_return_all(df: pd.DataFrame) -> pd.DataFrame:
    log = np.log(df / df.shift(1))
    log = log.fillna(0.0)
    return log

def _zscore_on_returns_all(returns_df: pd.DataFrame) -> pd.DataFrame:
    zs = returns_df.copy()
    for col in zs.columns:
        col_vals = zs[col].astype(float)
        mu = col_vals.mean()
        sd = col_vals.std(ddof=0) or 1.0
        zs[col] = (col_vals - mu) / sd
    return zs

def compute_and_save_normalisations(df: pd.DataFrame):
    pct = _pct_change_all(df)
    log = _log_return_all(df)
    zs = _zscore_on_returns_all(pct)

    pct.to_csv(OUT_PCT)
    log.to_csv(OUT_LOG)
    zs.to_csv(OUT_ZS)

    return pct, log, zs

#! --- MAIN ---

def main():
    ensure_dirs()
    df = load_clean()
    pct, log, zs = compute_and_save_normalisations(df)
    print("Normalised data saved to pbfe/data/normalised/")

if __name__ == "__main__":
    main()