
#! --- INFO ---

'''
run with python pbfe/src/utils.py

takes raw csv with EOD data and cleans it
then saves it to pbfe/data/processed/clean_ohlc.csv
'''

#! --- IMPORTS ---

from re import A
import os, time, math
import numpy as np
import pandas as pd
import argparse




#! --- CONFIG ---

CSV_PATH = "pbfe/data/raw/daily.csv"
DATE_COL = "Date"
OUT_DIR = "pbfe/data/processed"


#! --- UTILS ---

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_eod_csv(path: str, date_col: str): 
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' in CSV columns: {df.columns.tolist()}")

    must = ["Open", "High", "Low", "Close"]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    return df[["Open","High","Low","Close"]].copy()

def basic_bar_sanity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    #* remove invalid prices (negative or zero)
    rem = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    df = df[~rem]
    
    high_fix = df[["Open", "High", "Low", "Close"]].max(axis=1)
    low_fix = df[["Open", "High", "Low", "Close"]].min(axis=1)
    df["High"] = np.maximum(df["High"], high_fix)
    df["Low"] = np.minimum(df["Low"], low_fix)
    #* dedupe and sort 
    df = df[~df.index.duplicated(keep='last')]

    df = df.sort_index()

    return df

def ohlc_quality_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = (
        (df[["Open", "High", "Low", "Close"]].notnull().all(axis=1)) &
        (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1) &
        (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)) &
        (df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)) &
        (df["High"] >= df["Low"])
    )
    df_valid = df[mask]
    print(f"Removed {len(df) - len(df_valid)} invalid rows.")
    return df_valid

#! --- MAIN ---

def main():
    ensure_dir(OUT_DIR)

    df_raw = load_eod_csv(CSV_PATH, DATE_COL)
    print(f"Loaded raw data: {len(df_raw)} rows.")

    df_clean = basic_bar_sanity(df_raw)
    print(f"After basic_bar_sanity: {len(df_clean)} rows (dropped {len(df_raw) - len(df_clean)})")

    df_processed = ohlc_quality_check(df_clean)
    print(f"After ohlc_quality_check: {len(df_processed)} rows (dropped {len(df_clean) - len(df_processed)})")

    out_path = os.path.join(OUT_DIR, "clean_ohlc.csv")
    df_processed.to_csv(out_path, index=True)
    print(f"Saved cleaned OHLC to: {out_path}  (rows={len(df_processed)})")


if __name__ == "__main__":
    main()









