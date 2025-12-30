
# ! --- INFO ---
'''
detects highs and lows (pivots) in the data
pivot groups are grouped to form patterns 
'''




# ! --- IMPORTS ---

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ! --- DATA STRUCTURES---
@dataclass
class PivotPattern:
    '''
    class represents one 6 pivot pattern
    '''
    pivot_indices: List[int]
    pivot_dates: List[pd.Timestamp]
    pivot_prices: List[float]

# ! --- FUNCTIONS ---

def load_clean_ohlc(
    path: str = "pbfe/data/processed/clean_ohlc.csv",
    price_col: str = "Close"
    ) -> pd.DataFrame:

    #* loads clean OHLC data from a CSV file and returns a DataFrame

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if price_col not in df.columns:
        raise ValueError(f"Price column {price_col} not found in {path}")

    df["price"] = df[price_col].astype(float)
    return df

def detect_pivots(
    price: np.ndarray,
    fluct: float = 0.05
    ) -> Tuple[List[int], List[float]]:
    if len(price) == 0:
        return [], []
    
    pivot_indices = [0]
    pivot_prices = [float(price[0])]

    for t in range(1, len(price)):
        current = float(price[t])
        last_pivot_price = pivot_prices[-1]

        up_trigger   = current >= (1.0 + fluct) * last_pivot_price
        down_trigger = current <= (1.0 - fluct) * last_pivot_price

        if up_trigger or down_trigger:
            pivot_indices.append(t)
            pivot_prices.append(current)

    return pivot_indices, pivot_prices

def build_pivot_patterns(
    df: pd.DataFrame,
    pivot_indices: List[int],
    pivot_prices: List[float],
    pivots_per_pattern: int = 6
    )-> list[PivotPattern]:

    patterns: List[PivotPattern] = []

    if len(pivot_indices) < pivots_per_pattern:
        return patterns

    for start in range(0, len(pivot_indices) - pivots_per_pattern + 1):
        idx_slice = pivot_indices[start:start + pivots_per_pattern]
        price_slice = pivot_prices[start:start + pivots_per_pattern]
        dates_slice = [df.loc[i, "Date"] for i in idx_slice]

        pattern = PivotPattern(
            pivot_indices=idx_slice,
            pivot_dates=dates_slice,
            pivot_prices=price_slice
        )
        patterns.append(pattern)

    return patterns

def extract_features(
    path: str = "pbfe/data/processed/clean_ohlc.csv",
    price_col: str = "Close",
    fluct: float = 0.05,
    pivots_per_pattern: int = 6,
    ) -> Tuple[pd.DataFrame, List[PivotPattern]]:

    df = load_clean_ohlc(path=path, price_col=price_col)
    price = df["price"].values

    pivot_indices, pivot_prices = detect_pivots(price, fluct=fluct)
    patterns = build_pivot_patterns(df, pivot_indices, pivot_prices, pivots_per_pattern=pivots_per_pattern)

    print(f"Detected {len(pivot_indices)} pivots and {len(patterns)} {pivots_per_pattern}-pivot patterns.")
    return df, patterns

def div(a: float, b: float) -> float:
    '''
    safely divides a by b, returns NaN if b is 0
    '''
    return np.nan if b == 0 else a / b

def compute_features(pivot_prices: List[float]) -> np.ndarray:
    if len(pivot_prices) != 6:
        raise ValueError("expected 6 pivot prices, got {len(pivot_prices)}")

    P1, P2, P3, P4, P5, P6 = pivot_prices

    #! Short Term Features (F1 - F4)

    # Strength of latest move vs earliest move
    F1 = div(P6 - P5, P2 - P1) 
    # Strength of latest move vs 2nd leg
    F2 = div(P6 - P5, P3 - P2)
    # Strength of latest move vs 3rd leg
    F3 = div(P6 - P5, P4 - P3)
    # latest move vs previous move
    F4 = div(P6 - P5, P5 - P4)

    #! Medium Term Features (F5 - F6)
    F5 = div(P6 - P3, P2 - P1)
    F6 = div(P6 - P3, P3 - P2)
    #! Mid Term Features (F7 - F8)
    F7 = div(P5 - P4, P3 - P2)
    F8 = div(P6 - P5, P4 - P2)

    #! Long Term Features (F9 - F11)
    F9  = div(P3 - P2, P2 - P1)
    F10 = div(P4 - P3, P3 - P2)
    F11 = div(P5 - P4, P4 - P3)

    return np.array([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11], dtype=float)

def patterns2features(patterns: List[PivotPattern]) -> pd.DataFrame:
    rows = []

    for pid, pat in enumerate(patterns):
        row = {"pattern_id": pid}
    
        for i, (idx, date, price) in enumerate(
            zip(pat.pivot_indices, pat.pivot_dates, pat.pivot_prices), start =1
        ):
            row[f"pivot_{i}_idx"] = idx
            row[f"pivot_{i}_date"] = date
            row[f"pivot_{i}_price"] = price

        _features = compute_features(pat.pivot_prices)
        for i, f in enumerate(_features, start = 1):
            row[f"F{i}"] = f

        rows.append(row)

    return pd.DataFrame(rows)

def save_features(
    path: str,
    price_col: str = "Close",
    fluct: float = 0.05,
    pivots_per_pattern: int = 6,
    save_path: str = "pbfe/data/processed/features.csv")-> pd.DataFrame:
    
    df, patterns = extract_features(
        path=path,
        price_col=price_col,
        fluct=fluct,
        pivots_per_pattern=pivots_per_pattern,
    )

    features_df = patterns2features(patterns)

    if save_path is not None:
        features_df.to_csv(save_path, index=False)

    return features_df


