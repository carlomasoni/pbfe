
#! --- INFO ---

'''
FUNCTION HELL 

maybe rename to prep dataset / labelling for clarity? 


run with python pbfe/src/prepfeatureset.py

loads features.csv and creates dual training and validation sets
(check paper split)

INPUTS: features.csv and clean_ohlc.csv

pivot features: 
- pivot indices
- pivot dates
- pivot prices


LOOKBACK WINDOW of 40:

from paper:

'
After completing the initial steps, we prepared our dataset
for deep learning training as follows: we segmented the
data into 40-row segments, using all columns as input
features. We chose 40 rows per segment to match the trading
day duration of 10 hours for EUA, based on our selected
15-minute timeframe. Subsequently, we shuffled both the
input and output sets and allocated 80% for training, 10% for
validation, and 10% for testing.
For datasets without our novel features (only OHLCV),
the distribution was as follows: training set (43,327, 40, 5),
validation set (5,415, 40, 5), and test set (5,417, 40, 5).
After incorporating our novel features, the number of samples
was slightly reduced to (42,621, 40, 16) for training, (5,327,
40, 16) for validation, and (5,329, 40, 16) for testing. This
reduction occurred because, during the calculation of our
novel features, the initial trend data was omitted to accurately
identify primary pivots. Importantly, the datasets remained
consistent, and no different dataset was used
'

-> all cols are input. 
-> 



-> from features.csv pattern row: 
		- pick end of pattern at pivot6idx at t_end
		- take a lookback window of length L at t_end -> input sequence X_seq
		- 11 pivot features becomes X_feat
		- define label y on future price 

create X_seq -> [OHLC_t_{l+1}, OHLC_t_{l+2}, ... OHLC_t]
shape (l,4)



Purpose:
- Load PBFE pattern features (features.csv) and clean/normalised OHLC data.
- Build supervised dataset for dual-branch PBFE model:
    * X_seq: 40-row OHLC sequence ending at pivot_6_date
    * X_feat: 11 PBFE pattern features (F1..F11)
    * y: ternary label based on future H-day log-return with data-driven dead-zone
- Perform chronological 80/10/10 split (train/val/test).

Inputs:
- pattern CSV: features.csv
    - pivot indices, dates, prices
    - F1..F11 pattern features
- normalised OHLC CSV: norm_pct.csv
    - Date, Open, High, Low, Close (pct or other normalisation)
- clean OHLC CSV: clean_ohlc.csv
    - Date, Close (real prices for labels)

Ternary label:
    r = log(C_{t+H} / C_t)

    y = +1 if  r >  dead_zone
    y }=  0 if -dead_zone <= r <= dead_zone
      }= -1 if  r < -dead_zone

dead_zone (epsilon) is chosen *automatically* from the empirical
distribution of |r| so that approximately `zero_target` fraction
of samples fall into the 0-class band.

'''

#! --- IMPORTS ---
from dataclasses import dataclass
from typing import Tuple, Literal, List, Optional

import numpy as np
import pandas as pd

#! --- DATA STRUCTURES ---

@dataclass
class PBFEDatasetConfig:
    L: int = 40    # lookback window length    
    H: int = 10    # horizon length

    # PCT OHLC 
    seq_date_col: str = "Date"
    seq_cols: Tuple[str, ...] = (
        "Open",
        "High",
        "Low",
        "Close")

    price_date_col: str = "Date"
    price_col: str = "Close"


    pattern_id_col: str = "pattern_id"
    pivot_completion_date_col: str = "pivot_6_date"  # “time t” of the pattern
    feat_cols: Tuple[str, ...] = tuple(f"F{i}" for i in range(1, 12))

    dead_zone: Optional[float] = None  # in log-return space
    zero_target: float = 0.5           # desired fraction of 0-class
    min_dead_zone: Optional[float] = None  # optional lower bound on epsilon
    max_dead_zone: Optional[float] = None  # optional upper bound on epsilon

#! --- FUNCTIONS ---

def ternary_label_from_logret(logret: float, dead_zone: float) -> int:
    if logret > dead_zone:
        return 1
    elif logret < -dead_zone:
        return -1
    else:
        return 0

def estimate_optimal_dead_zone(
    future_logrets: np.ndarray,
    zero_target: float = 0.5,
    min_eps: Optional[float] = None,
    max_eps: Optional[float] = None) -> float:

    r = np.asarray(future_logrets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("No valid future log-returns to estimate dead_zone.")

    abs_r = np.abs(r)
    eps = np.quantile(abs_r, zero_target)

    if min_eps is not None:
        eps = max(eps, min_eps)
    if max_eps is not None:
        eps = min(eps, max_eps)

    return float(eps)

def build_pbfe_dataset_from_files(
    pattern_csv_path: str = "pbfe/data/processed/features.csv",
    ohlc_norm_csv_path: str = "pbfe/data/normalised/norm_pct.csv",
    ohlc_clean_csv_path: str = "pbfe/data/processed/clean_ohlc.csv",
    config: PBFEDatasetConfig = PBFEDatasetConfig(),
    ):
    f"""
    Build PBFE dataset from pattern CSV, normalised OHLC CSV, and clean OHLC CSV.
    Two passes:
        Pass One

        - enforce calender history and future checks
        - collect data and sort by completion date
        - compute all future log returns

        - estimate optimal dead zone
        - apply ternary label

        Pass Two

        - using chosen dead zone construct sequence and feature matrices


    returns: 
        - X_seq: (N, L, D_seq) f32
        - X_feat: (N, D_feat) f32
        - y: (N,) int64 {-1,0,+1}
        - meta: (N, 3)
            - pattern_id
            - pivot_completion_date
            - future_log_return_H

            p.s. allow the heavy commenting because this is a nightmare function 
    """
    pattern_df = pd.read_csv(pattern_csv_path)
    ohlc_norm_df = pd.read_csv(ohlc_norm_csv_path)
    ohlc_clean_df = pd.read_csv(ohlc_clean_csv_path)

    pattern_df[config.pivot_completion_date_col] = pd.to_datetime(
        pattern_df[config.pivot_completion_date_col]
    )
    ohlc_norm_df[config.seq_date_col] = pd.to_datetime(
        ohlc_norm_df[config.seq_date_col]
    )
    ohlc_clean_df[config.price_date_col] = pd.to_datetime(
        ohlc_clean_df[config.price_date_col]
    )

    ohlc_norm_df = ohlc_norm_df.sort_values(config.seq_date_col).reset_index(drop=True)
    ohlc_clean_df = ohlc_clean_df.sort_values(config.price_date_col).reset_index(drop=True)

    norm_dates = ohlc_norm_df[config.seq_date_col].values
    clean_dates = ohlc_clean_df[config.price_date_col].values

    if not np.array_equal(norm_dates, clean_dates):
        raise ValueError(
            "ERROR: Normalised OHLC and clean OHLC must have identical Date calendars "
            "and ordering for this implementation."
        )

    date_to_idx = {pd.Timestamp(norm_dates[i]): i for i in range(len(norm_dates))}

    seq_values = ohlc_norm_df[list(config.seq_cols)].to_numpy(dtype=float)
    price_values = ohlc_clean_df[config.price_col].to_numpy(dtype=float)

    L = config.L
    H = config.H

    #imp === FIRST PASS ===
    valid_patterns: List[dict] = []
    future_logrets: List[float] = []

    for _, row in pattern_df.iterrows():
        completion_date = row[config.pivot_completion_date_col]

        #! checks for date, history and future
        if completion_date not in date_to_idx:
            raise ValueError(
                f"pivot_6_date {completion_date} is not present in the OHLC calendar. "
                f"Check that pattern CSV and OHLC filesuse the same date range."
            )

        t = date_to_idx[completion_date]

        if t < L - 1:
            continue

        if t + H >= len(price_values):
            continue

        price_t = price_values[t]
        price_future = price_values[t + H]
        if not np.isfinite(price_t) or not np.isfinite(price_future) or price_t <= 0:
            continue

        future_logret = float(np.log(price_future / price_t))
        if not np.isfinite(future_logret):
            continue

        #! store pattern row + index + future_logret
        valid_patterns.append(
            {
                "row": row,
                "t": t,
                "future_logret": future_logret,
            }
        )
        future_logrets.append(future_logret)

    if not valid_patterns:
        raise RuntimeError(
            "No valid patterns found in the dataset. "
            "Check L, H and data coverage."
        )

    future_logrets_arr = np.array(future_logrets, dtype=float)

    #! dead zone from func above 
    if config.dead_zone is None:
        dead_zone = estimate_optimal_dead_zone(
            future_logrets_arr,
            zero_target=config.zero_target,
            min_eps=config.min_dead_zone,
            max_eps=config.max_dead_zone,
        )
        print(f"[prepfeatureset] Estimated dead_zone={dead_zone:.6f} "
              f"(price move ~ {np.exp(dead_zone) - 1:.2%})")
    else:
        dead_zone = float(config.dead_zone)

    #imp === SECOND PASS ===
    X_seq_list: List[np.ndarray] = []
    X_feat_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []

    for vp in valid_patterns:
        row = vp["row"]
        t = vp["t"]
        future_logret = vp["future_logret"]

        
        seq_window = seq_values[t - L + 1 : t + 1]   # (L, D_seq)


        feat_vec = row[list(config.feat_cols)].to_numpy(dtype=float)

        label = ternary_label_from_logret(future_logret, dead_zone)

        X_seq_list.append(seq_window)
        X_feat_list.append(feat_vec)
        y_list.append(label)
        meta_rows.append(
            {
                config.pattern_id_col: row[config.pattern_id_col],
                config.pivot_completion_date_col: row[config.pivot_completion_date_col],
                "future_log_return_H": future_logret,
            }
        )

    X_seq = np.stack(X_seq_list).astype(np.float32)   # (N, L, D_seq)
    X_feat = np.stack(X_feat_list).astype(np.float32) # (N, D_feat)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    meta = meta.sort_values(config.pivot_completion_date_col).reset_index(drop=True)
    order = meta.index.to_numpy()

    X_seq = X_seq[order]
    X_feat = X_feat[order]
    y = y[order]

    return X_seq, X_feat, y, meta

def split(
    X_seq: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,):

    N = X_seq.shape[0]
    assert X_feat.shape[0] == N and y.shape[0] == N and len(meta) == N

    n_train = int(0.8 * N)
    n_val = int(0.9 * N)  # val = 10%, test = last 10%

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_val)
    idx_test = slice(n_val, N)

    splits = {
        "train": {
            "X_seq": X_seq[idx_train],
            "X_feat": X_feat[idx_train],
            "y": y[idx_train],
            "meta": meta.iloc[idx_train].reset_index(drop=True),
        },
        "val": {
            "X_seq": X_seq[idx_val],
            "X_feat": X_feat[idx_val],
            "y": y[idx_val],
            "meta": meta.iloc[idx_val].reset_index(drop=True),
        },
        "test": {
            "X_seq": X_seq[idx_test],
            "X_feat": X_feat[idx_test],
            "y": y[idx_test],
            "meta": meta.iloc[idx_test].reset_index(drop=True),
        },
    }
    return splits


