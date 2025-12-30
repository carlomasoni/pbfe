# ! --- INFO ---

'''
run with python pbfe/src/normtest.py
Tests which normalisation is the most effective for the LSTM and GRU models.
saves the results to pbfe/notebooks/normtest_results.csv and pbfe/notebooks/normtest_results.md
this is used to test the effectiveness of the LSTM and GRU models on the normalised data



NB: oneDNN custom operations are on. enable for reproducibility.

'''


# ! --- IMPORTS ---

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, GRU, Dense # pyright: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras import Input  # pyright: ignore[reportMissingImports]
import time



# ! --- CONFIG ---

RESULTS_DIR = "pbfe/notebooks"


NORMALISED_DIR = "pbfe/data/normalised"
NORM_PCT = os.path.join(NORMALISED_DIR, "norm_pct.csv")
NORM_LOG = os.path.join(NORMALISED_DIR, "norm_log.csv")
NORM_ZS  = os.path.join(NORMALISED_DIR, "norm_zscore.csv")


COLS = ["Open", "High", "Low", "Close"]
window = 20
# len of sequence for prediction

# ! --- FUNCTIONS ---

def ensure_dirs():
    os.makedirs(NORMALISED_DIR, exist_ok=True)

def labels_for_testing(df: pd.DataFrame, window=20 ) -> pd.DataFrame:
    x = []
    y = []
    data = df[COLS].values
    close = df["Close"].values
    for i in range(window, len(data)-1):
        x.append(data[i-window:i])
        y.append(1 if close[i + 1] > close[i] else 0)
    X = np.array(x)
    Y = np.array(y)
    return X, Y

#*X is window of [Open, High, Low, Close
#* Y is binary label is next close is higher than current close

def build_lstm(input_shape: tuple) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

#* sigmoid due to binary outcomes
#* 32 output units. not sure if we need more for now. 

def build_gru(input_shape: tuple) -> Sequential:
    model = Sequential([
    Input(shape=input_shape),
    GRU(32),
    Dense(1, activation="sigmoid")
])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# GRU is similar to LSTM but has no memory of previous states

def train_and_evaluate(model, X, Y):
    start_time = time.time()
    model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    training_time = time.time() - start_time
    y_pred = ( model.predict(X) > 0.5 ).astype(int)
    metrics = {
        "accuracy": accuracy_score(Y, y_pred),
        "f1": f1_score(Y, y_pred),
        "precision": precision_score(Y, y_pred),
        "recall": recall_score(Y, y_pred),
        "training_time": training_time
    }
    return metrics

#! --- MAIN ---

def main():
    ensure_dirs()
    results = []
    for norm in [NORM_PCT, NORM_LOG, NORM_ZS]:
        df = pd.read_csv(norm, index_col=0)
        X, Y = labels_for_testing(df, window=window)
        lstm_model = build_lstm((X.shape[1], X.shape[2]))
        lstm_metrics = train_and_evaluate(lstm_model, X, Y)
        gru_model = build_gru((X.shape[1], X.shape[2]))
        gru_metrics  = train_and_evaluate(gru_model, X, Y)
        results.append({
            "norm": norm,
            "lstm_metrics": lstm_metrics,
            "gru_metrics": gru_metrics
        })
    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv(os.path.join(RESULTS_DIR, "normtest_results.csv"), index=False)

    df_to_markdown = pd.read_csv(os.path.join(RESULTS_DIR, "normtest_results.csv"))

    with open(os.path.join(RESULTS_DIR, "normtest_results.md"), "w") as f:
        f.write(df_to_markdown.to_markdown(index=False))

if __name__ == "__main__":
    main()
