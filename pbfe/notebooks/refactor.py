import pandas as pd
import ast
import os

def highlight_max(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    df_highlighted = df.copy().astype(str)
    for col in df.columns:
        if col in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            max_val = df[col].astype(float).max()
            mask = df[col].astype(float) == max_val
            df_highlighted.loc[mask, col] = '*' + df_highlighted.loc[mask, col] + '*'
    return df_highlighted

def main():
    os.makedirs('notebooks', exist_ok=True)
    df = pd.read_csv("pbfe/notebooks/normtest_results.csv")
    def parse_metrics_column(metrics_col):
        return metrics_col.apply(ast.literal_eval).apply(pd.Series)
    lstm_metrics = parse_metrics_column(df['lstm_metrics'])
    gru_metrics = parse_metrics_column(df['gru_metrics'])
    df_flat = pd.concat([
        df['norm'],
        lstm_metrics.add_prefix('lstm_'),
        gru_metrics.add_prefix('gru_'),
    ], axis=1)
    df_flat['norm'] = df_flat['norm'].str.replace('pbfe/data/normalised/', '')
    df_flat = df_flat.round(3)
    df_highlighted = highlight_max(df_flat, exclude_columns=['norm'])
    print(df_highlighted)
    with open('pbfe/notebooks/normresults.md', 'w') as f:
        f.write('# Model Results\n\n')
        f.write(df_highlighted.to_markdown(index=False))

if __name__ == "__main__":
    main()


