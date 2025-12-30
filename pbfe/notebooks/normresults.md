# Model Results

| norm                                 | lstm_accuracy   | lstm_f1   | lstm_precision   | lstm_recall   | lstm_training_time   | gru_accuracy   | gru_f1   | gru_precision   | gru_recall   | gru_training_time   |
|:-------------------------------------|:----------------|:----------|:-----------------|:--------------|:---------------------|:---------------|:---------|:----------------|:-------------|:--------------------|
| pbfe/data/normalised\norm_pct.csv    | 0.511           | *0.672*   | 0.508            | *0.994*       | 3.228                | 0.517          | *0.625*  | 0.513           | *0.799*      | *3.629*             |
| pbfe/data/normalised\norm_log.csv    | 0.514           | 0.667     | 0.509            | 0.967         | *3.367*              | 0.521          | 0.547    | 0.522           | 0.575        | 3.54                |
| pbfe/data/normalised\norm_zscore.csv | *0.527*         | 0.579     | *0.525*          | 0.645         | 3.279                | *0.531*        | 0.6      | *0.526*         | 0.697        | 3.605               |





# Results Analysis 

## --- PCT ---
-> best F1 and Recall for both

## --- Z-Score --- 

-> Best accuracy and precision for both 

## --- Log ---

-> fastest training time but nothing else (and only for LSTM) 
-> Discounted first. 

# Conclusion 

for LSTM: PCT > log > Z Score 

for GRU:  PCT > log > Z score 

Reasoning: PCT has top F1 and recall for both LSTM and GRU 
-> should be able to capture events better compared to accuracy 
