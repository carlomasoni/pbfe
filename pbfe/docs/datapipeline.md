# Pipeline for Data Flow. 

> daily.csv

### Step 1: Clean EOD 

-> remove missing data 
-> date ordering
-> numeric types
-> drop dupes 

> clean_ohlc.csv

### Step 2: Normalise and compare 

-> Normalise Data 
-> compare through training GRU 
-> Eval metrics and see which type to use. 

> norm_log/pct/zscore.csv

### Step 3: Price Series for detection 

-> Reconstruct Price series for pivot detection 
-> uses cleam_ohlc.csv
-> return pivot price series

### Step 4: Pivot Detection

-> pivot detection off ALGO 1 from paper 
-> P_t >= (1 + 0.05) P_pivot
-> P_t <= (1 - 0.05) P_pivot  

### Step 5: Form Pattern Units 

-> build 6 pivot patterns 
-> take window of 6 pivots and note index for step 6 

### Step 6: Pattern Features

-> Compute F1 to F11 from paper (e.q. 1 to 4)

### Step 7: Build Model Inputs

-> extract sequence window from normalised Data 

### Step 8: Build two datasets for Models 

-> Model A (Baseline), uses Normalised OHLC sequence from Step 2 

-> Model B (Enhanced), pivot-pattern features to same sequences 

# Pipeline Diagram: 

                RAW EOD OHLC (Date, Open, High, Low, Close)
                                  │
                                  ▼
                          CLEANED PRICE DATA
                                  │
                                  ├──────────────► NORMALISATION (PCT)
                                  │                   │
                                  │                   ▼
                                  │             NORMALISED DF
                                  │               (model input)
                                  │
                                  ▼
                        PIVOT-DETECTION PRICE SERIES
                                  │
                                  ▼
                    DETECT PIVOTS (±5%) — Algorithm 1
                                  │
                                  ▼
                     BUILD 6-PIVOT PATTERNS (P1..P6)
                                  │
                                  ▼
                      COMPUTE F1..F11 (Eq. 1–4)
                                  │
                                  ▼
         ┌─────────────────────────┴──────────────────────────┐
         │                                                    │
         ▼                                                    ▼
    SEQUENCE WINDOW (L bars)                            PATTERN FEATURES
    from NORMALISED DF                               (F1..F11, pivot idx)
    (Open_pct, High_pct, Low_pct, Close_pct)
         │                                                    │
         └──────────────► ALIGN BY LAST PIVOT (I6) ◄──────────┘
                                  │
                                  ▼
                      FINAL TRAINING DATASETS
                  
        Model A: X_seq(L,4)                    Model B: X_seq(L,4)
                                                    + F1..F11 (static)
                                  │
                                  ▼
                            ML TRAINING


