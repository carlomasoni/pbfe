#! --- INFO ---

"""
pbfe/main.py

Steps:
1. Clean raw EOD data -> pbfe/data/processed/clean_ohlc.csv
2. Normalise OHLC (pct, log, zscore) -> pbfe/data/normalised/*.csv
3. (Optional) Generate plots -> pbfe/data/plots/*.png
4. Run normalisation test (LSTM & GRU)
5. Run pivot detection + feature extraction
6. Build PBFE dataset (X_seq, X_feat, y, meta) and 80/10/10 splits
7. Keras hyperparameter tuning for Model A (seq only) and/or Model B (seq+PBFE)
8. Model training and evaluation
"""

#! --- IMPORTS ---
import os
from dataclasses import dataclass

import numpy as np

from pbfe.src import utils
from pbfe.src import preprocessing
from pbfe.src import feature_extraction
from pbfe.src import normtest as normtest_module
from pbfe.src import plots
from pbfe.src import prepfeatureset
from pbfe.src import hyp
from pbfe.src import train_eval


#! --- CONFIG ---

@dataclass
class PipelineConfig:
    run_plots: bool = False
    run_normtest: bool = False
    run_pivots_step: bool = True
    run_dataset_prep: bool = True
    run_tuning_model_a: bool = False
    run_tuning_model_b: bool = False
    run_train_eval_models: bool = True

    clean_ohlc_path: str = "pbfe/data/processed/clean_ohlc.csv"
    features_out_path: str = "pbfe/data/processed/features.csv"

    norm_pct_path: str = "pbfe/data/normalised/norm_pct.csv"
    dataset_out_dir: str = "pbfe/data/datasets"

    price_col: str = "Close"
    fluct: float = 0.02
    pivots_per_pattern: int = 6

    L: int = 20
    H: int = 5
    zero_target: float = 0.4

CONFIG = PipelineConfig()


#! --- PIPELINE FUNCTIONS ---

def raw2processed(config: PipelineConfig):
    print("\n[STEP 1] Cleaning raw EOD data...")
    utils.main()
    print("[STEP 1] Done.")


def preprocessingdata(config: PipelineConfig):
    print("\n[STEP 2] Normalising OHLC...")
    preprocessing.main()
    print("[STEP 2] Done.")


def generate_plots_wrapper(config: PipelineConfig):
    print("\n[STEP 3] Generating plots...")
    plots.main()
    print("[STEP 3] Done.")


def run_normtest(config: PipelineConfig):
    print("\n[STEP 4] Running normalisation test...")
    normtest_module.main()
    print("[STEP 4] Done.")


def run_pivot_detection(config: PipelineConfig):
    print("\n[STEP 5] Running pivot detection + feature extraction...")

    out_dir = os.path.dirname(config.features_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    features_df = feature_extraction.save_features(
        path=config.clean_ohlc_path,
        price_col=config.price_col,
        fluct=config.fluct,
        pivots_per_pattern=config.pivots_per_pattern,
        save_path=config.features_out_path,
    )

    print(f"[STEP 5] Detected {len(features_df)} pivot patterns.")
    print(f"[STEP 5] Saved pivot pattern features to: {config.features_out_path}")
    print("[STEP 5] Done.")


def run_dataset_prep(config: PipelineConfig):
    print("\n[STEP 6] Building PBFE training dataset (X_seq, X_feat, y, meta)...")

    os.makedirs(config.dataset_out_dir, exist_ok=True)

    ds_cfg = prepfeatureset.PBFEDatasetConfig(
        L=config.L,
        H=config.H,
        dead_zone=None,
        zero_target=config.zero_target,
    )

    X_seq, X_feat, y, meta = prepfeatureset.build_pbfe_dataset_from_files(
        pattern_csv_path=config.features_out_path,
        ohlc_norm_csv_path=config.norm_pct_path,
        ohlc_clean_csv_path=config.clean_ohlc_path,
        config=ds_cfg,
    )

    print(f"[STEP 6] Built dataset with {X_seq.shape[0]} samples.")
    print(f"[STEP 6] X_seq shape:  {X_seq.shape}  (N, L, D_seq)")
    print(f"[STEP 6] X_feat shape: {X_feat.shape} (N, D_feat)")

    full_path = os.path.join(config.dataset_out_dir, "pbfe_dataset_full.npz")
    np.savez_compressed(
        full_path,
        X_seq=X_seq,
        X_feat=X_feat,
        y=y,
    )
    meta.to_csv(
        os.path.join(config.dataset_out_dir, "pbfe_dataset_full_meta.csv"),
        index=False,
    )

    splits = prepfeatureset.split(X_seq, X_feat, y, meta)

    for split_name, split_data in splits.items():
        split_npz_path = os.path.join(
            config.dataset_out_dir, f"pbfe_{split_name}.npz"
        )
        split_meta_path = os.path.join(
            config.dataset_out_dir, f"pbfe_{split_name}_meta.csv"
        )

        np.savez_compressed(
            split_npz_path,
            X_seq=split_data["X_seq"],
            X_feat=split_data["X_feat"],
            y=split_data["y"],
        )
        split_data["meta"].to_csv(split_meta_path, index=False)

        print(
            f"[STEP 6] Saved {split_name} split: "
            f"{split_data['X_seq'].shape[0]} samples -> {split_npz_path}"
        )

    print("[STEP 6] Done.")


#! --- MAIN ---

def main(config: PipelineConfig = CONFIG):

    raw2processed(config)

    preprocessingdata(config)

    if config.run_plots:
        generate_plots_wrapper(config)
    else:
        print("\n[STEP 3] Skipped plots.")

    if config.run_normtest:
        run_normtest(config)
    else:
        print("\n[STEP 4] Skipped LSTM/GRU normtest.")

    if config.run_pivots_step:
        run_pivot_detection(config)
    else:
        print("\n[STEP 5] Skipped pivot detection.")

    if config.run_dataset_prep:
        run_dataset_prep(config)
    else:
        print("\n[STEP 6] Skipped dataset preparation.")

    if config.run_tuning_model_a:
        hyp.run_tuning(
            model_type="A",
            dataset_out_dir=config.dataset_out_dir,
            max_trials=100,
            executions_per_trial=2,
            max_epochs=50,
        )
    else:
        print("\n[STEP 7] Skipped Bayesian tuning for Model A.")

    if config.run_tuning_model_b:
        hyp.run_tuning(
            model_type="B",
            dataset_out_dir=config.dataset_out_dir,
            max_trials=100,
            executions_per_trial=2,
            max_epochs=50,
        )
    else:
        print("\n[STEP 7] Skipped Bayesian tuning for Model B.")

    if config.run_train_eval_models:
        print("\n[STEP 8] Training and evaluating tuned models...")
        train_eval.run_training_and_evaluation(
            dataset_dir=config.dataset_out_dir,
            train_models=True,
            max_epochs=100,
            batch_size=256,
        )
    else:
        print("\n[STEP 8] Skipped model training & evaluation.")

    print("\n=== Pipeline complete ===")

if __name__ == "__main__":
    main(CONFIG)

