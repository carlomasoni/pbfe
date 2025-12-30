
#! --- INFO ---

#! --- IMPORTS ---

import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers # pyright: ignore[reportMissingImports]

#! --- CONFIG ---


DATA_DIR = "pbfe/data/datasets"
TRAIN_PATH = os.path.join(DATA_DIR, "pbfe_train.npz")
VAL_PATH = os.path.join(DATA_DIR, "pbfe_val.npz")
TEST_PATH = os.path.join(DATA_DIR, "pbfe_test.npz")

NUM_CLASSES = 3

#! --- FUNCTIONS ---

def load_split(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X_seq = data["X_seq"].astype("float32")
    X_feat = data["X_feat"].astype("float32")
    y = data["y"].astype("int64") + 1
    return X_seq, X_feat, y


def build_model_a(hp: kt.HyperParameters, input_seq_shape):
    inputs = keras.Input(shape=input_seq_shape, name="seq")
    x = inputs

    num_layers = hp.Int("num_lstm_layers", 1, 10)
    lstm_units = hp.Int("lstm_units", 1, 20)
    lstm_dropout = hp.Float("lstm_dropout", 0.1, 0.5, step=0.1)

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        x = layers.LSTM(lstm_units, return_sequences=return_sequences)(x)
        x = layers.Dropout(lstm_dropout)(x)

    dense_units = hp.Int("dense_units", 10, 100, step=10)
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.5, step=0.1)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dense_dropout)(x)

    outputs = layers.Dense(NUM_CLASSES)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelA_seq_only")

    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model_b(hp: kt.HyperParameters, input_seq_shape, input_feat_shape):
    seq_in = keras.Input(shape=input_seq_shape, name="seq")
    feat_in = keras.Input(shape=input_feat_shape, name="feat")

    x = seq_in
    num_layers = hp.Int("num_lstm_layers", 1, 10)
    lstm_units = hp.Int("lstm_units", 1, 20)
    lstm_dropout = hp.Float("lstm_dropout", 0.1, 0.5, step=0.1)

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        x = layers.LSTM(lstm_units, return_sequences=return_sequences)(x)
        x = layers.Dropout(lstm_dropout)(x)

    feat_hidden = 32
    f = layers.Dense(feat_hidden, activation="relu")(feat_in)
    f = layers.Dropout(0.3)(f)

    combined = layers.concatenate([x, f])

    dense_units = hp.Int("dense_units", 10, 100, step=10)
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.5, step=0.1)

    h = layers.Dense(dense_units, activation="relu")(combined)
    h = layers.Dropout(dense_dropout)(h)

    outputs = layers.Dense(NUM_CLASSES)(h)

    model = keras.Model(inputs=[seq_in, feat_in], outputs=outputs, name="ModelB_pbfe")

    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def run_tuning(
    model_type: str,
    dataset_out_dir: str,
    max_trials: int = 100,
    executions_per_trial: int = 2,
    max_epochs: int = 50,
):
    train_path = os.path.join(dataset_out_dir, "pbfe_train.npz")
    val_path = os.path.join(dataset_out_dir, "pbfe_val.npz")
    test_path = os.path.join(dataset_out_dir, "pbfe_test.npz")

    X_seq_train, X_feat_train, y_train = load_split(train_path)
    X_seq_val, X_feat_val, y_val = load_split(val_path)
    X_seq_test, X_feat_test, y_test = load_split(test_path)

    input_seq_shape = X_seq_train.shape[1:]
    input_feat_shape = X_feat_train.shape[1:]

    if model_type == "A":
        def model_builder(hp):
            return build_model_a(hp, input_seq_shape)
        x_train = X_seq_train
        x_val = X_seq_val
        x_test = X_seq_test
        project_name = "pbfe_tuning_A"
    elif model_type == "B":
        def model_builder(hp):
            return build_model_b(hp, input_seq_shape, input_feat_shape)
        x_train = [X_seq_train, X_feat_train]
        x_val = [X_seq_val, X_feat_val]
        x_test = [X_seq_test, X_feat_test]
        project_name = "pbfe_tuning_B"
    else:
        raise ValueError("model_type must be 'A' or 'B'")

    tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_accuracy",
        max_trials=max_trials,
        num_initial_points=10,
        executions_per_trial=executions_per_trial,
        directory="keras_tuner",
        project_name=project_name,
        overwrite=False,
    )

    stop_early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print(f"\n[STEP 7] Starting Bayesian tuning for Model {model_type}...")
    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        callbacks=[stop_early],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[STEP 7] Best hyperparameters for Model {model_type}: {best_hp.values}")

    model = tuner.hypermodel.build(best_hp)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        callbacks=[stop_early],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[STEP 7] Model {model_type} test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

    weights_path = os.path.join(dataset_out_dir, f"pbfe_model_{model_type}_best.h5")
    model.save(weights_path)
    print(f"[STEP 7] Saved best Model {model_type} to: {weights_path}")

    return model, best_hp, history