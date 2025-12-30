import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.models import load_model

from pbfe.src import hyp

DEFAULT_DATASET_DIR = "pbfe/data/datasets"
CLASS_NAMES = ["Short", "Flat", "Long"]


def load_pbfe_splits(dataset_dir: str) -> Tuple[Dict, Dict, Dict]:
    train_path = os.path.join(dataset_dir, "pbfe_train.npz")
    val_path = os.path.join(dataset_dir, "pbfe_val.npz")
    test_path = os.path.join(dataset_dir, "pbfe_test.npz")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset split: {p}")

    X_seq_tr, X_feat_tr, y_tr = hyp.load_split(train_path)
    X_seq_va, X_feat_va, y_va = hyp.load_split(val_path)
    X_seq_te, X_feat_te, y_te = hyp.load_split(test_path)

    train = {"X_seq": X_seq_tr, "X_feat": X_feat_tr, "y": y_tr}
    val = {"X_seq": X_seq_va, "X_feat": X_feat_va, "y": y_va}
    test = {"X_seq": X_seq_te, "X_feat": X_feat_te, "y": y_te}

    return train, val, test


def _plot_learning_curves(history, out_path: str) -> None:
    if isinstance(history, dict):
        hist = history
    else:
        hist = history.history

    plt.figure(figsize=(10, 4))

    if "loss" in hist:
        plt.subplot(1, 2, 1)
        plt.plot(hist["loss"], label="train loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

    acc_key = None
    for key in ["accuracy", "sparse_categorical_accuracy"]:
        if key in hist:
            acc_key = key
            break

    if acc_key is not None:
        val_acc_key = "val_" + acc_key
        plt.subplot(1, 2, 2)
        plt.plot(hist[acc_key], label="train acc")
        if val_acc_key in hist:
            plt.plot(hist[val_acc_key], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _manual_train_loop(
    model: tf.keras.Model,
    train_split: Dict,
    val_split: Dict,
    model_type: str,
    max_epochs: int,
    batch_size: int,
    patience: int = 10,
) -> Dict[str, list]:
    assert model_type in ("A", "B")

    X_tr_seq = train_split["X_seq"]
    X_tr_feat = train_split["X_feat"]
    y_tr = train_split["y"]

    X_val_seq = val_split["X_seq"]
    X_val_feat = val_split["X_feat"]
    y_val = val_split["y"]

    n_train = y_tr.shape[0]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    base_opt = getattr(model, "optimizer", None)
    lr = 1e-3
    if base_opt is not None:
        try:
            lr_val = base_opt.learning_rate
            if hasattr(lr_val, "numpy"):
                lr = float(lr_val.numpy())
            else:
                lr = float(lr_val)
        except Exception:
            pass

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = np.inf
    best_weights = model.get_weights()
    wait = 0

    for epoch in range(max_epochs):
        idx = np.random.permutation(n_train)
        X_tr_seq_epoch = X_tr_seq[idx]
        X_tr_feat_epoch = X_tr_feat[idx]
        y_tr_epoch = y_tr[idx]

        epoch_loss = 0.0
        epoch_correct = 0
        total_seen = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            x_seq_batch = tf.convert_to_tensor(X_tr_seq_epoch[start:end])
            y_batch = tf.convert_to_tensor(y_tr_epoch[start:end], dtype=tf.int32)

            if model_type == "B":
                x_feat_batch = tf.convert_to_tensor(X_tr_feat_epoch[start:end])

            with tf.GradientTape() as tape:
                if model_type == "A":
                    logits = model(x_seq_batch, training=True)
                else:
                    logits = model([x_seq_batch, x_feat_batch], training=True)
                loss_value = loss_fn(y_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            batch_size_eff = end - start
            epoch_loss += float(loss_value) * batch_size_eff
            total_seen += batch_size_eff

            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            epoch_correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, y_batch), tf.int32)))

        train_loss = epoch_loss / max(total_seen, 1)
        train_acc = epoch_correct / max(total_seen, 1)

        x_val_seq_tf = tf.convert_to_tensor(X_val_seq)
        y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.int32)
        if model_type == "A":
            val_logits = model(x_val_seq_tf, training=False)
        else:
            x_val_feat_tf = tf.convert_to_tensor(X_val_feat)
            val_logits = model([x_val_seq_tf, x_val_feat_tf], training=False)

        val_loss_value = loss_fn(y_val_tf, val_logits)
        val_preds = tf.argmax(val_logits, axis=-1, output_type=tf.int32)
        val_correct = int(tf.reduce_sum(tf.cast(tf.equal(val_preds, y_val_tf), tf.int32)))
        val_loss = float(val_loss_value)
        val_acc = val_correct / max(y_val.shape[0], 1)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{max_epochs} "
            f"- loss: {train_loss:.4f} - acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered (no val_loss improvement for {patience} epochs).")
                break

    model.set_weights(best_weights)

    return history


def train_model_from_best(
    model_type: str,
    dataset_dir: str,
    max_epochs: int = 100,
    batch_size: int = 256,
) -> str:
    assert model_type in ("A", "B")

    model_best_path = os.path.join(dataset_dir, f"pbfe_model_{model_type}_best.h5")
    if not os.path.exists(model_best_path):
        raise FileNotFoundError(
            f"Best model file not found for Model {model_type}: {model_best_path}"
        )

    print(f"\n[STEP 8] Loading tuned model {model_type} from: {model_best_path}")
    model = load_model(model_best_path)

    train, val, _ = load_pbfe_splits(dataset_dir)

    print(f"[STEP 8] Training Model {model_type} with manual loop...")
    history = _manual_train_loop(
        model=model,
        train_split=train,
        val_split=val,
        model_type=model_type,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=10,
    )

    curves_path = os.path.join(dataset_dir, f"training_curves_{model_type}.png")
    _plot_learning_curves(history, curves_path)
    print(f"[STEP 8] Saved learning curves for Model {model_type} to: {curves_path}")

    final_path = os.path.join(dataset_dir, f"pbfe_model_{model_type}_final.h5")
    model.save(final_path)
    print(f"[STEP 8] Saved final Model {model_type} to: {final_path}")

    return final_path


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    out_path: str,
    normalize: bool = True,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype("float")
        row_sums[row_sums == 0.0] = 1.0
        cm = cm.astype("float") / row_sums

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_probability_histograms(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str,
    class_names=None,
) -> None:
    num_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    plt.figure(figsize=(6, 6))

    for c in range(num_classes):
        plt.subplot(num_classes, 1, c + 1)
        p_c = y_proba[y_true == c, c]
        if p_c.size == 0:
            plt.text(0.5, 0.5, "No samples", ha="center", va="center")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        else:
            plt.hist(p_c, bins=20, density=True)
            plt.xlim(0, 1)
        plt.ylabel("Density")
        plt.title(f"P(model predicts {class_names[c]}) | true = {class_names[c]}")

    plt.xlabel("Predicted probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_model(
    model_path: str,
    model_type: str,
    test_split: Dict,
    dataset_dir: str,
) -> Tuple[float, float]:
    assert model_type in ("A", "B")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n[STEP 8] Evaluating Model {model_type} on test set...")
    model = load_model(model_path, compile=False)

    X_seq, X_feat, y_true = test_split["X_seq"], test_split["X_feat"], test_split["y"]

    if model_type == "A":
        logits = model.predict(X_seq, verbose=0)
    else:
        logits = model.predict([X_seq, X_feat], verbose=0)

    y_proba = tf.nn.softmax(logits, axis=-1).numpy()
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"[STEP 8] Test accuracy (Model {model_type}): {acc:.4f}")
    print(f"[STEP 8] Test macro F1 (Model {model_type}): {macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[STEP 8] Confusion matrix (Model {model_type}):")
    print(cm)

    print(f"\n[STEP 8] Classification report (Model {model_type}):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    counts = np.bincount(y_true, minlength=len(CLASS_NAMES))
    print(f"[STEP 8] Class distribution in test set: {counts} (labels 0,1,2)")

    cm_path = os.path.join(dataset_dir, f"confusion_matrix_{model_type}.png")
    _plot_confusion_matrix(cm, CLASS_NAMES, cm_path, normalize=True)
    print(f"[STEP 8] Saved confusion matrix for Model {model_type} to: {cm_path}")

    prob_hist_path = os.path.join(dataset_dir, f"prob_hist_{model_type}.png")
    _plot_probability_histograms(
        y_true,
        y_proba,
        prob_hist_path,
        class_names=CLASS_NAMES,
    )
    print(f"[STEP 8] Saved probability histograms for Model {model_type} to: {prob_hist_path}")

    return acc, macro_f1


def run_training_and_evaluation(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    train_models: bool = True,
    max_epochs: int = 100,
    batch_size: int = 256,
) -> None:
    os.makedirs(dataset_dir, exist_ok=True)

    model_A_path = os.path.join(dataset_dir, "pbfe_model_A_final.h5")
    model_B_path = os.path.join(dataset_dir, "pbfe_model_B_final.h5")

    if train_models:
        model_A_path = train_model_from_best(
            model_type="A",
            dataset_dir=dataset_dir,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        model_B_path = train_model_from_best(
            model_type="B",
            dataset_dir=dataset_dir,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
    else:
        best_A = os.path.join(dataset_dir, "pbfe_model_A_best.h5")
        best_B = os.path.join(dataset_dir, "pbfe_model_B_best.h5")

        if not os.path.exists(model_A_path):
            model_A_path = best_A
        if not os.path.exists(model_B_path):
            model_B_path = best_B

    _, _, test = load_pbfe_splits(dataset_dir)
    acc_A, f1_A = evaluate_model(model_A_path, "A", test, dataset_dir)
    acc_B, f1_B = evaluate_model(model_B_path, "B", test, dataset_dir)

    print("\n[STEP 8] === Model Comparison on Test Set ===")
    print(f"Model A (seq only):    acc = {acc_A:.4f}, macro F1 = {f1_A:.4f}")
    print(f"Model B (seq + PBFE):  acc = {acc_B:.4f}, macro F1 = {f1_B:.4f}")

    if f1_B > f1_A:
        print(
            "\n[STEP 8] Interpretation:\n"
            "Model B outperforms Model A, which is a good sign that the PBFE features are adding value."
        )
    elif f1_B < f1_A:
        print(
            "\n[STEP 8] Interpretation:\n"
            "Model A outperforms Model B, which is a good sign that the PBFE features are not adding value."
        )
    else:
        print(
            "\n[STEP 8] Interpretation:\n"
            "- Both models achieve very similar macro F1.\n"
            "- PBFE features may be largely neutral (neither helping nor hurting), "
            "or they help specific classes while hurting others.\n"
            "- Inspect per-class F1 in the classification reports and confusion matrices "
            "to see which regimes benefit from PBFE."
        )


if __name__ == "__main__":
    run_training_and_evaluation(DEFAULT_DATASET_DIR, train_models=True)









