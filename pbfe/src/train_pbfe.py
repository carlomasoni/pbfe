
#! --- INFO ---

#! --- IMPORTS ---

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from pbfe.src.datasets import make_dataloaders
from pbfe.src.models import ModelA, ModelB



#! --- CONFIG ---

#! --- FUNCTIONS ---

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_seq, X_feat, y in loader:
        X_seq = X_seq.to(device)
        X_feat = X_feat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X_seq, X_feat)  # for ModelA we’ll ignore X_feat via wrapper
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_seq, X_feat, y in loader:
        X_seq = X_seq.to(device)
        X_feat = X_feat.to(device)
        y = y.to(device)

        logits = model(X_seq, X_feat)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def train_model(
    model_type: str = "A",  
    train_npz: str = "pbfe/data/datasets/pbfe_train.npz",
    val_npz: str = "pbfe/data/datasets/pbfe_val.npz",
    test_npz: str = "pbfe/data/datasets/pbfe_test.npz",
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_npz, val_npz, test_npz, batch_size=batch_size
    )

    X_seq, X_feat, y = next(iter(train_loader))
    L = X_seq.shape[1]
    d_seq = X_seq.shape[2]
    d_feat = X_feat.shape[1]

    if model_type == "A":
        model = ModelA(d_seq=d_seq)
        orig_forward = model.forward

        def forward_wrapper(x_seq, x_feat):
            return orig_forward(x_seq)

        model.forward = forward_wrapper
    elif model_type == "B":
        model = ModelB(d_seq=d_seq, d_feat=d_feat)
    else:
        raise ValueError("model_type must be 'A' or 'B'")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nBest val_loss={best_val_loss:.4f}")
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.3f}")

    return model