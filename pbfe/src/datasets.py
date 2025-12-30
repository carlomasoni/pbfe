
#! --- INFO ---

"""


"""
#! --- IMPORTS ---

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


#! --- CLASSES ---

class PBFEDataset(Dataset):
    def __init__(self, npz_path:str):
        data=np.load(npz_path, allow_pickle=True)
        X_seq = data['X_seq']
        X_feat = data['X_feat']
        y = data['y']

        y = y.astype(np.int64) + 1

        self.X_seq = torch.from_numpy(X_seq).float()
        self.X_feat = torch.from_numpy(X_feat).float()
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]


#! --- TRAINING ---

def make_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
):
    train_ds = PBFEDataset(train_path)
    val_ds = PBFEDataset(val_path)
    test_ds = PBFEDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


