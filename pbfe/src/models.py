
#! --- INFO ---

"""


"""
#! --- IMPORTS ---

import numpy as np
import torch.nn as nn
import torch
#! --- CLASSES ---

class ModelA(nn.Module):
    def __init__(
        self,
        d_seq: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_seq, x_feat=None):
        # x_seq: (B, L, D_seq)
        out, (h_n, c_n) = self.lstm(x_seq)
        h_last = h_n[-1]            # (B, hidden_size)
        logits = self.head(h_last)  # (B, num_classes)
        return logits


class ModelB(nn.Module):
    def __init__(
        self,
        d_seq: int = 4,
        d_feat: int = 11,
        hidden_size: int = 32,
        num_layers: int = 1,
        feat_hidden: int = 32,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(d_feat, feat_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size + feat_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )


    def forward(self, x_seq, x_feat):
        out, (h_n, c_n) = self.lstm(x_seq)
        h_last = h_n[-1]                    
        feat_emb = self.feat_mlp(x_feat)   
        combined = torch.cat([h_last, feat_emb], dim=-1)
        logits = self.head(combined)
        return logits
