"""Нарезка временного ряда на окна"""
import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 30


class WindowDataset(Dataset):
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, i):
        x = self.X[i:i + self.seq_len]
        y = self.y[i + self.seq_len - 1]
        return x, y


def make_windows(X, y, seq_len=SEQ_LEN):
    X = np.asarray(X)
    Xw = np.stack([X[i:i + seq_len].ravel() for i in range(len(X) - seq_len + 1)])
    return Xw, align(y, seq_len)


def align(values, seq_len=SEQ_LEN):
    return np.asarray(values)[seq_len - 1:]
