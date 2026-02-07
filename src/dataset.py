import numpy as np
import torch
from torch.utils.data import Dataset


# -------------------------
# 1) Synthetic data
# -------------------------
def generate_poly_series(
    T=100_000, coeffs=(0.5, 0.1, 0.02, 0.005), noise_std=0.1, seed=0
):
    """
    y_t = b1*x + b2*x^2 + b3*x^3 + b4*x^4 + eps
    x = t/(T-1) in [0,1]
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    x = t / (T - 1.0)
    b1, b2, b3, b4 = coeffs
    y = b1 * x + b2 * (x**2) + b3 * (x**3) + b4 * (x**4)
    y += rng.normal(0.0, noise_std, size=T).astype(np.float32)
    return y


# -------------------------
# 2) Dataset (sliding window)
# -------------------------
class WindowDataset(Dataset):
    """
    从一维序列 y[0..T-1] 构造 (X_past, Y_future)
    X: (n, 1), Y: (m, 1)
    """

    def __init__(
        self,
        y: np.ndarray,
        n: int,
        m: int,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
    ):
        assert y.ndim == 1
        self.y = y.astype(np.float32)
        self.n = n
        self.m = m
        self.stride = stride
        self.start = start
        self.end = len(y) if end is None else end

        # 最后一个窗口起点 s 满足 s+n+m <= end
        self.max_s = self.end - (n + m)
        if self.max_s < self.start:
            raise ValueError("序列太短，无法构造一个窗口。")

        self.idxs = np.arange(self.start, self.max_s + 1, stride, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = int(self.idxs[i])
        x = self.y[s : s + self.n]  # (n,)
        y = self.y[s + self.n : s + self.n + self.m]  # (m,)

        X = torch.from_numpy(x).unsqueeze(-1)  # (n,1)
        Y = torch.from_numpy(y).unsqueeze(-1)  # (m,1)

        X_valid_len = torch.tensor(self.n, dtype=torch.long)
        Y_valid_len = torch.tensor(self.m, dtype=torch.long)
        return X, X_valid_len, Y, Y_valid_len


def split_by_time(series, train_ratio=0.7, val_ratio=0.15):
    T = len(series)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return series[:t1], series[t1:t2], series[t2:]
