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


def generate_poly_series_V2(
    T=10_000, coeffs=(0.5, 0.1, 0.02, 0.005), noise_std=0.1, seed=0
):
    """
    y_t = b1*x + b2*x^2 + b3*x^3 + b4*x^4 + eps
    x = t/(T-1) in [0,1]
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    x = t / (T - 1.0)
    b1, b2, b3, b4 = coeffs
    y = b1 * x + b2 * (x**2) + b3 * (x**3) + b4 * (x**4) + np.sin(x * 10 * np.pi) * 0.1
    y += rng.normal(0.0, noise_std, size=T).astype(np.float32)
    xy = np.stack((x, y), axis=1)
    return xy


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


import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, Any


class WindowDatasetV2(Dataset):
    """
    从时间序列构造 (X_past, Y_future) 窗口对。

    输入:
      series: np.ndarray 或 torch.Tensor
        - shape: (T,) 或 (T, C) 或更高维 (T, ...)
          约定第 0 维是时间维 T，其余维度视为特征维/通道维（原样保留）。
      n: 过去窗口长度
      m: 未来窗口长度
      stride: 窗口滑动步长
      start/end: 有效区间 [start, end)
      x_slice: 选择 X 的特征维切片/索引（作用在时间维之后的维度上）
      y_slice: 选择 Y 的特征维切片/索引
      return_meta: 是否额外返回 meta（包含起点 s 等）

    输出（单样本）:
      - 若 return_meta=False: (X, Y)
      - 若 return_meta=True : (X, Y, meta)
        X shape: (n, *Fx)
        Y shape: (m, *Fy)
    """

    def __init__(
        self,
        series: Union[np.ndarray, torch.Tensor],
        n: int,
        m: int,
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        x_slice: Optional[Any] = None,
        y_slice: Optional[Any] = None,
        return_meta: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        if n <= 0 or m <= 0:
            raise ValueError("n 和 m 必须为正整数。")
        if stride <= 0:
            raise ValueError("stride 必须为正整数。")

        # 统一成 torch.Tensor，且保证时间维在 dim=0
        if isinstance(series, np.ndarray):
            if series.ndim == 1:
                series = series[:, None]  # (T,) -> (T,1)
            self.series = torch.from_numpy(series.astype(np.float32))
        elif torch.is_tensor(series):
            if series.ndim == 1:
                series = series.unsqueeze(-1)  # (T,) -> (T,1)
            self.series = series
        else:
            raise TypeError("series 必须是 np.ndarray 或 torch.Tensor")

        self.series = self.series.to(dtype=dtype)

        T = self.series.shape[0]
        self.n = int(n)
        self.m = int(m)
        self.stride = int(stride)
        self.start = int(start)
        self.end = int(T if end is None else end)

        if not (0 <= self.start < T):
            raise ValueError(f"start 越界: start={self.start}, T={T}")
        if not (0 < self.end <= T):
            raise ValueError(f"end 越界: end={self.end}, T={T}")
        if self.start >= self.end:
            raise ValueError("start 必须小于 end。")

        self.x_slice = x_slice
        self.y_slice = y_slice
        self.return_meta = bool(return_meta)

        # 最后一个窗口起点 s 满足 s + n + m <= end
        self.max_s = self.end - (self.n + self.m)
        if self.max_s < self.start:
            raise ValueError("序列太短，无法构造一个窗口。")

        self.idxs = torch.arange(
            self.start, self.max_s + 1, self.stride, dtype=torch.long
        )

    def __len__(self) -> int:
        return int(self.idxs.numel())

    def _apply_feature_slice(
        self, X: torch.Tensor, feat_slice: Optional[Any]
    ) -> torch.Tensor:
        """
        对时间维之后的特征维做切片/索引。
        X: (time, *features)
        feat_slice: None / slice / int / list / tuple / np.ndarray / torch.Tensor (用于索引特征维)
        """
        if feat_slice is None:
            return X
        # 只对 dim>=1 部分做索引：X[:, feat_slice] 适用于 (T, C) 以及 (T, ...) 的常见场景
        return X[(slice(None),) + (feat_slice,)]

    def __getitem__(self, i: int):
        s = int(self.idxs[i].item())

        # 原始切窗：保持除时间维外的所有特征维
        X = self.series[s : s + self.n]  # (n, *F)
        Y = self.series[s + self.n : s + self.n + self.m]  # (m, *F)

        # 选择特征子集（可选）
        X = self._apply_feature_slice(X, self.x_slice)  # (n, *Fx)
        Y = self._apply_feature_slice(Y, self.y_slice)  # (m, *Fy)

        if self.return_meta:
            meta = {
                "start": s,
                "x_range": (s, s + self.n),
                "y_range": (s + self.n, s + self.n + self.m),
            }
            return X, Y, meta
        return X, Y


def collate_fixed(batch):
    # batch: [(X,Y), ...] 或 [(X,Y,meta), ...]
    if len(batch[0]) == 2:
        Xs, Ys = zip(*batch)
        metas = None
    else:
        Xs, Ys, metas = zip(*batch)

    X = torch.stack(Xs, 0)
    Y = torch.stack(Ys, 0)
    X_len = torch.full((len(batch),), X.size(1), dtype=torch.long)
    Y_len = torch.full((len(batch),), Y.size(1), dtype=torch.long)

    if metas is None:
        return X, X_len, Y, Y_len
    return X, X_len, Y, Y_len, metas


def split_by_time(series, train_ratio=0.7, val_ratio=0.15):
    T = len(series)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return series[:t1], series[t1:t2], series[t2:]
