import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import generate_poly_series, WindowDataset, split_by_time
import matplotlib.pyplot as plt

# train_poly_transformer.py
# Minimal encoder-only Transformer for 1-step time series regression on synthetic polynomial + noise
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
#
# -------------------------


class MaskedMSELoss(nn.Module):
    def forward(self, y_hat, y, valid_len=None):
        # y_hat, y: (B, m, 1) 或 (B, m, d)
        if valid_len is None:
            return ((y_hat - y) ** 2).mean()

        B, m = y.shape[0], y.shape[1]
        mask = (
            torch.arange(m, device=y.device)[None, :] < valid_len[:, None]
        ).unsqueeze(-1)
        se = (y_hat - y) ** 2
        se = se * mask
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (se.sum(dim=(1, 2)) / denom).mean()


def make_dec_input_with_bos_flag(Y, device=None):
    """
    Y: (B, m, 1) float
    return dec_in: (B, m, 2) float
    """
    if device is None:
        device = Y.device
    B, m, _ = Y.shape
    # value 通道：第 0 步用 0，占位；后续右移一位 teacher forcing
    dec_val0 = torch.zeros((B, 1, 1), device=device, dtype=Y.dtype)
    dec_val = torch.cat([dec_val0, Y[:, :-1, :]], dim=1)  # (B,m,1)

    # flag 通道：第 0 步为 1，其余为 0
    dec_flag = torch.zeros((B, m, 1), device=device, dtype=Y.dtype)
    dec_flag[:, 0, 0] = 1.0

    return torch.cat([dec_val, dec_flag], dim=-1)  # (B,m,2)


def make_dec_input_with_bos_flag_V2(Y: torch.Tensor, device=None) -> torch.Tensor:
    """
    通用版（仅考虑 Y 为 (B,m) 或 (B,m,d)):
    输入:
      Y:
        - (B, m)       或
        - (B, m, d)
    输出:
      dec_in:
        - 若 Y 是 (B,m)     -> (B, m, 2)   (右移值 1 维 + bos 1 维)
        - 若 Y 是 (B,m,d)   -> (B, m, d+1) (右移值 d 维 + bos 1 维)
    规则:
      - teacher forcing 右移:t=0 用 0 占位,t>0 用 Y[:, t-1]
      - bos flag: t=0 为 1, 其余为 0, 作为最后一维额外通道拼接
    """
    if device is None:
        device = Y.device
    if Y.dim() not in (2, 3):
        raise ValueError(f"只支持 Y 为 (B,m) 或 (B,m,d)，但收到 shape={tuple(Y.shape)}")

    # (B,m) -> (B,m,1)
    if Y.dim() == 2:
        Y = Y.unsqueeze(-1)  # (B,m,1)

    B, m, d = Y.shape

    # shift
    dec_val0 = torch.zeros((B, 1, d), device=device, dtype=Y.dtype)
    dec_val = torch.cat([dec_val0, Y[:, :-1, :]], dim=1)  # (B,m,d)
    # cat bos flag
    dec_flag = torch.zeros((B, m, 1), device=device, dtype=Y.dtype)
    dec_flag[:, 0, 0] = 1.0
    dec_in = torch.cat([dec_val, dec_flag], dim=-1)  # (B,m,d+1)
    return dec_in


def make_dec_input_with_bos_flag_V3(
    X: torch.Tensor, Y: torch.Tensor, device=None
) -> torch.Tensor:
    """
    update: dec_val0 is the last time step of X instead of zeros,
        to provide a smoother transition from encoder to decoder inputs.
    通用版（仅考虑 Y 为 (B,m) 或 (B,m,d)):
    输入:
      Y:
        - (B, m)       或
        - (B, m, d)
    输出:
      dec_in:
        - 若 Y 是 (B,m)     -> (B, m, 2)   (右移值 1 维 + bos 1 维)
        - 若 Y 是 (B,m,d)   -> (B, m, d+1) (右移值 d 维 + bos 1 维)
    规则:
      - teacher forcing 右移:t=0 用 0 占位,t>0 用 Y[:, t-1]
      - bos flag: t=0 为 1, 其余为 0, 作为最后一维额外通道拼接
    """
    if device is None:
        device = Y.device
    if Y.dim() not in (2, 3):
        raise ValueError(f"只支持 Y 为 (B,m) 或 (B,m,d)，但收到 shape={tuple(Y.shape)}")

    # (B,m) -> (B,m,1)

    if X.dim() == 2:
        X = X.unsqueeze(-1)  # (B,n,1)
    if Y.dim() == 2:
        Y = Y.unsqueeze(-1)  # (B,m,1)
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(
            f"期望 X,Y 为 (B,T,d)；但得到 X{tuple(X.shape)} Y{tuple(Y.shape)}"
        )

    B, m, d = Y.shape
    if X.shape[0] != B:
        raise ValueError(f"batch 不一致: X{tuple(X.shape)} Y{tuple(Y.shape)}")
    if X.shape[2] != d:
        raise ValueError(f"特征维不一致: X_last_dim={X.shape[2]} vs Y_last_dim={d}")

    B, m, d = Y.shape

    # shift
    dec_val0 = X[:, -1:, :]  # use the last time step of X instead of zeros
    dec_val = torch.cat([dec_val0, Y[:, :-1, :]], dim=1)  # (B,m,d)
    # cat bos flag
    dec_flag = torch.zeros((B, m, 1), device=device, dtype=Y.dtype)
    dec_flag[:, 0, 0] = 1.0
    dec_in = torch.cat([dec_val, dec_flag], dim=-1)  # (B,m,d+1)
    return dec_in


def train_seq2seq_regression(
    net,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    device,
    grad_clip=1.0,
):
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedMSELoss()

    for epoch in range(1, num_epochs + 1):
        net.train()
        tr_loss_sum, tr_count = 0.0, 0

        for X, X_len, Y, Y_len in train_loader:
            X = X.to(device)  # (B,n,2)
            X_len = X_len.to(device)  # (B,) or scalar per sample
            Y = Y.to(device)  # (B,m,2)
            Y_len = Y_len.to(device)

            dec_in = make_dec_input_with_bos_flag_V3(X, Y)  # (B,m,3)

            opt.zero_grad()
            out = net(X, dec_in, X_len)  # (B,m,1) if decoder input_size=3
            y_hat = out

            # print("y_hat", y_hat.shape)
            # print("Y    ", Y.shape)
            # print("Y[:,:,1]", Y[:, :, 1:2].shape)

            loss = loss_fn(y_hat, Y[:, :, 1:2], Y_len)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            tr_loss_sum += float(loss.detach()) * X.shape[0]
            tr_count += X.shape[0]

        tr_loss = tr_loss_sum / max(tr_count, 1)

        # val
        net.eval()
        va_loss_sum, va_count = 0.0, 0
        with torch.no_grad():
            for X, X_len, Y, Y_len in val_loader:
                X = X.to(device)
                X_len = X_len.to(device)
                Y = Y.to(device)
                Y_len = Y_len.to(device)

                dec_in = make_dec_input_with_bos_flag_V3(X, Y)
                out = net(X, dec_in, X_len)
                y_hat = out
                # print("y_hat", y_hat.shape)
                # print("Y    ", Y.shape)
                # print("Y[:,:,1]", Y[:, :, 1:2].shape)

                loss = loss_fn(y_hat, Y[:, :, 1:2], Y_len)

                va_loss_sum += float(loss) * X.shape[0]
                va_count += X.shape[0]
        va_loss = va_loss_sum / max(va_count, 1)

        print(f"epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}")
