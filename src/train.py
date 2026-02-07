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
            X = X.to(device)  # (B,n,1)
            X_len = X_len.to(device)  # (B,) or scalar per sample
            Y = Y.to(device)  # (B,m,1)
            Y_len = Y_len.to(device)

            dec_in = make_dec_input_with_bos_flag(Y)  # (B,m,2)

            opt.zero_grad()
            out = net(X, dec_in, X_len)  # (B,m,2) if decoder input_size=2
            y_hat = out

            loss = loss_fn(y_hat, Y, Y_len)
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

                dec_in = make_dec_input_with_bos_flag(Y)
                out = net(X, dec_in, X_len)
                y_hat = out[..., :1]
                loss = loss_fn(y_hat, Y, Y_len)

                va_loss_sum += float(loss) * X.shape[0]
                va_count += X.shape[0]
        va_loss = va_loss_sum / max(va_count, 1)

        print(f"epoch {epoch:3d} | train {tr_loss:.6f} | val {va_loss:.6f}")
