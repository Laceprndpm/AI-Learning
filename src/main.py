from dataset import *
from model import *
from train import *


def build_loaders(y, n, m, batch_size=128, stride=1, split=0.8):
    T = len(y)
    cut = int(T * split)
    train_ds = WindowDataset(y, n=n, m=m, stride=stride, start=0, end=cut)
    val_ds = WindowDataset(y, n=n, m=m, stride=stride, start=cut, end=T)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


# 生成数据
x = np.arange(10_000, dtype=np.float32)
y = generate_poly_series(T=10_000, noise_std=0.1, seed=0)

plt.figure(figsize=(10, 4))
plt.plot(x, y, label="y")
plt.show()

n, m = 64, 16
train_loader, val_loader = build_loaders(y, n, m, batch_size=256, stride=1, split=0.8)

# 你自己设定模型超参，注意维度一致：
# encoder input_size=1
# decoder input_size=2（因为有 bos_flag）
H = 128
num_heads = 8
num_layers = 2
dropout = 0.1

encoder = TransformerEncoder(
    input_size=2,
    key_size=H,
    query_size=H,
    value_size=H,
    num_hiddens=H,
    norm_shape=H,
    ffn_num_input=H,
    ffn_num_hiddens=4 * H,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
)

decoder = TransformerDecoder(
    input_size=3,  # 关键：value+bos_flag
    key_size=H,
    query_size=H,
    value_size=H,
    num_hiddens=H,
    norm_shape=H,
    ffn_num_input=H,
    ffn_num_hiddens=4 * H,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
)

net = EncoderDecoder(encoder, decoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_seq2seq_regression(
    net, train_loader, val_loader, num_epochs=10, lr=1e-3, device=device
)
