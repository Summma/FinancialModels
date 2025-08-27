from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


def make_data(
    data_file: str, foresight: int = 30, context_length: int = 30, stock: str = "NVDA"
):
    df = pd.read_csv(data_file, header=[0, 1])

    n = df.shape[0] - foresight - context_length

    data = []
    total = 0
    for i in range(n):
        X = df[i : i + context_length]["Close"].to_numpy()

        y_now = df[i + foresight + context_length : i + foresight + context_length + 1][
            "Close"
        ][stock].to_numpy()
        y_past = df[i + context_length : i + context_length + 1]["Close"][
            stock
        ].to_numpy()
        y = y_now / y_past

        data.append([X, y, i])
        total += y

    return data, total


class FinanceData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y, i = self.data[idx]

        return X, y, i


class TokenPool(nn.Module):
    """
    Multi-head *sequence-to-one* attention pool.
    h_t --(heads & softmax)--> weights --> weighted-sum --> pooled vector
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        # linear maps to project keys & values *per head*
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        # one learned query vector per head (shape: [n_heads, 1, d_head])
        self.query = nn.Parameter(torch.randn(n_heads, 1, self.d_head))

    def forward(self, h):  # h: [B, T, d_model]
        B, T, _ = h.shape

        # ---------- make K & V ------------------------------------------------
        k = self.key(h)  # [B, T, d_model]
        v = self.value(h)

        # split heads: [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ---------- broadcast the query --------------------------------------
        # q: [B, n_heads, 1, d_head]   (same query for every example in batch)
        q = self.query.unsqueeze(0).repeat(B, 1, 1, 1)

        # ---------- scaled dot-product ---------------------------------------
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, h, 1, T]
        scores = scores / (self.d_head**0.5)
        weights = F.softmax(scores, dim=-1)  # attention

        # ---------- weighted sum to get context ------------------------------
        ctx = torch.matmul(weights, v)  # [B, h, 1, d]
        ctx = ctx.squeeze(2).reshape(B, -1)  # concat heads

        # (optional) return weights so you can plot them
        return ctx, weights.squeeze(2)


class LSTM(nn.Module):
    def __init__(
        self,
        n_stocks: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
    ):
        super().__init__()  # type: ignore
        self.lstm = nn.LSTM(
            input_size=n_stocks,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.attn_W = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.attn_v = nn.Parameter(torch.randn(hidden_size))

        self.token = TokenPool(hidden_size, n_heads)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.h = None
        self.hidden_size = hidden_size
        self.num_layers = n_layers

    def reset_hidden_state(self, batch_size: int, device: torch.device):
        """Resets the hidden state to zeros."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )

    def forward(self, x):
        if self.h is None or self.h[0].size(1) != x.size(0):
            self.h = self.reset_hidden_state(batch_size=x.size(0), device=x.device)
        else:
            self.h = (self.h[0].detach(), self.h[1].detach())

        out, self.h = self.lstm(x, self.h)

        context, _w = self.token(out)

        return self.head(context)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            y_hat = model(X)

            losses.append(loss_fn(y_hat, y).item())

            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.cpu().numpy())

    # direction‐accuracy
    dir_acc = ((np.array(y_true) > 1) == (np.array(y_pred) > 1)).mean()
    return np.mean(losses), float(dir_acc), y_true, y_pred


def train(model, loader, loss_fn, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        losses = []
        for i, (X, y, idx) in enumerate(train_loader):
            X = X.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            l = loss_fn(y_hat, y)
            losses.append(l.item())

            l.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {np.mean(losses)}")


if __name__ == "__main__":
    tscv = TimeSeriesSplit(n_splits=5)
    device = torch.device("mps")
    model = LSTM(17, 1024, 2, 0.1, 8).to(device)

    data, total = make_data("finance_data.csv", 30, 10, stock="MSFT")

    test_size = int(0.3 * len(data))
    train_val = data[:-test_size]
    final_test = data[-test_size:]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val)):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        train_loader = DataLoader(FinanceData(train_data), batch_size=32, shuffle=False)
        val_loader = DataLoader(FinanceData(val_data), batch_size=32, shuffle=False)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        loss = nn.HuberLoss(delta=1.0)

        train(model, train_loader, loss, optimizer, device, epochs=20)

        losses, dir_acc, _, _ = evaluate(model, val_loader, loss, device)
        print(f"Test Loss on fold {fold + 1}: {np.mean(losses)}")
        print(f"Test Direction Accuracy on fold {fold + 1}: {dir_acc}")

    final_val = DataLoader(FinanceData(train_val), batch_size=32, shuffle=False)
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    train(model, train_loader, loss, optimizer, device, epochs=20)

    test = DataLoader(FinanceData(final_test), batch_size=32, shuffle=False)
    losses, dir_acc, y_true, y_pred = evaluate(model, test, loss, device)
    print(f"Test Loss on final validation set: {np.mean(losses)}")
    print(f"Test Direction Accuracy on final validation set: {dir_acc}")

    torch.save(model.state_dict(), "Test_Attn_Lstm_Model_2_Weird.pt")

    # Graphing both y and y_hat separately
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.savefig("actual_vs_predicted.png")
    plt.show()
