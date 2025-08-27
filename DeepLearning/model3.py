from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import yfinance as yf
from finance_data import stocks


def make_data(
    data_file: str, foresight: int = 30, context_length: int = 30, stock: str = "NVDA"
):
    df = pd.read_csv(data_file, header=[0, 1])
    # tickers = yf.Tickers(" ".join(stocks))
    # df = tickers.download(
    #     start="2025-05-01",
    #     end="2025-05-08",
    #     interval="1m"
    # )

    n = df.shape[0] - foresight - context_length

    data = []
    total = 0
    for i in range(n):
        X = df[i : i + context_length]["Close"].to_numpy()

        idx = i + context_length + foresight

        value = df[idx : idx + 1]["Close"][stock].to_numpy()
        prev_value = df[idx - 1 : idx]["Close"][stock].to_numpy()
        ratio = value / prev_value

        if ratio >= 1.01:
            y = np.array([1, 0, 0])  # Buy
        elif 1 <= ratio < 1.01:
            y = np.array([0, 1, 0])  # Hold
        elif ratio < 1:
            y = np.array([0, 0, 1])  # Sell

        data.append([X, y, i])
        total += value

    return data, total


class FinanceData(Dataset):
    def __init__(self, data):
        self.data = [
            (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                idx,
            )
            for X, y, idx in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.softmax = nn.Softmax(dim=1)

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

        logits = self.head(context)
        probs = self.softmax(logits)

        return probs, logits


def array_to_class(y):
    return np.argmax(y, axis=1)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            y = y.to(device)

            probs, logits = model(X)

            loss = loss_fn(logits, y)
            losses.append(loss.item())

            y_true.extend(y.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

    p_y_true = array_to_class(np.array(y_true))
    p_y_pred = array_to_class(np.array(y_pred))
    f1 = f1_score(p_y_true, p_y_pred, average="macro")
    accuracy = accuracy_score(p_y_true, p_y_pred)
    cm = confusion_matrix(p_y_true, p_y_pred, labels=[0, 1, 2])
    return np.mean(losses), float(f1), float(accuracy), cm, y_true, y_pred


def train(model, loader, loss_fn, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        losses = []
        for i, (X, y, idx) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            probs, logits = model(X)
            l = loss_fn(logits, y)
            losses.append(l.item())

            l.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {np.mean(losses)}")


def plot_class_counts(data):
    counts = [0, 0, 0]
    for datum in data:
        y_val = datum[1]
        counts[np.argmax(y_val)] += 1

    plt.bar(range(len(counts)), counts)
    plt.savefig("graphs/class_counts.png")
    plt.show()


if __name__ == "__main__":
    tscv = TimeSeriesSplit(n_splits=5)
    device = torch.device("mps")
    model = LSTM(17, 1024, 2, 0.1, 8).to(device)

    data, total = make_data("finance_data.csv", 1, 10, stock="MSFT")

    test_size = int(0.3 * len(data))
    train_val = data[:-test_size]
    final_test = data[-test_size:]

    weights = compute_class_weight(
        "balanced",
        classes=np.arange(3),
        y=array_to_class(np.array([datum[1] for datum in train_val])),
    )

    print(weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss(
        weight=torch.tensor(weights).to(torch.float32).to(device)
    )

    # for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val)):
    #     train_data = [data[i] for i in train_idx]
    #     val_data = [data[i] for i in val_idx]

    #     train_loader = DataLoader(FinanceData(train_data), batch_size=32, shuffle=False)
    #     val_loader = DataLoader(FinanceData(val_data), batch_size=32, shuffle=False)

    #     train(model, train_loader, loss, optimizer, device, epochs=20)

    #     losses, f1, accuracy, cm, _, _ = evaluate(model, val_loader, loss, device)
    #     print(f"Test Loss on fold {fold + 1}: {np.mean(losses)}")
    #     print(f"Test F1-Score on fold {fold + 1}: {f1}")
    #     print(f"Test Accuracy on fold {fold + 1}: {accuracy}")

    final_val = DataLoader(FinanceData(train_val), batch_size=32, shuffle=False)
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    train(model, final_val, loss, optimizer, device, epochs=1)

    test = DataLoader(FinanceData(final_test), batch_size=32, shuffle=False)
    losses, f1, accuracy, cm, y_true, y_pred = evaluate(model, test, loss, device)
    print(f"Test Loss on final validation set: {np.mean(losses)}")
    print(f"Test F1-Score on final validation set: {f1}")
    print(f"Test Accuracy on final validation set: {accuracy}")

    print(f"Confusion Matrix:\n{cm}")

    torch.save(model.state_dict(), "Test_Attn_Lstm_Model_3.pt")
