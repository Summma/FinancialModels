from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def make_data(data_file: str, foresight: int = 30, context_length: int = 30):
    df = pd.read_csv(data_file, header=[0, 1])

    n = df.shape[0] - foresight - context_length

    data = []
    total = 0
    for i in range(n):
        X = df[i : i + context_length]["Close"].to_numpy()
        try:
            y_now = df[
                i + foresight + context_length : i + foresight + context_length + 1
            ]["Close"]["NVDA"].to_numpy()
            y_past = df[i + context_length : i + context_length + 1]["Close"][
                "NVDA"
            ].to_numpy()
            y = y_now / y_past
        except KeyError:
            print(
                df[i + foresight + context_length : i + foresight + context_length + 1][
                    "Close"
                ]
            )

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
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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


if __name__ == "__main__":
    device = torch.device("mps")
    model = LSTM(17, 1024, 2, 0.1, 8).to(device)

    data, total = make_data("finance_data.csv", 100, 10)

    train_data, test_data = train_test_split(
        data, test_size=0.3, shuffle=False, random_state=42
    )

    test_loader = DataLoader(FinanceData(test_data), batch_size=32, shuffle=False)
    train_loader = DataLoader(FinanceData(train_data), batch_size=32, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss = nn.HuberLoss(delta=1.0)

    for epoch in range(20):
        losses = []
        for i, (X, y, idx) in enumerate(train_loader):
            X = X.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            losses.append(l.item())

            l.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {np.mean(losses)}")

    # Evaluate the model on the test set
    model.eval()
    y_hats = []
    ys = []
    indices = []
    with torch.no_grad():
        test_loss = []
        for X, y, idx in test_loader:
            X = X.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            # y_hat = np.exp(model(X).cpu()).to(device)
            y_hat = model(X).to(device)

            l = loss(y_hat, y)

            test_loss.append(l.item())

            y_hats.extend(map(lambda x: x.item(), y_hat.detach().cpu().numpy()))
            ys.extend(map(lambda x: x.item(), y.detach().cpu().numpy()))
            indices.extend(map(lambda x: x.item(), idx.detach().cpu().numpy()))

        print(f"Test Loss: {np.mean(test_loss)}")

    torch.save(model.state_dict(), "Test_Attn_Lstm_Model.pt")

    ys = sorted(ys, key=lambda x: indices[ys.index(x)])
    y_hats = sorted(y_hats, key=lambda x: indices[y_hats.index(x)])
    indices = sorted(indices)

    # Graphing both y and y_hat separately
    plt.plot(indices, ys, label="Actual")
    plt.plot(indices, y_hats, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.savefig("actual_vs_predicted.png")
    plt.show()

    t1 = np.array(ys)
    t2 = np.array(y_hats)

    mask = (t1 > 1) == (t2 > 1)
    print(np.sum(mask))
    print(np.sum(mask) / t1.shape[0])
