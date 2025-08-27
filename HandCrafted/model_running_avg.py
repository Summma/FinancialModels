import torch
import numpy as np


def predict(X, stock, time: str = "Close"):
    data = X["Close"][stock].to_numpy()
    n = X.shape[0]
    avg = np.mean(data[: n - 1])
    # print(avg, data[n-1:n])

    if data[n - 1 : n] > avg:
        return torch.tensor([1, 0, 0])
    elif data[n - 1 : n] == avg:
        return torch.tensor([0, 1, 0])
    else:
        return torch.tensor([0, 0, 1])
