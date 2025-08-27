import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model2 import LSTM, make_data, FinanceData
import numpy as np

device = torch.device("mps")
model = LSTM(17, 1024, 2, 0.1, 8).to(device)
model.load_state_dict(
    torch.load("Test_Attn_Lstm_Model_2_Weird.pt", map_location=device)
)
model.eval()

foresight = 30
history = 10
test_data, total = make_data("finance_data.csv", foresight, history, stock="MSFT")

# _, test_data = train_test_split(test_data, test_size=0.3, shuffle=False, random_state=42)
test_size = int(0.3 * len(test_data))
test_data = test_data[-test_size:]

test_data = FinanceData(test_data)

money = 1000
pred_money = 1000
last = 1
pred_last = 1

y_hats = []
ys = []

for i, (X, y, i) in enumerate(test_data):
    y_hat = model(torch.Tensor(X).to(device).reshape(1, history, 17)).item()

    y_hats.append(y_hat)
    ys.append(float(y))

    if y_hat > 1:
        money *= y / last
        pred_money *= y_hat / pred_last

    last = y
    pred_last = y_hat

print(money)
print(pred_money)

t1 = np.array(ys)
t2 = np.array(y_hats)

mask = (t1 > 1) == (t2 > 1)

print(np.sum(mask) / t1.shape[0])
print(np.sum(t1 > 1) / t1.shape[0])
