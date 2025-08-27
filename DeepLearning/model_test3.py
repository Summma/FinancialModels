import torch
from model3 import LSTM
from trading_endpoint import Market
from finance_data import stocks
from model_running_avg import predict

device = torch.device("mps")
model = LSTM(17, 1024, 2, 0.1, 8).to(device)
model.load_state_dict(torch.load("Test_Attn_Lstm_Model_3.pt", map_location=device))
model.eval()

history = 10
horizon = 1
stock = "AAPL"
days = 3650
fee = 0.72
time = "Close"

market = Market(stocks, trade_fee=fee)
market_baseline = Market(stocks, trade_fee=fee)
market.start_market_sim("2010-01-01", days, "1d", 1000.0)
market_baseline.start_market_sim("2010-01-01", days, "1d", 1000.0)
running = True

print("Starting Simulation")
while running:
    X, _ = market.get_data_point(history, horizon)
    dnn_X = X[time].to_numpy()

    if dnn_X.shape != (history, 17):
        break

    probs, logits = model(torch.Tensor(dnn_X).to(device).reshape(1, history, 17))
    cls = torch.argmax(probs).item()

    probs_baseline = predict(X, stock, time)
    cls_baseline = torch.argmax(probs_baseline).item()

    if cls == 0:  # Buy
        market.purchase_stock(stock, -1, time)
    elif cls == 2:  # Sell
        market.sell_stock(stock, -1, time)

    if cls_baseline == 0:  # Buy
        market_baseline.purchase_stock(stock, -1, time)
    elif cls_baseline == 2:  # Sell
        market_baseline.sell_stock(stock, -1, time)

    running = market.step()
    market_baseline.step()

print("Ending Simulation")
market.sell_portfolio()
market_baseline.sell_portfolio()

print(market.balance)
print(market_baseline.balance)
