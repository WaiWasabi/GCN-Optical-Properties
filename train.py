from model.emax_regression import GCN
from model.train_step import *
from model.hparam_tuner import grid_search
from torch_geometric.data import DataLoader
from logs.log import logfile
import torch
import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')
data_path = 'data/interim/unpadded_emax'

with open(data_path, 'rb') as file:
    data = pickle.load(file)

batch_size = 64
data_size = len(data)
train_loader = DataLoader(data[:int(data_size * 0.8)], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], batch_size=batch_size, shuffle=True)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = GCN().to(gpu)

# mse_loss = torch.nn.MSELoss()
# adam = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-4)

log, loss_arr = grid_search(train_loader, [0.0001, 0.01], 0.001, [32, 128], 16, 15, gpu)

logfile(log, "heatmap")
logfile(loss_arr, "loss-trends")

"""print("Starting training...")
losses = []
logs = []
epochs = 50
for epoch in range(epochs):
    loss, embedding, log = train_cro_only(model, train_loader, mse_loss, adam, gpu, history=True)
    losses.append(loss)
    logs.extend(log)
    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Train Loss {loss}")

model_path = 'model/saved_models/emax'
torch.save(model.state_dict(), model_path)


# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(gpu)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row)
df["y_pred"] = df["y_pred"].apply(lambda row: row)
print(df)

losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i,l in enumerate(losses_float)]
plt.plot(loss_indices, losses_float)
plt.show()
"""
