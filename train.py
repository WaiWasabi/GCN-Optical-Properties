import os

from model.with_solvent import SolventGCN
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

with open('data/processed/train-test-loaders', 'rb') as file:
    train_loader, test_loader = pickle.load(file)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SolventGCN().to(gpu)
mse_loss = torch.nn.MSELoss()
adam = torch.optim.Adam(model.parameters(), lr=0.0006)

"""log, loss_arr = grid_search(train_loader, [0.0001, 0.01], 0.001, [32, 128], 16, 15, gpu)

logfile(log, "heatmap")
logfile(loss_arr, "loss-trends")"""

print("Starting training...")
losses = []
logs = []
epochs = 50

for epoch in range(epochs):
    loss, embedding, log = train_with_solvent(model, train_loader, mse_loss, adam, gpu, history=True)
    losses.append(loss)
    logs.extend(log)
    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1} | Train Loss {loss}")

model_path = 'model/saved_models/emax-solvent'
torch.save(model.state_dict(), model_path)

# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(gpu)
    pred, embed = model(test_batch.x_c.float(), test_batch.edge_index_c, test_batch.edge_attr_c, test_batch.x_c_batch,
                        test_batch.x_s.float(), test_batch.edge_index_s, test_batch.edge_attr_s, test_batch.x_s_batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row)
df["y_pred"] = df["y_pred"].apply(lambda row: row)
print(df)

losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i, l in enumerate(losses_float)]
plt.plot(loss_indices, losses_float)
plt.show()
