from model.emax_regression import GCN
from model.train_step import *
from torch_geometric.data import DataLoader, Data
import torch
import pickle
import warnings

warnings.filterwarnings('ignore')
data_path = 'data/processed/cro_emax'

with open(data_path, 'rb') as file:
    data = pickle.load(file)

batch_size = 64
data_size = len(data)
train_loader = DataLoader(data[:int(data_size * 0.8)], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], batch_size=batch_size, shuffle=True)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GCN().to(gpu)

mse_loss = torch.nn.MSELoss()
adam = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
losses = []
logs = []
epochs = 20
for epoch in range(epochs):
    loss, embedding, log = train_step(model, train_loader, mse_loss, adam, gpu, history=True)
    losses.append(loss)
    logs.extend(log)
    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Train Loss {loss}")

model_path = 'model/saved_models/emax'
torch.save(model.state_dict(), model_path)
