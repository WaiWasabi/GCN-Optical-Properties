from torch_geometric.data import Data
import torch
import pickle

with open('data/processed/cro_emax', 'rb') as file:
    data = pickle.load(file)
torch.set_printoptions(profile="full")

print(data[0].x)
