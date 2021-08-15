from torch_geometric.data import Data
import torch
import pickle
import pandas as pd

interim = 'data/interim/multi-graph-dict'
processed = 'data/processed/em-with-solvent'

with open(interim, 'rb') as file:
    data = pickle.load(file)

# torch.set_printoptions(profile="full")

for i in range(10):
    print("c-graph shape:", data['cro_nodes'][i].shape)
    print("s-graph shape:", data['solv_nodes'][i].shape)
    print("c-edge shape:", data['cro_edges'][i].shape)
    print("s-edge shape:", data['solv_edges'][i].shape)
    print('-------')

"""for batch in train:
    print(batch)"""


