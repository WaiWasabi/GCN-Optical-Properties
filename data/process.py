import pickle

from data.dataset import MoleculeDataset
from torch_geometric.data import DataLoader
from data.vars import batch_size
import os
import shutil

shutil.rmtree('processed')
try:
    os.mkdir('processed')
    os.mkdir('processed/train')
    os.mkdir('processed/test')
except OSError:
    pass

train_ds = MoleculeDataset(os.getcwd(), 'train_rev02.csv', test=False)
test_ds = MoleculeDataset(os.getcwd(), 'test_rev02.csv', test=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, follow_batch=['x_c', 'x_s'], shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, follow_batch=['x_c', 'x_s'], shuffle=True)

with open('processed/train-test-loaders', 'wb') as file:
    pickle.dump([train_loader, test_loader], file, protocol=pickle.HIGHEST_PROTOCOL)
