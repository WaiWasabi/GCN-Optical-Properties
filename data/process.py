from dataset import MultiGraphDataset
from torch.utils.data import DataLoader
from vars import batch_size
import pickle
from random import shuffle

with open('interim/multi-graph-dict', 'rb') as file:
    data = list(zip(*pickle.load(file)))

shuffle(data)
split = int(len(data)*0.8)
train_ds = MultiGraphDataset(data[:split])
test_ds = MultiGraphDataset(data[split:])


def collate_fn(batch):
    x_c, x_s, edge_c, edge_s, y = zip(*batch)
    return {'x_c': x_c, 'x_s': x_s, 'edge_c': edge_c, 'edge_s': edge_s, 'y': y}


train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

with open('processed/em-with-solvent', 'wb') as file:
    pickle.dump([train_loader, test_loader], file, protocol=pickle.HIGHEST_PROTOCOL)
