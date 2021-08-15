from dataset import MultiGraphDataset
from torch.utils.data import DataLoader
from vars import batch_size
import pickle

with open('interim/multi-graph-dict', 'rb') as file:
    data_dict = pickle.load(file)

x_c, x_s, edge_c, edge_s, y = data_dict['x_c'], data_dict['x_s'], \
                              data_dict['edge_c'], data_dict['edge_s'], data_dict['y']

dataset = MultiGraphDataset(data_dict)
size = len(dataset)
train_ds = dataset[:int(size*0.8)]
test_ds = dataset[int(size*0.8):]



def collate_fn(batch):
    x_c, x_s, edge_c, edge_s, y = batch
    print(len(x_c))
    return {'x_c': x_c, 'x_s': x_s, 'edge_c': edge_c, 'edge_s': edge_s, 'y': y}


train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

for item in train_loader:
    print(type(item))
    print('lol')


"""with open('processed/em-with-solvent', 'wb') as file:
    pickle.dump(train_loader, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_loader, file, protocol=pickle.HIGHEST_PROTOCOL)"""
