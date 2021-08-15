import torch
from torch.utils.data import Dataset


class MultiGraphDataset(Dataset):
    def __init__(self, data_dict):
        self.x_c = data_dict['cro_nodes']
        self.x_s = data_dict['solv_nodes']
        self.edge_c = data_dict['cro_edges']
        self.edge_s = data_dict['solv_edges']
        self.y = data_dict['ys']

    def __getitem__(self, index):
        return self.x_c[index], self.x_s[index], self.edge_c[index], self.edge_s[index], self.y[index]

    def __len__(self):
        return len(self.y)
