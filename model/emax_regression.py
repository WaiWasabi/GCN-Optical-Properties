import torch
from torch.nn import Linear, Module
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from data.vars import atoms
# NOTE: data-dependent parameters
output_dim = 1
input_shape = len(atoms)
# NOTE: hyperparameters
embedding_size = 64


class GCN(Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv0 = GCNConv(input_shape, embedding_size)  # embedding?
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.out = Linear(embedding_size * 2, output_dim)

    def forward(self, x, edge_index, batch_index):
        hidden = f.tanh(self.conv0(x, edge_index))
        hidden = f.tanh(self.conv1(hidden, edge_index))
        hidden = f.tanh(self.conv2(hidden, edge_index))
        hidden = f.tanh(self.conv3(hidden, edge_index))
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out, hidden
