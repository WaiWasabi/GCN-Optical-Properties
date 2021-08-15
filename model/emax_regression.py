import torch
from torch.nn import Linear, Module
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from data.vars import atoms
# NOTE: data-dependent parameters
output_dim = 1
input_shape = len(atoms)
# NOTE: hyperparameters
dense_shape = 128


class GCN(Module):
    def __init__(self, embedding_size):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv0 = GCNConv(input_shape, embedding_size)  # embedding?
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.dense0 = Linear(embedding_size * 2, dense_shape)
        self.dense1 = Linear(dense_shape, dense_shape)
        self.dense2 = Linear(dense_shape, dense_shape)
        self.out = Linear(dense_shape, 1)

    def forward(self, x, edge_index, batch_index):
        hidden = f.relu(self.conv0(x, edge_index))
        hidden = f.relu(self.conv1(hidden, edge_index))
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        mlp = f.relu(self.dense0(hidden))
        mlp = f.dropout(mlp, 0.2)
        mlp = f.relu(self.dense1(mlp))
        mlp = f.dropout(mlp, 0.2)
        mlp = f.relu(self.dense2(mlp))
        out = self.out(mlp)
        return out, hidden
