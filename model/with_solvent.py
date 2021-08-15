import torch
from torch.nn import Linear, Module
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from data.vars import atoms
# NOTE: data-dependent parameters
output_dim = 1
input_shape = len(atoms)
# NOTE: hyperparameters
cro_embed = 64
solv_embed = 32
dense_dim = 128


class SolventGCN(Module):
    def __init__(self):
        super(SolventGCN, self).__init__()
        torch.manual_seed(42)
        self.cro_conv0 = GCNConv(input_shape, cro_embed)  # embedding?
        self.cro_conv1 = GCNConv(cro_embed, cro_embed)
        self.cro_conv2 = GCNConv(cro_embed, cro_embed)

        self.solv_conv0 = GCNConv(input_shape, solv_embed)
        self.solv_conv1 = GCNConv(solv_embed, solv_embed)
        self.solv_conv2 = GCNConv(solv_embed, solv_embed)

        self.dense = Linear((cro_embed + solv_embed)*2, dense_dim)
        self.out = Linear(dense_dim, output_dim)

    def forward(self, c, c_edge, c_batch, s, s_edge, s_batch):
        cro = f.relu(self.cro_conv0(c, c_edge))
        cro = f.relu(self.cro_conv1(cro, c_edge))
        cro = f.relu(self.cro_conv2(cro, c_edge))

        solv = f.relu(self.solv_conv0(s, s_edge))
        solv = f.relu(self.solv_conv1(solv, s_edge))
        solv = f.relu(self.solv_conv2(solv, s_edge))

        embed = torch.cat([gmp(cro, c_batch), gap(cro, c_batch),
                           gmp(solv, s_batch), gap(solv, s_batch)], dim=1)

        dense = f.relu(self.dense(embed))
        out = self.out(dense)
        return out, embed
