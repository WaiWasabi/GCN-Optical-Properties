import torch
from torch.nn import Linear, Module
import torch.nn.functional as f
from torch_geometric.nn import norm, GCNConv, DeepGCNLayer, global_mean_pool as gap, global_max_pool as gmp
from data.vars import *
# NOTE: data-dependent parameters
output_dim = 1
input_shape = node_embed
# NOTE: hyperparameters
cro_embed = 64
solv_embed = 32
dense_dim = 128
num_layers = 3


class SolventGCN(Module):
    def __init__(self):
        super(SolventGCN, self).__init__()
        torch.manual_seed(42)

        self.cro_encoder = Linear(node_embed, cro_embed)
        self.solv_encoder = Linear(node_embed, solv_embed)

        self.c0 = GCNConv(cro_embed, cro_embed)
        self.s0 = GCNConv(solv_embed, solv_embed)

        self.c_layers = torch.nn.ModuleList()
        self.s_layers = torch.nn.ModuleList()

        for i in range(num_layers-1):
            conv_c = GCNConv(cro_embed, cro_embed)
            act_c = torch.nn.ReLU(inplace=True)
            norm_c = torch.nn.LayerNorm(cro_embed)

            conv_s = GCNConv(solv_embed, solv_embed)
            act_s = torch.nn.ReLU(inplace=True)
            norm_s = torch.nn.LayerNorm(solv_embed)

            self.c_layers.append(DeepGCNLayer(conv_c, norm_c, act_c, dropout=0.2))
            self.s_layers.append(DeepGCNLayer(conv_s, norm_s, act_s, dropout=0.2))

        self.dense = Linear((cro_embed + solv_embed)*2, dense_dim)
        self.out = Linear(dense_dim, output_dim)

    def forward(self, c, c_edge, c_attrib, c_batch, s, s_edge, s_attrib, s_batch):
        cro = self.cro_encoder(c)
        solv = self.solv_encoder(s)
        cro = self.c0(cro, c_edge)
        solv = self.s0(solv, s_edge)

        for layer in self.c_layers:
            cro = layer(cro, c_edge)

        for layer in self.s_layers:
            solv = layer(solv, s_edge)

        embed = torch.cat([gmp(cro, c_batch), gap(cro, c_batch),
                           gmp(solv, s_batch), gap(solv, s_batch)], dim=1)

        dense = f.relu(self.dense(embed))
        out = self.out(dense)
        return out, embed
