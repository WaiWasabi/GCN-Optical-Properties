from torch_geometric.datasets import MoleculeNet

interim = 'data/interim/multi-graph-dict'
processed = 'data/processed/em-with-solvent'

data = MoleculeNet(root=".", name="ESOL")
print(data[0].edge_attr.shape)
print(data[0].edge_index.shape)
print(data[0].x.shape)
