from torch_geometric.data import Dataset, Data
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
import numpy as np
import os


def _get_node_features(mol):
    all_node_feats = []

    for atom in mol.GetAtoms():
        node_feats = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetHybridization(),
                      atom.GetIsAromatic(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), atom.IsInRing(),
                      atom.GetChiralTag()]
        """Feature 1: Atomic number
        Feature 2: Atom degree
        Feature 3: Formal charge
        Feature 4: Hybridization
        Feature 5: Aromaticity
        Feature 6: Total Num Hs
        Feature 7: Radical Electrons
        Feature 8: In Ring
        Feature 9: Chirality"""
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)


def _get_labels(label):
    label = np.asarray([label])
    return torch.tensor(label, dtype=torch.int64)


def _get_edge_features(mol):
    """
    This will return a matrix / 2d array of the shape
    [Number of edges, Edge Feature size]
    """
    all_edge_feats = []

    for bond in mol.GetBonds():
        edge_feats = [bond.GetBondTypeAsDouble(), bond.IsInRing()]
        """Feature 1: Bond type (as double)
        Feature 2: Rings
        Append node features to matrix (twice, per direction)"""
        all_edge_feats += [edge_feats, edge_feats]

    all_edge_feats = np.asarray(all_edge_feats)
    return torch.tensor(all_edge_feats, dtype=torch.float)


def _get_adjacency_info(mol):
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices


class MolPair(Data):
    def __init__(self, x_c, edge_c, attrib_c, x_s, edge_s, attrib_s, y):
        super(MolPair, self).__init__()
        self.y = y
        self.attrib_s = attrib_s
        self.edge_s = edge_s
        self.x_s = x_s
        self.attrib_c = attrib_c
        self.edge_c = edge_c
        self.x_c = x_c

    def __inc__(self, key, value):
        if key == 'edge_c':
            return self.x_c.size(0)
        if key == 'edge_s':
            return self.x_s.size(0)
        else:
            return super().__inc__(key, value)


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.root = root
        self.filename = filename
        self.test = test
        self.data = pd.read_csv(self.raw_paths[0])[['Chromophore', 'Solvent', 'Emission max (nm)']].dropna()
        self.data = self.data[self.data.Solvent != 'gas'].reset_index(inplace=False, drop=True)
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        idx = self.data.reset_index(inplace=False, drop=True)
        if self.test:
            return [os.path.join('test', f'{i}.pt') for i in list(idx.index)]
        else:
            return [os.path.join('train', f'{i}.pt') for i in list(idx.index)]

    def download(self):
        pass

    def process(self):
        for index, series in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_c = Chem.MolFromSmiles(series['Chromophore'])
            mol_s = Chem.MolFromSmiles(series['Solvent'])
            x_c = _get_node_features(mol_c)
            x_s = _get_node_features(mol_s)
            edge_c = _get_adjacency_info(mol_c)
            edge_s = _get_adjacency_info(mol_s)
            attrib_c = _get_edge_features(mol_c)
            attrib_s = _get_edge_features(mol_s)
            label = _get_labels(series['Emission max (nm)'])
            data = MolPair(x_c, edge_c, attrib_c, x_s, edge_s, attrib_s, label)
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, 'test', f'{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'train', f'{index}.pt'))

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 'test', f'{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 'train', f'{idx}.pt'))
        return data
