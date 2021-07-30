import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysmiles import read_smiles
import networkx as nx
import torch
from torch_geometric.data import Data
import pickle
import logging
from vars import *

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

path = 'raw/rev02.csv'
dataframe = pd.read_csv(path, delimiter=',')[['Chromophore', 'Emission max (nm)']].dropna()


def to_one_hot(atom):  # generate one-hot encodings for an individual atom
    if atom in atoms:
        return [int(atom == valid) for valid in atoms]
    else:
        return [int(i + 1 == len(atoms)) for i in range(len(atoms))]


def pad_vec(vec_list, target):
    np_arr = np.array(vec_list)
    return np.append(np_arr, np.zeros([target - np_arr.shape[0], np_arr.shape[1]]), axis=0)


def to_adj_list(adj_matrix):
    output = [[], []]
    for row in range(adj_matrix.shape[0]):
        for col in range(adj_matrix.shape[1]):
            if adj_matrix[row, col] != 0:
                output[0].append(row)
                output[1].append(col)
    return torch.LongTensor(output)


mols = []  # list of objects

for smiles in dataframe['Chromophore']:
    mols.append(read_smiles(smiles.replace('se', 'Se')))

adjs = map(lambda x: nx.adjacency_matrix(x).todense(), mols)
vecs = map(lambda x: [to_one_hot(enum[1]) for enum in x.nodes(data='element')], mols)

data_dict = {'xs': [torch.tensor(pad_vec(vec, MAX_SIZE)) for vec in vecs],
             'edge_indices': [to_adj_list(adj) for adj in adjs],
             'edge_attribs': None,
             'ys': [torch.tensor(y) for y in dataframe['Emission max (nm)']]}

data = [Data(x=vec, edge_index=adj, y=y) for vec, adj, y in
        zip(data_dict['xs'], data_dict['edge_indices'], data_dict['ys'])]

with open('processed/cro_emax', 'wb') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
