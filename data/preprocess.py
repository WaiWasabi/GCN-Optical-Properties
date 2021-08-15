import numpy as np
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
dataframe = pd.read_csv(path, delimiter=',')[['Chromophore', 'Solvent', 'Emission max (nm)']].dropna()
dataframe = dataframe[dataframe.Solvent != 'gas']


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


def graph_from_smiles(smiles_list):
    mols = []  # list of objects
    for smiles in smiles_list:
        mols.append(read_smiles(smiles.replace('se', 'Se')))
    edges = map(lambda x: nx.adjacency_matrix(x).todense(), mols)
    nodes = map(lambda x: [to_one_hot(enum[1]) for enum in x.nodes(data='element')], mols)
    return nodes, edges


cro_nodes, cro_edges = graph_from_smiles(dataframe['Chromophore'])
solv_nodes, solv_edges = graph_from_smiles(dataframe['Solvent'])

data_dict = {'cro_nodes': [torch.tensor(nodes) for nodes in cro_nodes],
             'cro_edges': [to_adj_list(edges) for edges in cro_edges],
             'cro_attribs': None,
             'solv_nodes': [torch.tensor(nodes) for nodes in solv_nodes],
             'solv_edges': [to_adj_list(edges) for edges in solv_edges],
             'solv_attribs': None,
             'ys': [torch.tensor(y) for y in dataframe['Emission max (nm)']]}

with open('interim/multi-graph-dict', 'wb') as file:
    pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
