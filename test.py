from torch_geometric.data import Data
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from data.dataset import MoleculeDataset

interim = 'data/interim/multi-graph-dict'
processed = 'data/processed/em-with-solvent'

dataset = MoleculeDataset('data', 'rev02.csv')
