from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import warnings
from prody import parsePDB, matchChains, matchAlign
import prody
import numpy as np
import seaborn as sns
import glob
import csv
import re
from scipy.stats import pearsonr
from featurizer import *
import glob
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import ast
import torch.multiprocessing as mp
from multiprocessing import Pool
import gc
from featurizer_inter import pdbqtGraphs, pdbGraphs


mp.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")


def tokenizer(sequence):
    "Tokenizes SMILES string"
    pattern =  "[A-Za-z]"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(sequence)]
    return tokens

def makeVariablee(sequence, seqVoc):
    resultVec = []
    char_list = tokenizer(sequence)
    for item in char_list:
        resultVec.append(seqVoc[item])
    return resultVec, len(resultVec)

def flatteningResult(data):
    dataNew = [item for sublist in data for item in sublist]
    return dataNew

class formDatasetTwin_muts_inter(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 element1=None,  element2=None, 
                 mutVecExp=None, mutVecAf=None,
                 smilesExp=None, smilesAf=None, 
                 mutSeqExp=None, mutSeqAf=None,
                 reldock=None, deltay=None, exp=None,
                 blosum62_vectors=None):
        super(formDatasetTwin_muts_inter, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af, self.data_inter_exp, self.data_inter_af = torch.load(self.processed_paths[0])

    def process(self, element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp, blosum62_vectors):

        chunk_size = 400  
        args_list = [(idx, ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex, blosum62_vectors) 
                     for idx, (ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex) in enumerate(zip(element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(15)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_muts_inter, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol_exp = []
            data_list_mol_af = []
            data_list_pro_exp = []
            data_list_pro_af = []
            data_list_inter_exp = []
            data_list_inter_af = []

            for res in result:
                GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af, GCNData_inter_exp, GCNData_inter_af = res.get()
                data_list_mol_exp.append(GCNData_mol_exp)
                data_list_mol_af.append(GCNData_mol_af)
                data_list_pro_exp.append(GCNData_pro_exp)
                data_list_pro_af.append(GCNData_pro_af)
                data_list_inter_exp.append(GCNData_inter_exp)
                data_list_inter_af.append(GCNData_inter_af)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af, data_list_inter_exp, data_list_inter_af), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af, data_list_inter_exp, data_list_inter_af
            gc.collect()

        self.data_mol_exp = []
        self.data_mol_af = []
        self.data_pro_exp = []
        self.data_pro_af = []
        self.data_inter_exp = []
        self.data_inter_af = []
        for chunk_file in chunk_files:
            data_mol_exp, data_mol_af, data_pro_exp, data_pro_af, data_inter_exp, data_inter_af = torch.load(chunk_file)
            self.data_mol_exp.extend(data_mol_exp)
            self.data_mol_af.extend(data_mol_af)
            self.data_pro_exp.extend(data_pro_exp)
            self.data_pro_af.extend(data_pro_af)
            self.data_inter_exp.extend(data_inter_exp)
            self.data_inter_af.extend(data_inter_af)

        torch.save((self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af, self.data_inter_exp, self.data_inter_af), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol_exp)
    
    def __getitem__(self, idx):
        return self.data_mol_exp[idx], self.data_mol_af[idx], self.data_pro_exp[idx], self.data_pro_af[idx], self.data_inter_exp[idx], self.data_inter_af[idx]

def createGcnDataTwin_muts_inter(args):
    
    idx, element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp, blosum62_vectors = args

    expKey = element1
    mutExp = element1.split('_')[0]
    afKey = element2
    afMut = afKey.split('_')[0]
    modi = afKey.split('_')[1]
    ri = afKey.split('_')[2]
    seedi = afKey.split('_')[3]
    afKpi = afKey.split('_')[4]
    
    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smilesExp))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol_exp = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol_exp.__setitem__('c_size', torch.LongTensor([c_size]))

    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smilesAf))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol_af = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol_af.__setitem__('c_size', torch.LongTensor([c_size]))

    ligand_pdb = glob.glob('../data/ligand/{}_*.pdb'.format(element1.split('_')[1]))[0]
    protein_pdbqt = glob.glob('../data/autoDock_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_exp = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)
    
    ligand_pdb = glob.glob('../data/ligand/{}_*.pdb'.format(afKpi))[0]
    protein_pdbqt = glob.glob('../data/autoDock_big/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_af = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)

    contactPath = glob.glob('../data/contact_small/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('../data/ABL_AlphaFold_small/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeqExp, blosum62_vectors)
    GCNData_pro_exp = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_exp.mutVecExp = torch.LongTensor([mutVecExp])
    GCNData_pro_exp.exp = torch.FloatTensor([float(exp)])
    GCNData_pro_exp.expName = element1
    GCNData_pro_exp.__setitem__('target_size', torch.LongTensor([target_size]))
    
    mapAfName = f'{afKey}_contactMap.csv'
    contactPath = '../data/contact_big/{}'.format(mapAfName)
    pdbFile = glob.glob('../data/ABL_AlphaFold_big/data_raw/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(afMut, modi, ri, seedi))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeqAf, blosum62_vectors)
    GCNData_pro_af = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_af.mutVecAf = torch.LongTensor([mutVecAf])
    GCNData_pro_af.relative_dock = torch.FloatTensor([float(reldock)])
    GCNData_pro_af.deltaY = torch.FloatTensor([float(deltay)])
    GCNData_pro_af.afName = element2
    GCNData_pro_af.__setitem__('target_size', torch.LongTensor([target_size]))

    return GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af, GCNData_inter_exp, GCNData_inter_af

