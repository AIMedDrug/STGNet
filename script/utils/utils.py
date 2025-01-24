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
from featurizer_inter import pdbqtGraphs, pdbGraphs, pdbGraphsPred


mp.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")

class formDataset(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 kpi=None, mutation=None, wildSeq=None, 
                 muSeq=None, strain=None, y=None, 
                 ecfp=None, smiles=None,
                 wildReSeq=None, mutReSeq=None,
                 subPath=None):
        super(formDataset, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(kpi, mutation, wildSeq, muSeq, strain, y, ecfp, smiles, wildReSeq, mutReSeq, subPath)
            self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, kpi, mutation, wildSeq, muSeq, strain, y, ecfp, smiles, wildReSeq, mutReSeq, subPath):
        data_list = []
        
        for idx, (kpii, muName, wids, mus, stra, yy, ep, smi) in enumerate(zip(kpi, mutation, wildSeq, muSeq, strain, y, ecfp, smiles)):
            keyname = muName + '_' + kpii
            node_f, edge_index, edge_attr, bonds = mol2vec(Chem.MolFromSmiles(smi))
            for bond in bonds: 
                edge_attr.append(bond_features(bond))   
                data = Data(x=torch.tensor(node_f, dtype=torch.float),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
                        )
            
            data.muName = muName
            data.wildSeq = torch.LongTensor([wids])
            data.muSeq = torch.LongTensor([mus])
            data.strain = torch.FloatTensor([stra])
            data.y = torch.FloatTensor([float(yy)])

            alignment = align_sequence(mutReSeq[keyname], wildReSeq[keyname])
            alignment_file = "/home/data1/BGM/mdrugEffect/data/multiMutation_ProteinGcn/aln/{}_aligned_sequences_with_ref.aln".format(keyname)
            save_alignment_to_file(alignment, alignment_file, mutReSeq[keyname])
            contactPath = glob.glob('../../mutationEffect/data/pocket/contact/contact_{}*.csv'.format(keyname.split('_')[0]))[0]
            pdbFile = glob.glob('../../mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(keyname.split('_')[0], keyname.split('_')[0]))[0]
            
            target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(keyname, contactPath, mutReSeq[keyname], alignment_file, pdbFile)
            
            data.protein_cSize = torch.tensor([target_size], dtype=torch.float)
            data.protein_edge_index = torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0)
            data.protein_edge_feature = torch.tensor(target_feature, dtype=torch.float)
            data.proten_weight = torch.tensor(target_edge_weight, dtype=torch.float)
            data.pos_matrix = torch.tensor(pos_matrix, dtype=torch.float)

            data_list.append(data)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

class formDataset02(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 mutation=None, wildSeq=None, 
                 muSeq=None, strain=None, y=None, 
                 smiles=None,
                 wildReSeq=None, mutReSeq=None,
                 blosum62_vectors=None):
        super(formDataset02, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(mutation, wildSeq, muSeq, strain, y, smiles, wildReSeq, mutReSeq, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro_wild, self.data_pro_mut = torch.load(self.processed_paths[0])

    def process(self, mutation, wildSeq, muSeq, strain, y, smiles, wildReSeq, mutReSeq, blosum62_vectors):
        resultDock = {}
        dockMod1File = '../bigData/resistance_mod1_top12.csv'
        with open(dockMod1File, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                wilddock = ast.literal_eval(row[2])
                mutdock = ast.literal_eval(row[3])
                resultDock[row[0]] = wilddock + mutdock

        chunk_size = 1000  
        args_list = [(idx, muName, wids, mus, stra, yy, smi, resultDock, wildReSeq, mutReSeq, blosum62_vectors) 
                     for idx, (muName, wids, mus, stra, yy, smi) in enumerate(zip(mutation, wildSeq, muSeq, strain, y, smiles))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnData, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro_wild = []
            data_list_pro_mut = []

            for res in result:
                GCNData_mol, GCNData_pro_wild, GCNData_pro_mut = res.get()
                data_list_mol.append(GCNData_mol)
                data_list_pro_wild.append(GCNData_pro_wild)
                data_list_pro_mut.append(GCNData_pro_mut)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro_wild, data_list_pro_mut), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro_wild, data_list_pro_mut
            gc.collect()

        self.data_mol = []
        self.data_pro_wild = []
        self.data_pro_mut = []
        for chunk_file in chunk_files:
            data_mol, data_pro_wild, data_pro_mut = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro_wild.extend(data_pro_wild)
            self.data_pro_mut.extend(data_pro_mut)

        torch.save((self.data_mol, self.data_pro_wild, self.data_pro_mut), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro_wild[idx], self.data_pro_mut[idx]

def createGcnData(args):
    idx, muName, wids, mus, stra, yy, smi, resultDock, wildReSeq, mutReSeq, blosum62_vectors = args

    keyname = muName
    muti = keyname.split('_')[0]
    modi = keyname.split('_')[1]
    ri = keyname.split('_')[2]
    seedi = keyname.split('_')[3]
    lig = keyname.split('_')[4]

    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smi))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

    contactPath = '../bigData/contact/{}_contactMap_wild.csv'.format(keyname)
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_WILD_600k_filtering_8dc32_256_512_32/pdb/model_{}_ptm_{}_{}.pdb'.format(modi, ri, seedi))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, wildReSeq[keyname], blosum62_vectors)
    GCNData_pro_wild = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_wild.wildSeq = torch.LongTensor([wids])
    GCNData_pro_wild.vina = torch.FloatTensor([resultDock[keyname][:5]])
    GCNData_pro_wild.__setitem__('target_size', torch.LongTensor([target_size]))

    contactPath = '../bigData/contact/{}_contactMap.csv'.format(keyname)

    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(muti, modi, ri, seedi))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutReSeq[keyname], blosum62_vectors)
    GCNData_pro_mut = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_mut.muSeq = torch.LongTensor([mus])
    GCNData_pro_mut.strain = torch.FloatTensor([stra])
    GCNData_pro_mut.vina = torch.FloatTensor([resultDock[keyname][5:]])
    GCNData_pro_mut.y = torch.FloatTensor([float(yy)])

    GCNData_pro_mut.mutName = keyname
    GCNData_pro_mut.__setitem__('target_size', torch.LongTensor([target_size]))

    return GCNData_mol, GCNData_pro_wild, GCNData_pro_mut


class formDatasetTwin(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 mutation=None,  smiles=None, 
                 mutVec=None, mutSeq=None,
                 exp = None, relativeDock=None, deltaY=None,
                 blosum62_vectors=None):
        super(formDatasetTwin, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(mutation, smiles, mutVec, mutSeq, exp, relativeDock, deltaY, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro_exp, self.data_pro_dock = torch.load(self.processed_paths[0])

    def process(self, mutation, smiles, mutVec, mutSeq, exp, relativeDock, deltaY, blosum62_vectors):
        
        chunk_size = 1000  
        args_list = [(idx, muName, smi, muVec, muSeq, ex, relDock, delY, blosum62_vectors) 
                     for idx, (muName, smi, muVec, muSeq, ex, relDock, delY) in enumerate(zip(mutation, smiles, mutVec, mutSeq, exp, relativeDock, deltaY))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro_exp = []
            data_list_pro_dock = []

            for res in result:
                GCNData_mol, GCNData_pro_exp, GCNData_pro_dock = res.get()
                data_list_mol.append(GCNData_mol)
                data_list_pro_exp.append(GCNData_pro_exp)
                data_list_pro_dock.append(GCNData_pro_dock)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro_exp, data_list_pro_dock), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro_exp, data_list_pro_dock
            gc.collect()

        self.data_mol = []
        self.data_pro_exp = []
        self.data_pro_dock = []
        for chunk_file in chunk_files:
            data_mol, data_pro_exp, data_pro_dock = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro_exp.extend(data_pro_exp)
            self.data_pro_dock.extend(data_pro_dock)

        torch.save((self.data_mol, self.data_pro_exp, self.data_pro_dock), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro_exp[idx], self.data_pro_dock[idx]

def createGcnDataTwin(args):
    idx, mutation, smiles, mutVec, mutSeq, exp, relativeDock, deltaY, blosum62_vectors = args
    
    keyname = mutation
    muti = keyname.split('_')[0]
    modi = keyname.split('_')[1]
    ri = keyname.split('_')[2]
    seedi = keyname.split('_')[3]
    lig = keyname.split('_')[4]

    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smiles))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(muti))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(muti, muti))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeq, blosum62_vectors)
    GCNData_pro_exp = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_exp.mutVec = torch.LongTensor([mutVec])
    GCNData_pro_exp.exp = torch.FloatTensor([float(exp)])
    GCNData_pro_exp.__setitem__('target_size', torch.LongTensor([target_size]))

    contactPath = '../bigData/contact/{}_contactMap.csv'.format(keyname)
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(muti, modi, ri, seedi))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeq, blosum62_vectors)
    GCNData_pro_dock = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_dock.mutVec = torch.LongTensor([mutVec])
    GCNData_pro_dock.relativeDock = torch.FloatTensor([float(relativeDock)])
    GCNData_pro_dock.deltaY = torch.FloatTensor([float(deltaY)])

    GCNData_pro_dock.mutName = keyname
    GCNData_pro_dock.__setitem__('target_size', torch.LongTensor([target_size]))

    return GCNData_mol, GCNData_pro_exp, GCNData_pro_dock

class formDatasetTwin_wildMut(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 element1=None,  element2=None, 
                 wildVec=None, mutVec=None,
                 smilesExp=None, smilesAf=None, 
                 wildSeq=None, mutSeq=None,
                 deltay=None, exp=None,
                 blosum62_vectors=None):
        super(formDatasetTwin_wildMut, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(element1, element2, wildVec, mutVec, smilesExp, smilesAf, wildSeq, mutSeq, deltay, exp, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af = torch.load(self.processed_paths[0])

    def process(self, element1, element2, wildVec, mutVec, smilesExp, smilesAf, wildSeq, mutSeq, deltay, exp, blosum62_vectors):

        chunk_size = 1000  
        args_list = [(idx, ele1, ele2, wvec, mvec, smiExp, smiAf, wSeq, mSeq, delta, ex, blosum62_vectors) 
                     for idx, (ele1, ele2, wvec, mvec, smiExp, smiAf, wSeq, mSeq, delta, ex) in enumerate(zip(element1, element2, wildVec, mutVec, smilesExp, smilesAf, wildSeq, mutSeq, deltay, exp))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_wildMut, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol_exp = []
            data_list_mol_af = []
            data_list_pro_exp = []
            data_list_pro_af = []

            for res in result:
                GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af = res.get()
                data_list_mol_exp.append(GCNData_mol_exp)
                data_list_mol_af.append(GCNData_mol_af)
                data_list_pro_exp.append(GCNData_pro_exp)
                data_list_pro_af.append(GCNData_pro_af)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af
            gc.collect()

        self.data_mol_exp = []
        self.data_mol_af = []
        self.data_pro_exp = []
        self.data_pro_af = []
        for chunk_file in chunk_files:
            data_mol_exp, data_mol_af, data_pro_exp, data_pro_af = torch.load(chunk_file)
            self.data_mol_exp.extend(data_mol_exp)
            self.data_mol_af.extend(data_mol_af)
            self.data_pro_exp.extend(data_pro_exp)
            self.data_pro_af.extend(data_pro_af)

        torch.save((self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol_exp)
    
    def __getitem__(self, idx):
        return self.data_mol_exp[idx], self.data_mol_af[idx], self.data_pro_exp[idx], self.data_pro_af[idx]

def createGcnDataTwin_wildMut(args):
    
    idx, element1, element2, wildVec, mutVec, smilesExp, smilesAf, wildSeq, mutSeq, deltay, exp, blosum62_vectors = args

    expKey = element1
    mutExp = element1.split('_')[0]
    afKey = element2
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

    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeq, blosum62_vectors)
    GCNData_pro_exp = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_exp.mutVec = torch.LongTensor([mutVec])
    GCNData_pro_exp.exp = torch.FloatTensor([float(exp)])
    GCNData_pro_exp.expName = element1
    GCNData_pro_exp.__setitem__('target_size', torch.LongTensor([target_size]))
    
    mapAfName = f'{mutExp}_{modi}_{ri}_{seedi}_{afKpi}_contactMap_wild.csv'
    contactPath = '../bigData/contact/{}'.format(mapAfName)
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_WILD_*/pdb/model_{}_ptm_{}_{}.pdb'.format(modi, ri, seedi))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, wildSeq, blosum62_vectors)
    GCNData_pro_af = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_af.wildVec = torch.LongTensor([wildVec])
    GCNData_pro_af.deltaY = torch.FloatTensor([float(deltay)])
    GCNData_pro_af.afName = element2
    GCNData_pro_af.__setitem__('target_size', torch.LongTensor([target_size]))

    return GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af

class formDatasetTwin_muts(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 element1=None,  element2=None, 
                 mutVecExp=None, mutVecAf=None,
                 smilesExp=None, smilesAf=None, 
                 mutSeqExp=None, mutSeqAf=None,
                 reldock=None, deltay=None, exp=None,
                 blosum62_vectors=None):
        super(formDatasetTwin_muts, self).__init__(root)
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
        self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af = torch.load(self.processed_paths[0])

    def process(self, element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp, blosum62_vectors):

        chunk_size = 1000  
        args_list = [(idx, ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex, blosum62_vectors) 
                     for idx, (ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex) in enumerate(zip(element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_muts, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol_exp = []
            data_list_mol_af = []
            data_list_pro_exp = []
            data_list_pro_af = []

            for res in result:
                GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af = res.get()
                data_list_mol_exp.append(GCNData_mol_exp)
                data_list_mol_af.append(GCNData_mol_af)
                data_list_pro_exp.append(GCNData_pro_exp)
                data_list_pro_af.append(GCNData_pro_af)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol_exp, data_list_mol_af, data_list_pro_exp, data_list_pro_af
            gc.collect()

        self.data_mol_exp = []
        self.data_mol_af = []
        self.data_pro_exp = []
        self.data_pro_af = []
        for chunk_file in chunk_files:
            data_mol_exp, data_mol_af, data_pro_exp, data_pro_af = torch.load(chunk_file)
            self.data_mol_exp.extend(data_mol_exp)
            self.data_mol_af.extend(data_mol_af)
            self.data_pro_exp.extend(data_pro_exp)
            self.data_pro_af.extend(data_pro_af)

        torch.save((self.data_mol_exp, self.data_mol_af, self.data_pro_exp, self.data_pro_af), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol_exp)
    
    def __getitem__(self, idx):
        return self.data_mol_exp[idx], self.data_mol_af[idx], self.data_pro_exp[idx], self.data_pro_af[idx]

def createGcnDataTwin_muts(args):
    
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

    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
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
    contactPath = '../bigData/contact/{}'.format(mapAfName)
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(afMut, modi, ri, seedi))[0]
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

    return GCNData_mol_exp, GCNData_mol_af, GCNData_pro_exp, GCNData_pro_af


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

    #ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/ensemble/ligand/{}_*.pdb'.format(element1.split('_')[1]))[0]
    #protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdbqt'.format(element1.split('_')[0], element1.split('_')[0]))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_exp = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)
    
    #ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/ensemble/ligand/{}_*.pdb'.format(afKpi))[0]
    #protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    ligand_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/receptor_pdbqt/Abl1_{}_*/model_{}_ptm_{}_{}*.pdbqt'.format(afMut, modi, ri, seedi))[0]

    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_af = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)

    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
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
    contactPath = '../bigData/contact/{}'.format(mapAfName)
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(afMut, modi, ri, seedi))[0]
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

class formDatasetTwin_muts_inter_small(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 element1=None,  element2=None, 
                 mutVecExp=None, mutVecAf=None,
                 smilesExp=None, smilesAf=None, 
                 mutSeqExp=None, mutSeqAf=None,
                 reldock=None, deltay=None, exp=None,
                 blosum62_vectors=None):
        super(formDatasetTwin_muts_inter_small, self).__init__(root)
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

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_muts_inter_small, args=(args,)))
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

def createGcnDataTwin_muts_inter_small(args):
    
    idx, element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp, blosum62_vectors = args

    expKey = element1
    mutExp = element1.split('_')[0]
    afKey = element2
    afMut = afKey.split('_')[0]
    afKpi = afKey.split('_')[1]
    
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

    #ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/ensemble/ligand/{}_*.pdb'.format(element1.split('_')[1]))[0]
    #protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    ligand_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdbqt'.format(element1.split('_')[0], element1.split('_')[0]))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_exp = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)
    
    #ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/ensemble/ligand/{}_*.pdb'.format(afKpi))[0]
    #protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element2.split('_')[0], element2.split('_')[0], element2.split('_')[1]))[0]
    ligand_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.pdbqt'.format(element2.split('_')[0], element2.split('_')[0], element2.split('_')[1]))[0]
    protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdbqt'.format(element1.split('_')[0], element1.split('_')[0]))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, distance_threshold)
    GCNData_inter_af = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)

    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, mutSeqExp, blosum62_vectors)
    GCNData_pro_exp = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro_exp.mutVecExp = torch.LongTensor([mutVecExp])
    GCNData_pro_exp.exp = torch.FloatTensor([float(exp)])
    GCNData_pro_exp.expName = element1
    GCNData_pro_exp.__setitem__('target_size', torch.LongTensor([target_size]))
    
    contactPath = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(afMut))[0]
    pdbFile = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(afMut, afMut))[0]
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


class formDatasetTwin_inter_small_generalNet(InMemoryDataset):
    def __init__(self, root='../', dataset='data_train',
                 nameAll=None, flags=None, 
                 smiles=None, sequence=None, 
                 targetY=None, blosum62_vectors=None):
        super(formDatasetTwin_inter_small_generalNet, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(nameAll, flags, smiles, sequence, targetY, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro, self.data_inter, = torch.load(self.processed_paths[0])

    def process(self, nameAll, flags, smiles, sequence, targetY, blosum62_vectors):
        chunk_size = 400  
        args_list = [(idx, name, flag, smi, seq, yy, blosum62_vectors) 
                     for idx, (name, flag, smi, seq, yy) in enumerate(zip(nameAll, flags, smiles, sequence, targetY))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_inter_small_generalNet, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro = []
            data_list_inter = []

            for res in result:
                GCNData_mol, GCNData_pro, GCNData_inter = res.get()
                if GCNData_inter == None:
                    continue
                data_list_mol.append(GCNData_mol)
                data_list_pro.append(GCNData_pro)
                data_list_inter.append(GCNData_inter)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro, data_list_inter), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro, data_list_inter
            gc.collect()

        self.data_mol = []
        self.data_pro = []
        self.data_inter = []
        for chunk_file in chunk_files:
            data_mol, data_pro, data_inter = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro.extend(data_pro)
            self.data_inter.extend(data_inter)

        torch.save((self.data_mol, self.data_pro, self.data_inter), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_inter[idx]

def createGcnDataTwin_inter_small_generalNet(args):
    idx, name, flag, smiles, sequence, targetY, blosum62_vectors = args
    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smiles))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

    
    if int(flag) == 1:
        #print("processing mutat\n")
        ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*.mol2'.format(name.split('_')[0],name.split('_')[0], name.split('_')[1]))[0]
        protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*pocket.pdbqt'.format(name.split('_')[0], name.split('_')[0]))[0]
        protein_pdb = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]
        distance_threshold = 15
        #print(ligand_pdb, protein_pdbqt)
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                edge_index_intra=edge_index_intra,
                                edge_index_inter=edge_index_inter,
                                pos=pos,
                                split=split)
    elif int(flag) == 2:
        #print("processing wild\n")
        ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/WILD*_{}_*.mol2'.format(name.split('_')[0], name.split('_')[1]))[0]
        protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/WILD_*pocket.pdbqt'.format(name.split('_')[0], name.split('_')[0]))[0]
        protein_pdb = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/WILD_*pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]
        distance_threshold = 15
        #print(ligand_pdb, protein_pdbqt)
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                edge_index_intra=edge_index_intra,
                                edge_index_inter=edge_index_inter,
                                pos=pos,
                                split=split)
    elif int(flag) == 3:
        #print("processing refined\n")
        ligand_pdb = f'/home/data1/BGM/mdrugEffect/bigData/refined_casf/{name}/{name}_ligand.mol2'
        protein_pdbqt = f'/home/data1/BGM/mdrugEffect/bigData/refined_casf/{name}/{name}_pocket.pdbqt'
        protein_pdb = f'/home/data1/BGM/mdrugEffect/bigData/refined_casf/{name}/{name}_pocket.pdb'
        distance_threshold = 15
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                    edge_index_intra=edge_index_intra,
                                    edge_index_inter=edge_index_inter,
                                    pos=pos,
                                    split=split)
    
    else:
        #print("processing refined\n")
        ligand_pdb = f'/home/data1/BGM/mdrugEffect/bigData/refined_set/{name}/{name}_ligand.mol2'
        protein_pdbqt = f'/home/data1/BGM/mdrugEffect/bigData/refined_set/{name}/{name}_pocket.pdbqt'
        protein_pdb = f'/home/data1/BGM/mdrugEffect/bigData/refined_set/{name}/{name}_pocket.pdb'
        distance_threshold = 15
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                    edge_index_intra=edge_index_intra,
                                    edge_index_inter=edge_index_inter,
                                    pos=pos,
                                    split=split)
    
    if int(flag) == 1:
        contactPath = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/contact_{}_*.csv'.format(name.split('_')[0], name.split('_')[0]))[0]
        #print(contactPath)
        pdbFile = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*aligned_pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]

        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    elif int(flag) == 2:
        #contactPath = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/contact_WILD_8dc32.csv'.format( name.split('_')[0]))[0]
        #pdbFile = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/WILD_8dc32_aligned_pocket.pdb'.format( name.split('_')[0]))[0]
        contactPath = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/D276G_ead78/contact_WILD_8dc32.csv')[0]
        pdbFile = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/D276G_ead78/WILD_8dc32_aligned_pocket.pdb')[0]
        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, pdbFile, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    elif int(flag) == 3:
        contactPath = f'/home/data1/BGM/mdrugEffect/bigData/contact_refined_casf/contact_{name}.csv'
        pdbFile = f'/home/data1/BGM/mdrugEffect/bigData/refined_casf/{name}/{name}_pocket_modify.pdb'
        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph_refined(contactPath, pdbFile, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    
    else:
        contactPath = f'/home/data1/BGM/mdrugEffect/bigData/contact_refined/contact_{name}.csv'
        pdbFile = f'/home/data1/BGM/mdrugEffect/bigData/refined_set/{name}/{name}_pocket_modify.pdb'
        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph_refined(contactPath, pdbFile, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    
    
    return GCNData_mol, GCNData_pro, GCNData_inter

class formDatasetTwin_inter_small_generalNet_pred(InMemoryDataset):
    def __init__(self, root='../', dataset='data_pred',
                 predProName=None, predLigName=None, 
                 smiles=None, sequence=None, 
                 targetY=None,blosum62_vectors=None):
        super(formDatasetTwin_inter_small_generalNet_pred, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(predProName, predLigName, smiles, sequence, targetY, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro, self.data_inter, = torch.load(self.processed_paths[0])

    def process(self, predProName, predLigName, smiles, sequence, targetY, blosum62_vectors):
        chunk_size = 400  
        args_list = [(idx, proName, ligName, smi, seq, blosum62_vectors) 
                     for idx, (proName, ligName, smi, seq) in enumerate(zip(predProName, predLigName, smiles, sequence, targetY))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_inter_small_generalNet_pred, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro = []
            data_list_inter = []

            for res in result:
                GCNData_mol, GCNData_pro, GCNData_inter = res.get()
                if GCNData_inter == None:
                    continue
                data_list_mol.append(GCNData_mol)
                data_list_pro.append(GCNData_pro)
                data_list_inter.append(GCNData_inter)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro, data_list_inter), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro, data_list_inter
            gc.collect()

        self.data_mol = []
        self.data_pro = []
        self.data_inter = []
        for chunk_file in chunk_files:
            data_mol, data_pro, data_inter = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro.extend(data_pro)
            self.data_inter.extend(data_inter)

        torch.save((self.data_mol, self.data_pro, self.data_inter), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_inter[idx]

def createGcnDataTwin_inter_small_generalNet_pred(args):
    idx, predProName, predLigName, smiles, sequence, targetY, blosum62_vectors = args
    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smiles))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
    GCNData_mol.ligName = predLigName
    
 

    contactPath = f'/home/data1/BGM/mdrugEffect/script_big/refined_preds/contact_{predProName}.csv'
    pdbFile = f'/home/data1/BGM/mdrugEffect/script_big/refined_preds/{predProName}.pdb'
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph_refined(contactPath, pdbFile, sequence, blosum62_vectors)
    GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro.proName = predProName
    GCNData_pro.targety = torch.FloatTensor([float(targetY)])
    GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

    ligand_pdb = f'/home/data1/BGM/mdrugEffect/script_big/refined_preds/{predLigName}.pdb'
    protein_pdbqt = f'/home/data1/BGM/mdrugEffect/script_big/refined_preds/{predProName}_pocket.pdbqt'
    protein_pdb = f'/home/data1/BGM/mdrugEffect/script_big/refined_preds/{predProName}_pocket.pdbqt'

    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
    if x == None or edge_index_intra == None or edge_index_inter == None:
        GCNData_inter = None
    else:
        GCNData_inter = Data(x=x, 
                                    edge_index_intra=edge_index_intra,
                                    edge_index_inter=edge_index_inter,
                                    pos=pos,
                                    split=split)
    
    
    return GCNData_mol, GCNData_pro, GCNData_inter

class formDatasetTwin_inter_small_coreset_pred(InMemoryDataset):
    def __init__(self, root='../', dataset='data_coreset',
                 predProName=None, predLigName=None, 
                 smiles=None, sequence=None, 
                 targetY=None, blosum62_vectors=None):
        super(formDatasetTwin_inter_small_coreset_pred, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(predProName, predLigName, smiles, sequence, targetY, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro, self.data_inter, = torch.load(self.processed_paths[0])

    def process(self, predProName, predLigName, smiles, sequence, targetY, blosum62_vectors):
        chunk_size = 400  
        args_list = [(idx, proName, ligName, smi, seq, yy, blosum62_vectors) 
                     for idx, (proName, ligName, smi, seq, yy) in enumerate(zip(predProName, predLigName, smiles, sequence, targetY))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_inter_small_coreset_pred, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro = []
            data_list_inter = []

            for res in result:
                GCNData_mol, GCNData_pro, GCNData_inter = res.get()
                if GCNData_inter == None:
                    continue
                data_list_mol.append(GCNData_mol)
                data_list_pro.append(GCNData_pro)
                data_list_inter.append(GCNData_inter)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro, data_list_inter), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro, data_list_inter
            gc.collect()

        self.data_mol = []
        self.data_pro = []
        self.data_inter = []
        for chunk_file in chunk_files:
            data_mol, data_pro, data_inter = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro.extend(data_pro)
            self.data_inter.extend(data_inter)

        torch.save((self.data_mol, self.data_pro, self.data_inter), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_inter[idx]

def createGcnDataTwin_inter_small_coreset_pred(args):
    idx, predProName, predLigName, smiles, sequence, targetY, blosum62_vectors = args
    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smiles))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
    GCNData_mol.ligName = predLigName
    
 

    contactPath = f'/home/data1/BGM/mdrugEffect/bigData/contact_coreset/contact_{predProName}.csv'
    pdbFile = f'/home/data1/BGM/mdrugEffect/bigData/coreset/{predProName}/{predProName}_pocket.pdb'
    target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph_refined(contactPath, pdbFile, sequence, blosum62_vectors)
    GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                        edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                        edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                        pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
    GCNData_pro.name = predProName
    GCNData_pro.targety = torch.FloatTensor([float(targetY)])
    GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

    ligand_pdb = f'/home/data1/BGM/mdrugEffect/bigData/coreset/{predProName}/{predProName}_ligand.mol2'
    protein_pdbqt = f'/home/data1/BGM/mdrugEffect/bigData/coreset/{predProName}/{predProName}_pocket.pdbqt'
    protein_pdb = f'/home/data1/BGM/mdrugEffect/bigData/coreset/{predProName}/{predProName}_pocket.pdb'

    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
    if x == None or edge_index_intra == None or edge_index_inter == None:
        GCNData_inter = None
    else:
        GCNData_inter = Data(x=x, 
                                    edge_index_intra=edge_index_intra,
                                    edge_index_inter=edge_index_inter,
                                    pos=pos,
                                    split=split)
    
    
    return GCNData_mol, GCNData_pro, GCNData_inter


class formDatasetTwin_inter_small_CASP(InMemoryDataset):
    def __init__(self, root='../', dataset='data_casp',
                 nameAll=None, flags=None, 
                 smiles=None, sequence=None, 
                 targetY=None, blosum62_vectors=None):
        super(formDatasetTwin_inter_small_CASP, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.exists(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.load_data()
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(nameAll, flags, smiles, sequence, targetY, blosum62_vectors)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def load_data(self):
        self.data_mol, self.data_pro, self.data_inter, = torch.load(self.processed_paths[0])

    def process(self, nameAll, flags, smiles, sequence, targetY, blosum62_vectors):
        chunk_size = 400  
        args_list = [(idx, name, flag, smi, seq, yy, blosum62_vectors) 
                     for idx, (name, flag, smi, seq, yy) in enumerate(zip(nameAll, flags, smiles, sequence, targetY))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(13)
            result = []
            for args in chunk_args:
                result.append(pool.apply_async(func=createGcnDataTwin_inter_small_CASP, args=(args,)))
            pool.close()
            pool.join()

            data_list_mol = []
            data_list_pro = []
            data_list_inter = []

            for res in result:
                GCNData_mol, GCNData_pro, GCNData_inter = res.get()
                if GCNData_inter == None:
                    continue
                data_list_mol.append(GCNData_mol)
                data_list_pro.append(GCNData_pro)
                data_list_inter.append(GCNData_inter)

            chunk_file = f"{self.processed_paths[0]}_chunk_{i // chunk_size}.pt"
            torch.save((data_list_mol, data_list_pro, data_list_inter), chunk_file)
            chunk_files.append(chunk_file)
            print(f'Chunk {i // chunk_size} saved to {chunk_file}')

            del data_list_mol, data_list_pro, data_list_inter
            gc.collect()

        self.data_mol = []
        self.data_pro = []
        self.data_inter = []
        for chunk_file in chunk_files:
            data_mol, data_pro, data_inter = torch.load(chunk_file)
            self.data_mol.extend(data_mol)
            self.data_pro.extend(data_pro)
            self.data_inter.extend(data_inter)

        torch.save((self.data_mol, self.data_pro, self.data_inter), self.processed_paths[0])
        print('All chunks merged. Graph construction done. Saving to file.')

        for chunk_file in chunk_files:
            os.remove(chunk_file)
            print(f'Chunk file {chunk_file} deleted.')
        print('\n')
    
    def __len__(self):
        return len(self.data_mol)
    
    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_inter[idx]

def createGcnDataTwin_inter_small_CASP(args):
    idx, name, flag, smiles, sequence, targetY, blosum62_vectors = args
    node_f, edge_index, edge_attr, bonds, c_size = mol2vec(Chem.MolFromSmiles(smiles))
    for bond in bonds: 
        edge_attr.append(bond_features(bond))   
    GCNData_mol = Data(x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

    
    if int(flag) == 0:
        ligand_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/ligand_{}.pdbqt'.format(name.split('_')[0],name.split('_')[0]))[0]
        protein_pdbqt = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdbqt'.format(name.split('_')[0], name.split('_')[0]))[0]
        protein_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]
        distance_threshold = 15
        #print(ligand_pdb, protein_pdbqt)
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                edge_index_intra=edge_index_intra,
                                edge_index_inter=edge_index_inter,
                                pos=pos,
                                split=split)
    elif int(flag) == 1:
        ligand_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/ligand_{}_vina_output.pdbqt'.format(name.split('_')[0],name.split('_')[0]))[0]
        protein_pdbqt = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdbqt'.format(name.split('_')[0], name.split('_')[0]))[0]
        protein_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]
        distance_threshold = 15
        #print(ligand_pdb, protein_pdbqt)
        x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
        if x == None or edge_index_intra == None or edge_index_inter == None:
            GCNData_inter = None
        else:
            GCNData_inter = Data(x=x, 
                                edge_index_intra=edge_index_intra,
                                edge_index_inter=edge_index_inter,
                                pos=pos,
                                split=split)
    
    if int(flag) == 0:
        contactPath = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/contact_{}.csv'.format(name.split('_')[0], name.split('_')[0]))[0]
        #print(contactPath)
        protein_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]

        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, protein_pdb, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    elif int(flag) == 1:
        contactPath = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/contact_{}.csv'.format(name.split('_')[0], name.split('_')[0]))[0]
        protein_pdb = glob.glob('/home/data1/BGM/mdrugEffect/script_big/CASP16/CASP_prepared/{}/{}_aligned_pocket.pdb'.format(name.split('_')[0], name.split('_')[0]))[0]
        target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix = proteinSeq_to_graph(contactPath, protein_pdb, sequence, blosum62_vectors)
        GCNData_pro = Data(x=torch.tensor(target_feature, dtype=torch.float),
                            edge_index=torch.tensor(target_edge_index, dtype=torch.long).transpose(1, 0),
                            edge_weight=torch.tensor(target_edge_weight, dtype=torch.float),
                            pos_matrix=torch.tensor(pos_matrix, dtype=torch.float))
        GCNData_pro.name = name
        GCNData_pro.targety = torch.FloatTensor([float(targetY)])
        GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
    
    
    return GCNData_mol, GCNData_pro, GCNData_inter

def align_sequence(proSeq, ref_seq):
    align = pairwise2.align.globalxx(ref_seq, proSeq)
    return align

def save_alignment_to_file(alignment, output_file, sequence):
    maxLen = 0
    with open(output_file, 'w') as f:
        for align in alignment:
            for aligned_seq in align:
                if isinstance(aligned_seq, str):
                    if len(aligned_seq.splitlines()[0] ) <= (len(sequence)+2):
                        # seq_id = ">" +  str(len(sequence) + 1)
                        aligned_seq_str = aligned_seq.splitlines()[0] 
                        # f.write(seq_id + "\n")
                        f.write(aligned_seq_str + "\n")

def get_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filename)
    return files

def SdfToSMILES(input_file):
    suppl = Chem.SDMolSupplier(input_file)

    smiles_list = []

    for mol in suppl:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
    return smiles_list[0]

def ergoidcSdf():
    directoryPath = '../data/ligand/sdf'
    fileDic = get_files_in_directory(directoryPath)
    smilesDic = {'pdb_id': [], 'smiles': []}
    for file in fileDic:
        name = file.split('_')[0]
        ligandPath = '{}/{}'.format(directoryPath, file)
        smiles = SdfToSMILES(ligandPath)
        smilesDic['pdb_id'].append(name)
        smilesDic['smiles'].append(smiles)
    df = pd.DataFrame(smilesDic)
    df.to_csv('../data/ligand_smiles.csv', index=False)
    print('../data/ligand_smiles.csv finished!')

def getMutationFile(filePath):
    fileDic = {}
    df = pd.read_csv(filePath)['mutation'].tolist()
    for item in df:
        name = item.split('_')[0]
        fileDic[name] = item
    return fileDic

def calContactMap(mutationName):
    distances_array = np.zeros((len(mutationName), 287, 287))
    for idx, af in enumerate(mutationName):
        data = []
        with open(glob.glob('../../mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mutationName))[0], 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data.append(row)
        distances_array[idx] = np.array(data)
    return distances_array
        # df = pd.DataFrame(distances1)
        # df.to_csv('../data/pocket/contact/contact_'+str(af_id)+'.csv', index=False, header=False)

def getStrain(mutation, strainPath):
    df = pd.read_csv('{}/{}_strain.csv'.format(strainPath, mutation))
    column2 = df['effectiveStrain'].tolist()
    epsilon = 1e-10
    strain_array = np.array(column2)
    log_strain_array = np.log(np.where(strain_array > 0, strain_array, epsilon))
    log_stain_list = log_strain_array.tolist()
    return log_stain_list

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

def flatteningYPredResult(data):
    dataNew = []
    for tensor_item in data:
        if tensor_item.dim() == 0:
            dataNew.append(tensor_item.item())
            continue
        for num in tensor_item:
            dataNew.append(num.item())
    return dataNew

def xlsxTocsv(file):
    df = pd.read_excel(file)
    mutationSeq = {'mutationName': [], 'sequence': []}
    for name, sequence in zip(df['sequence'].tolist(), df['229-515'].tolist()):
        if name == 'original':
            name = 'wild'
        mutationSeq['mutationName'].append(name)
        mutationSeq['sequence'].append(sequence)
    df_result = pd.DataFrame(mutationSeq)
    df_result.to_csv('../data/mutationSeq.csv', index=False)

def calculate_metrics(yTrue, yPred):
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    rmse = np.sqrt(np.mean((yTrue - yPred)**2))
    mae = np.mean(np.abs(yTrue - yPred))
    if yTrue.shape[0] < 2 and yPred.shape[0] < 2:
        r_value = 0
    else:
        r_value, _ = pearsonr(yTrue, yPred)
    return rmse, mae, r_value

def writeMetrics(metrics, savePath):
    df = pd.DataFrame([metrics], columns=metrics.keys())
    print(df)
    df.to_csv(savePath, index=False)
    print(f"Metrics have been written to {savePath}.")

def writeLoss(trainLoss, savePath):
    df = pd.DataFrame(trainLoss)
    df.to_csv(savePath, index=False)
    print(f"Loss have been written to {savePath}.")

def writeRow(data, savePath):
    rows = [(key, value) for key, value in data.items()]
    with open(savePath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value']) 
        for row in rows:
            writer.writerow(row) 
    print(f"metrics have been written to {savePath}.")

def writeMiddleResult(data, savePath):
    with open(savePath, 'w') as f:
        for tensor in data:
            tensor_str = ' '.join(map(str, tensor.tolist()))
            f.write(tensor_str + '\n')
    print(f"middle results have been written to {savePath}.")


def createFeatureToCsv(mutationName):
    trainData = formDataset(root='../data/multiPropertyPre/{}'.format(mutationName), dataset='data_train')
    testData = formDataset(root='../data/multiPropertyPre/{}'.format(mutationName), dataset='data_test')

    wildSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].wildSeq.view(-1))))
    mutationSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].muSeq.view(-1))))
    contactMapArray = np.empty((len(trainData)+len(testData), 411845))
    strainArray = np.empty((len(trainData)+len(testData), len(trainData[0].strain.view(-1))))
    edgeFeatureArray = np.empty((len(trainData)+len(testData), 3042))
    yResultArray = np.empty(len(trainData)+len(testData))
    
    trainNum = 0
    for batch_idx, data in enumerate(trainData):
        wildSeqArray[batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[batch_idx] = data.muSeq.view(-1).numpy()
        iterName = data.muName
        contactMapArray[batch_idx] = calContactMap(iterName).flatten()
        strainArray[batch_idx] = data.strain.view(-1).numpy()
        edgeFeatureLen = len(data.x.view(-1))
        edgeFeatureArray[batch_idx, :edgeFeatureLen] = data.x.view(-1).numpy()
        yResultArray[batch_idx] = data.y.item()
        trainNum = trainNum + 1
    
    testNum = 0
    curr_idx = trainNum
    for batch_idx, data in enumerate(testData):
        wildSeqArray[curr_idx + batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[curr_idx + batch_idx] = data.muSeq.view(-1).numpy()
        iterName = data.muName
        contactMapArray[curr_idx + batch_idx] = calContactMap(iterName).flatten()
        strainArray[curr_idx + batch_idx] = data.strain.view(-1).numpy()
        edgeFeatureLen = len(data.x.view(-1))
        edgeFeatureArray[curr_idx + batch_idx, :edgeFeatureLen] = data.x.view(-1).numpy()
        yResultArray[curr_idx + batch_idx] = data.y.item()
        testNum = testNum + 1
    
    X = np.column_stack((wildSeqArray, mutationSeqArray, contactMapArray, strainArray, edgeFeatureArray))
    Y = yResultArray
    return X, Y, trainNum, testNum

def createFeatureToCsv02(mutationName):
    # trainData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_train')
    # testData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_test')
    trainData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_train')
    testData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_test')

    wildSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].wildSeq.view(-1))))
    mutationSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].muSeq.view(-1))))
    contactMapArray = np.empty((len(trainData)+len(testData), 411845))
    strainArray = np.empty((len(trainData)+len(testData), len(trainData[0].strain.view(-1))))
    ecfpArray = np.empty((len(trainData)+len(testData), len(trainData[0].ep.view(-1))))
    yResultArray = np.empty(len(trainData)+len(testData))
    
    trainNum = 0
    for batch_idx, data in enumerate(trainData):
        wildSeqArray[batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[batch_idx] = data.muSeq.view(-1).numpy()
        iterName = data.muName
        contactMapArray[batch_idx] = calContactMap(iterName).flatten()
        strainArray[batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[batch_idx] = data.ep.view(-1).numpy()
        yResultArray[batch_idx] = data.y.item()
        trainNum = trainNum + 1
    
    testNum = 0
    curr_idx = trainNum
    for batch_idx, data in enumerate(testData):
        wildSeqArray[curr_idx + batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[curr_idx + batch_idx] = data.muSeq.view(-1).numpy()
        iterName = data.muName
        contactMapArray[curr_idx + batch_idx] = calContactMap(iterName).flatten()
        strainArray[curr_idx + batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[curr_idx + batch_idx] = data.ep.view(-1).numpy()
        yResultArray[curr_idx + batch_idx] = data.y.item()
        testNum = testNum + 1
    
    X = np.column_stack((wildSeqArray, mutationSeqArray, contactMapArray, strainArray, ecfpArray))
    Y = yResultArray
    return X, Y, trainNum, testNum


def createFeatureToCsv03(mutationName):
    # trainData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_train')
    # testData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_test')
    trainData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_train')
    testData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_test')

    wildSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].wildSeq.view(-1))))
    mutationSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].muSeq.view(-1))))
    # contactMapArray = np.empty((len(trainData)+len(testData), 411845))
    strainArray = np.empty((len(trainData)+len(testData), len(trainData[0].strain.view(-1))))
    ecfpArray = np.empty((len(trainData)+len(testData), len(trainData[0].ep.view(-1))))
    yResultArray = np.empty(len(trainData)+len(testData))
    
    trainNum = 0
    for batch_idx, data in enumerate(trainData):
        wildSeqArray[batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[batch_idx] = data.muSeq.view(-1).numpy()
        # iterName = data.muName
        # contactMapArray[batch_idx] = calContactMap(iterName).flatten()
        strainArray[batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[batch_idx] = data.ep.view(-1).numpy()
        yResultArray[batch_idx] = data.y.item()
        trainNum = trainNum + 1
    
    testNum = 0
    curr_idx = trainNum
    for batch_idx, data in enumerate(testData):
        wildSeqArray[curr_idx + batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[curr_idx + batch_idx] = data.muSeq.view(-1).numpy()
        # iterName = data.muName
        # contactMapArray[curr_idx + batch_idx] = calContactMap(iterName).flatten()
        strainArray[curr_idx + batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[curr_idx + batch_idx] = data.ep.view(-1).numpy()
        yResultArray[curr_idx + batch_idx] = data.y.item()
        testNum = testNum + 1
    
    X = np.column_stack((wildSeqArray, mutationSeqArray, strainArray, ecfpArray))
    Y = yResultArray
    return X, Y, trainNum, testNum

def createFeatureToCsv04(mutationName):
    # trainData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_train')
    # testData = formDataset02(root='../data/multiPropertyPre02/{}'.format(mutationName), dataset='data_test')
    trainData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_train')
    testData = formDataset02(root='../data/multiPropertyPreAll/{}'.format(mutationName), dataset='data_test')

    wildSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].wildSeq.view(-1))))
    mutationSeqArray = np.empty((len(trainData)+len(testData), len(trainData[0].muSeq.view(-1))))
    # contactMapArray = np.empty((len(trainData)+len(testData), 411845))
    # strainArray = np.empty((len(trainData)+len(testData), len(trainData[0].strain.view(-1))))
    ecfpArray = np.empty((len(trainData)+len(testData), len(trainData[0].ep.view(-1))))
    yResultArray = np.empty(len(trainData)+len(testData))
    
    trainNum = 0
    for batch_idx, data in enumerate(trainData):
        wildSeqArray[batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[batch_idx] = data.muSeq.view(-1).numpy()
        # iterName = data.muName
        # contactMapArray[batch_idx] = calContactMap(iterName).flatten()
        # strainArray[batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[batch_idx] = data.ep.view(-1).numpy()
        yResultArray[batch_idx] = data.y.item()
        trainNum = trainNum + 1
    
    testNum = 0
    curr_idx = trainNum
    for batch_idx, data in enumerate(testData):
        wildSeqArray[curr_idx + batch_idx] = data.wildSeq.view(-1).numpy()
        mutationSeqArray[curr_idx + batch_idx] = data.muSeq.view(-1).numpy()
        # iterName = data.muName
        # contactMapArray[curr_idx + batch_idx] = calContactMap(iterName).flatten()
        # strainArray[curr_idx + batch_idx] = data.strain.view(-1).numpy()
        ecfpArray[curr_idx + batch_idx] = data.ep.view(-1).numpy()
        yResultArray[curr_idx + batch_idx] = data.y.item()
        testNum = testNum + 1
    
    X = np.column_stack((wildSeqArray, mutationSeqArray, ecfpArray))
    Y = yResultArray
    return X, Y, trainNum, testNum

def gene_ECFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    radius = 2
    nBits = 1024
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp

def initializationModel(model):
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Elastic': ElasticNet(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoost': GradientBoostingRegressor(),
        'SVR': SVR(),
        'ANN': MLPRegressor(),
        'KNN': KNeighborsRegressor()
    }
    return models[model]


# if __name__ == '__main__':
#     file = '/home/data1/BGM/mutationEffect/data/abl_sequence.xlsx'
#     xlsxTocsv(file)
        
