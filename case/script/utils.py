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
#from torch.utils.data import DataLoader, TensorDataset

mp.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")


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

        chunk_size = 512
        args_list = [(idx, ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex, blosum62_vectors) 
                     for idx, (ele1, ele2, mvecExp, mvecAf, smiExp, smiAf, mSeqExp, mSeqAf, reDock, delta, ex) in enumerate(zip(element1, element2, mutVecExp, mutVecAf, smilesExp, smilesAf, mutSeqExp, mutSeqAf, reldock, deltay, exp))]

        chunk_files = [] 

        for i in range(0, len(args_list), chunk_size):
            chunk_args = args_list[i:i + chunk_size]

            pool = Pool(12)
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
            torch.cuda.empty_cache()

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
        gc.collect()
        torch.cuda.empty_cache()
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

    # EXP featurizer
    ligand_pdb = glob.glob('../data/exp/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.mol2'.format(element1.split('_')[0], element1.split('_')[0], element1.split('_')[1]))[0]
    protein_pdbqt = glob.glob('../data/exp/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdbqt'.format(element1.split('_')[0], element1.split('_')[0]))[0]
    protein_pdb = glob.glob('../data/exp/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdb'.format(element1.split('_')[0], element1.split('_')[0]))[0]
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
    GCNData_inter_exp = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)
    
    #ligand_pdb = glob.glob('/home/data1/BGM/mutationEffect/ensemble/ligand/{}_*.pdb'.format(afKpi))[0]
    #protein_pdbqt = glob.glob('/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    # AF2 featurizer
    ligand_pdbqt = glob.glob('../data/af/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    ligand_pdb = glob.glob('../data/af/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out_temp.pdb'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
    protein_pdbqt = glob.glob('../data/af/data_raw/unzip/Abl1_{}_*256_512_32/pdb/model_{}_ptm_{}_{}_{}_pocket.pdbqt'.format(afMut, modi, ri, seedi, afKpi))[0]
    protein_pdb = glob.glob('../data/af/data_raw/unzip/Abl1_{}_*256_512_32/pdb/model_{}_ptm_{}_{}_{}_pocket.pdb'.format(afMut, modi, ri, seedi, afKpi))[0]

    # build both AF and exp features
    distance_threshold = 15
    x, edge_index_intra, edge_index_inter, pos, split = pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold)
    GCNData_inter_af = Data(x=x, 
                             edge_index_intra=edge_index_intra,
                             edge_index_inter=edge_index_inter,
                             pos=pos,
                             split=split)

    contactPath = glob.glob('../data/exp/pocket/contact/contact_{}*.csv'.format(mutExp))[0]
    pdbFile = glob.glob('../data/exp/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mutExp, mutExp))[0]
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
    contactPath = '../data/af/contact/{}'.format(mapAfName)
    pdbFile = glob.glob('../data/af/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(afMut, modi, ri, seedi))[0]
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



def gene_ECFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    radius = 2
    nBits = 1024
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp

