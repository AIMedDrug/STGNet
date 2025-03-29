from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))
 
 
def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))
 
 
def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths"""
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
 
  return intervals
 
 
def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found"""
  try:
    return l.index(e) # return the index of element in list
  except:
    return len(l)

possible_atom_list = [
    #'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    #'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Br', 'Fe', 'Ca', 'Cu', 'Pd', 'Al', 'H'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']
 
reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list]
 
intervals = get_intervals(reference_lists)


def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list, atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  #features[6] = safe_index(possible_chirality_list, atom.)
  return features
 
 
def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals"""
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]
 
  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id
 
 
def id_to_features(id, intervals):
  features = 6 * [0]
 
  # Correct for null
  id -= 1
 
  for k in range(0, 6 - 1):
  # print(6-k-1, id)
    features[6 - k - 1] = id // intervals[6 - k - 1]
    id -= features[6 - k - 1] * intervals[6 - k - 1]
  # Correct for last one
  features[0] = id
  return features
 
 
def atom_to_id(atom):
  """Return a unique id corresponding to the atom type"""
  features = get_feature_list(atom)
  return features_to_id(features, intervals)
 
 
def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
  results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C','N', 'O','S','F','Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As', 'Al','I','B','V', 'K','Tl','Yb','Sb',
        'Sn','Ag', 'Pd','Co', 'Se', 'Ti','Zn','H', 'Li', 'Ge', 'Cu', 'Au','Ni','Cd','In','Mn', 'Zr','Cr','Pt','Hg', 'Pb', 'Unknown'
        #'C','N', 'O','S','F','P','Cl','Br','Mg', 'Ca','Fe','As', 'Al','H', 'Zn', 'Pt', 'Unknown',
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
  if not explicit_H:
      #results = results + list(atom.GetTotalNumHs())
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
  if use_chirality:
    try:
      results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
      results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
 
  return np.array(results)
 
def bond_features(bond, use_chirality=False):
  from rdkit import Chem
  bt = bond.GetBondType()
  bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
    ]
  if use_chirality:
    bond_feats = bond_feats + one_of_k_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
  return np.array(bond_feats)
 

def get_bond_pair(mol):
  bonds = mol.GetBonds()
  res = [[],[]]
  for bond in bonds:
    res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
  return res
 
def mol2vec(mol):
  atoms = mol.GetAtoms()
  bonds = mol.GetBonds()
  node_f= [atom_features(atom) for atom in atoms] #array for each atom in molecule
  edge_index = get_bond_pair(mol)
  edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
  c_size = mol.GetNumAtoms()
  # edge_attr = [] 
  # for bond in bonds:
  #   edge_attr.append(bond_features(bond, use_chirality=False))
  return node_f, edge_index, edge_attr, bonds, c_size

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

def getBlosum62_vectors():
    blosum62 = substitution_matrices.load("BLOSUM62")

    amino_acids = blosum62.alphabet

    blosum62_vectors = {aa: np.zeros(23) for aa in amino_acids}

    for i, aa1 in enumerate(amino_acids):
        if i >=23: 
            break
        for j, aa2 in enumerate(amino_acids):
            if j>=23:
                break
            if i < len(amino_acids) and j < len(amino_acids):
    
                blosum62_vectors[aa1][j] = blosum62[i, j]
                blosum62_vectors[aa2][i] = blosum62[j, i]
    return blosum62_vectors

def PSSM_calculation(aln_file, pro_seq):
    
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)+2))
    with open(aln_file, 'r') as f:
        lines = f.readlines()
        line_count = len(lines)
        for line in lines:
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    pssmDic = {}
    for re in pro_res_table:
        pssmDic[re] = pssm_mat[pro_res_table.index(re)]
    return pssmDic


def target_to_feature(pro_seq, blosum62_vectors):
  pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
  pro_property = np.zeros((len(pro_seq), 12))
  pro_blosum62 = np.zeros((len(pro_seq), 23))
  pro_pssm = np.zeros((len(pro_seq), len(pro_seq)+2))

  # pssmDic = PSSM_calculation(alignment_file, pro_seq)
  for i in range(len(pro_seq)):
    pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
    pro_property[i,] = residue_features(pro_seq[i])
    pro_blosum62[i,] = np.array(blosum62_vectors[pro_seq[i]])
    # pro_pssm[i,] = pssmDic[pro_seq[i]]
  return np.concatenate((pro_hot, pro_property, pro_blosum62), axis=1)


def extract_protein_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)  
    structure = parser.get_structure('protein', pdb_file)

    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  
                    residue_coords = []
                    for atom in residue:
                        residue_coords.append(atom.coord)
                    residue_coords = np.array(residue_coords)
                    coordinates.append(residue_coords)

    return coordinates

def proteinSeq_to_graph(contactPath, pdbFile, proSeq, blosum62_vectors):
  target_size = len(proSeq)
  target_edge_index = []
  target_edge_weight = []
  contact_map = pd.read_csv(contactPath, header=None).values
  contact_map = np.add(np.eye(contact_map.shape[0]), contact_map)
  index_row, index_col = np.where((contact_map > 0) & (contact_map <= 0.5))
  # blosum62_vectors = getBlosum62_vectors()
  for i, j in zip(index_row, index_col):
    target_edge_index.append([i, j])
    target_edge_weight.append(contact_map[i, j])
    target_feature = target_to_feature(proSeq, blosum62_vectors)
  target_edge_index = np.array(target_edge_index)
  coordinates = extract_protein_coordinates(pdbFile)
  pos_matrix = np.array([coords[0] for coords in coordinates])
  return target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix

def proteinSeq_to_graph_refined(contactPath, pdbFile, proSeq, blosum62_vectors):
  target_size = len(proSeq)
  target_edge_index = []
  target_edge_weight = []
  contact_map = pd.read_csv(contactPath, header=None).values
  contact_map = np.add(np.eye(contact_map.shape[0]), contact_map)
  index_row, index_col = np.where((contact_map > 0) & (contact_map <= 0.5))
  target_feature = target_to_feature(proSeq, blosum62_vectors)
  # blosum62_vectors = getBlosum62_vectors()
  for i, j in zip(index_row, index_col):
    if i < target_feature.shape[0] and j < target_feature.shape[0]:
      target_edge_index.append([i, j])
      target_edge_weight.append(contact_map[i, j])
  target_edge_index = np.array(target_edge_index)
  coordinates = extract_protein_coordinates(pdbFile)
  pos_matrix = np.array([coords[0] for coords in coordinates])
  return target_size, target_feature, target_edge_index, target_edge_weight, pos_matrix


def getProteinGraphBatch(protein_cSize, protein_feature, device):
  num_graphs = len(protein_cSize)
  num_nodes = protein_feature.shape[0] 
  protein_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
  cumsum = torch.cat([torch.tensor([0], device=device), protein_cSize.cumsum(dim=0)])
  for i in range(num_graphs):
    start_idx = int(cumsum[i])
    end_idx = int(cumsum[i + 1])
    protein_batch[start_idx:end_idx] = i
  return protein_batch
