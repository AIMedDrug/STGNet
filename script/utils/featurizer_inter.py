from Bio.PDB import PDBParser
import numpy as np
import torch
import networkx as nx
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from torch_geometric.data import Data
import glob
import csv
import os
from scipy.spatial import distance_matrix
import subprocess

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def read_ligandMol2_file(file_path):
    mol = Chem.MolFromMol2File(file_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read PDB file: {file_path}")
    return mol


def atom_features(mol, graph, explicit_H=True):
    # atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_symbols = [
        'C','N', 'O','S','F','Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As', 'Al','I','B','V', 'K','Tl','Yb','Sb',
        'Sn','Ag', 'Pd','Co', 'Se', 'Ti','Zn','H', 'Li', 'Ge', 'Cu', 'Au','Ni','Cd','In','Mn', 'Zr','Cr','Pt','Hg', 'Pb', 'Unknown'
      ]
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    edge_set = set()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if (i, j) not in edge_set and (j, i) not in edge_set:  # 检查边是否已经添加
            graph.add_edge(i, j)
            edge_set.add((i, j))

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)
    #graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    x = x[:, :64]
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    return x, edge_index

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

def read_ligandPdb_file(file_path):
    mol = Chem.MolFromPDBFile(file_path, removeHs=False) 
    if mol is None:
        raise ValueError(f"Could not read PDB file: {file_path}")
    return mol
def convert_pdbqt_to_pdb(pdbqt_file, pdb_file):
    with open(pdbqt_file, 'r') as infile, open(pdb_file, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                pdb_line = line[:66].strip() + '\n'
                if ' CG ' in line:
                    line = line.replace(' CG ', ' C  ')
                outfile.write(pdb_line)
            else:
                outfile.write(line)

def read_ligandPdbqt_file(file_path):
    temp_pdb_file = file_path.replace(".pdbqt", "_temp.pdb")
    convert_pdbqt_to_pdb(file_path, temp_pdb_file)
    
    mol = Chem.MolFromPDBFile(temp_pdb_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read PDBQT file: {file_path}")
    return mol

def read_proteinPdb_file(filePath):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", filePath)
        # print("The PDB file format is correct")
    except Exception as e:
        print(f"Could not read PDB file: {e}")
    
    atom_objects = []
    atom_positions = []
    atom_indices = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:  
                    atom_objects.append(atom)
                    atom_positions.append(atom.coord)
                    atom_indices.append(len(atom_indices)) 

    atom_positions = np.array(atom_positions)

    return structure, atom_objects, atom_positions, atom_indices

def read_pdbqt_file(filePath):
    atom_objects = []
    atom_positions = []
    atom_indices = []
    
    with open(filePath, 'r') as file:
        lines = file.readlines()
    
    model_found = False
    for line in lines:
        if line.startswith("MODEL"):

            if model_found:
                break
            model_found = True

        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_info = {
                "type": line[0:6].strip(),
                "atom_number": int(line[6:11].strip()),
                "atom_name": line[12:16].strip(),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "residue_number": int(line[22:26].strip()),
                "x": float(line[30:38].strip()),
                "y": float(line[38:46].strip()),
                "z": float(line[46:54].strip()),
                "element": line[76:78].strip(),
                "charge": line[78:80].strip() if len(line) > 78 else None
            }

            atom_objects.append(atom_info)
            atom_positions.append([atom_info["x"], atom_info["y"], atom_info["z"]])
            atom_indices.append(len(atom_indices)) 
    
    atom_positions = np.array(atom_positions)
    return atom_objects, atom_positions, atom_indices

def protein_atom_features_af(atom_objects, close_protein_indices, explicit_H=True):
    atom_features = []
    # atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_symbols = [
        'C','N', 'O','S','F','Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As', 'Al','I','B','V', 'K','Tl','Yb','Sb',
        'Sn','Ag', 'Pd','Co', 'Se', 'Ti','Zn','H', 'Li', 'Ge', 'Cu', 'Au','Ni','Cd','In','Mn', 'Zr','Cr','Pt','Hg', 'Pb', 'Unknown'
      ]
    for idx in close_protein_indices:
        atom = atom_objects[idx]
        atom_symbol = atom["element"] 
        residue_name = atom.get("residue_name", "Unknown") 
        degree = len([a for a in atom.get_parent().get_atoms() if a != atom]) if hasattr(atom, 'get_parent') else 0 
        hybridization = 'Unknown' 
        is_aromatic = False  
        results = one_of_k_encoding_unk(atom_symbol, atom_symbols + ['Unknown']) + \
                  one_of_k_encoding_unk(degree, [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(hybridization, ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Unknown']) + \
                  [is_aromatic]
        if explicit_H:
            num_hydrogens = atom.get_total_num_hydrogens() if hasattr(atom, 'get_total_num_hydrogens') else 0
            results += one_of_k_encoding_unk(num_hydrogens, [0, 1, 2, 3, 4])

        atom_features.append(np.array(results).astype(np.float32))
    
    return np.array(atom_features)

def protein_atom_features_exp(atom_objects, close_protein_indices, explicit_H=True):
    atom_features = []
    atom_symbols = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
        'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]
    
    for idx in close_protein_indices:
        atom = atom_objects[idx]
        atom_symbol = atom.element if atom.element else 'Unknown'
        residue = atom.get_parent()
        residue_name = residue.get_resname() if residue else "Unknown"
        if residue:
            bonds = [a for a in residue.get_atoms() if a != atom]
            degree = len(bonds)
        else:
            degree = 0
        hybridization = 'Unknown'
        is_aromatic = atom.is_aromatic if hasattr(atom, 'is_aromatic') else False
        results = one_of_k_encoding_unk(atom_symbol, atom_symbols + ['Unknown']) + \
                  one_of_k_encoding_unk(degree, [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(hybridization, ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Unknown']) + \
                  [is_aromatic]
        if explicit_H:
            num_hydrogens = atom.get_total_num_hydrogens() if hasattr(atom, 'get_total_num_hydrogens') else 0
            results += one_of_k_encoding_unk(num_hydrogens, [0, 1, 2, 3, 4])
        atom_features.append(np.array(results).astype(np.float32))
    
    return np.array(atom_features)

def proteinGraph_af(protein_pdb, atom_objects, atom_positions, ligand_pos, distance_threshold):
    #ligand = read_ligandMol2_file(ligand_pdb)
    #ligand_atom_num = ligand.GetNumAtoms()
    ligand_pos = np.array(ligand_pos)
 
    graph = nx.Graph()
    #print(protein_pdb)
    protein_mol = Chem.MolFromPDBFile(protein_pdb)
    #print(protein_mol)
    #ligand_center = np.mean(ligand_pos, axis=0)
    #atom_center = np.mean(atom_positions, axis=0)
    #ligand_pos_center = ligand_pos-ligand_center
    #atom_pos_center = atom_positions-atom_center
    #dis_matrix = distance_matrix(ligand_pos_center, atom_pos_center)
    dis_matrix = distance_matrix(ligand_pos, atom_positions)
    
    #close_ligand_indices, close_protein_indices = np.where(dis_matrix < distance_threshold)
    #close_protein_indices = np.unique(close_protein_indices)

    min_distances = np.min(dis_matrix, axis=0)  # 沿着配体原子维度取最小值
    close_protein_indices = np.where(min_distances < distance_threshold)[0]
    protein_feats = protein_atom_features_af(atom_objects, close_protein_indices)
    #print(close_protein_indices) 
    for i, atom_feats in zip(close_protein_indices, protein_feats):
        graph.add_node(i, feats=torch.from_numpy(atom_feats))
    
    close_protein_positions = atom_positions[close_protein_indices]

    # 计算这些蛋白质原子之间的两两距离
    """
    protein_distance_matrix = distance_matrix(close_protein_positions, close_protein_positions)
    for i in range(len(close_protein_indices)):
       for j in range(i + 1, len(close_protein_indices)):
           if protein_distance_matrix[i, j] < 1.5:
               graph.add_edge(close_protein_indices[i], close_protein_indices[j])

    """
    try:
        for bond in protein_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i in close_protein_indices and j in close_protein_indices:
                graph.add_edge(i, j)
    except AttributeError:
        # 如果 protein_mol 是 None，根据距离判断是否有边
        print("Warning: protein_mol is None. Using distance-based edge detection.")
        n_atoms = len(atom_positions)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(atom_positions[i] - atom_positions[j])
                if distance < 1.5:
                    graph.add_edge(i, j)
    #graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    protein_atoms = len(graph.nodes)

    return x, edge_index, protein_atoms, close_protein_positions

def proteinGraph_exp(atom_objects, atom_positions, ligand_pos, distance_threshold):
    ligand_pos = np.array(ligand_pos)
 
    graph = nx.Graph()

    #ligand_center = np.mean(ligand_pos, axis=0)
    #atom_center = np.mean(atom_positions, axis=0)
    #ligand_pos_center = ligand_pos-ligand_center
    #atom_pos_center = atom_positions-atom_center
    #dis_matrix = distance_matrix(ligand_pos_center, atom_pos_center)
    dis_matrix = distance_matrix(ligand_pos, atom_positions)

    close_ligand_indices, close_protein_indices = np.where(dis_matrix < distance_threshold)
    close_protein_indices = np.unique(close_protein_indices)
    
    protein_feats = protein_atom_features_exp(atom_objects, close_protein_indices)
    
    for i, atom_feats in zip(close_protein_indices, protein_feats):
        graph.add_node(i, feats=torch.from_numpy(atom_feats))
  
    close_protein_positions = atom_positions[close_protein_indices]

    # 计算这些蛋白质原子之间的两两距离
    protein_distance_matrix = distance_matrix(close_protein_positions, close_protein_positions)
    for i in range(len(close_protein_indices)):
       for j in range(i + 1, len(close_protein_indices)):
           if protein_distance_matrix[i, j] < 1.5:
               graph.add_edge(close_protein_indices[i], close_protein_indices[j])

    
    #graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    protein_atoms = len(graph.nodes)

    return x, edge_index, protein_atoms

def inter_graph(ligand, ligand_pos, atom_positions, distance_threshold):
    ligand_atom_num = ligand.GetNumAtoms()
    ligand_pos = np.array(ligand_pos)

    graph_inter = nx.Graph()
    #ligand_center = np.mean(ligand_pos, axis=0)
    #atom_center = np.mean(atom_positions, axis=0)
    #ligand_pos_center = ligand_pos-ligand_center
    #atom_pos_center = atom_positions-atom_center
    #dis_matrix = distance_matrix(ligand_pos_center, atom_pos_center)
    dis_matrix = distance_matrix(ligand_pos, atom_positions)
    #print("distance: ", dis_matrix)
    node_idx = np.where(dis_matrix < distance_threshold)
    #print("max, min:", max(node_idx[0]), min(node_idx[1]))
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+ligand_atom_num)
    #print("protein atom numb, protein nodes:", len(atom_positions), len(node_idx[0]), len(node_idx[1]))
    #print("\n",node_idx[0], node_idx[1],"\n")
    #graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T
    return edge_index_inter 

def pdbqtGraphs(ligand_pdb, protein_pdbqt, protein_pdb, distance_threshold=15):
    file_extension = os.path.splitext(ligand_pdb)[1].lower()

    if file_extension == '.pdb':
        ligand = read_ligandPdb_file(ligand_pdb)
    elif file_extension == '.mol2':
        ligand = read_ligandMol2_file(ligand_pdb)
    elif file_extension == '.pdbqt':
        ligand = read_ligandPdbqt_file(ligand_pdb)
    ligand_atom_num = ligand.GetNumAtoms()

    atom_objects, atom_positions, atom_indices = read_pdbqt_file(protein_pdbqt)
    
    ligand_pos = ligand.GetConformers()[0].GetPositions()
    
    x_l, edge_index_l = mol2graph(ligand)
    # the index in edge cannot exced the index of node, therefore we need to include all nodes of protein pocket.
    # the protein needs to be extracted as pocket. the distance_threshold losses its meaning
    x_p, edge_index_p, protein_atoms_num, close_protein_positions = proteinGraph_af(protein_pdb, atom_objects, atom_positions, ligand_pos, distance_threshold=25)
    #print(x_l.shape, x_p.shape,"\n")
    x = torch.cat([x_l, x_p], dim=0)
    #print("concated x, lig_num, prot_num: ",x.shape, ligand_atom_num, protein_atoms_num)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+ligand_atom_num], dim=-1)
    #edge_index_intra = torch.cat([edge_index_l, edge_index_p], dim=-1)
    edge_index_inter = inter_graph(ligand, ligand_pos, atom_positions, distance_threshold=15)

    #print("lig_edge, prot_edge, intra_edge", edge_index_l.shape, edge_index_p.shape, edge_index_intra.shape)
    #print(protein_atoms_num)
    #print(edge_index_intra)
    #print(ligand_pdb, ligand_pos)
    #print(protein_pdbqt, atom_positions)
    #print(edge_index_intra.max(), edge_index_inter.max(), x.shape[0])
    if edge_index_intra.max() >= x.shape[0] or edge_index_inter.max() >= x.shape[0]:
        print(ligand_pdb, protein_pdbqt)
    """
    if edge_index_intra.max() >= x.shape[0] or edge_index_inter.max() >= x.shape[0]:
        x = None
        edge_index_intra = None
        edge_index_inter = None
        pos = None
        split = None
        return x, edge_index_intra, edge_index_inter, pos, split
    """
    #pos = torch.concat([torch.from_numpy(ligand_pos), torch.from_numpy(atom_positions)], dim=0)
    pos = torch.concat([torch.from_numpy(ligand_pos), torch.from_numpy(close_protein_positions)], dim=0)
    split = torch.cat([torch.zeros((ligand_atom_num,)), torch.ones((protein_atoms_num,))], dim=0)


    return x, edge_index_intra, edge_index_inter, pos, split 
    
    
def pdbGraphs(ligand_pdb, protein_pdb, distance_threshold=15):
    ligand = read_ligandMol2_file(ligand_pdb)
   
    ligand_atom_num = ligand.GetNumAtoms()

    structure, atom_objects, atom_positions, atom_indices = read_proteinPdb_file(protein_pdb)
    #atom_objects, atom_positions, atom_indices = read_pdbqt_file(protein_pdbqt)
    

    ligand_pos = ligand.GetConformers()[0].GetPositions()
    
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p, protein_atoms_num = proteinGraph_exp(atom_objects, atom_positions, ligand_pos, distance_threshold=15)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+ligand_atom_num], dim=-1)
    edge_index_inter = inter_graph(ligand, ligand_pos, atom_positions, distance_threshold=5)

    #print(ligand_pdb, ligand_pos)
    #print(protein_pdb, atom_positions)
    #print(edge_index_intra.max(), edge_index_inter.max(), x.shape[0])
    if edge_index_intra.max() >= x.shape[0] or edge_index_inter.max() >= x.shape[0]:
        x = None
        edge_index_intra = None
        edge_index_inter = None
        pos = None
        split = None
        return x, edge_index_intra, edge_index_inter, pos, split
    pos = torch.concat([torch.from_numpy(ligand_pos), torch.from_numpy(atom_positions)], dim=0)
    split = torch.cat([torch.zeros((ligand_atom_num,)), torch.ones((protein_atoms_num,))], dim=0)

    return x, edge_index_intra, edge_index_inter, pos, split


def pdbGraphsPred(ligand_pdb, protein_pdb, distance_threshold=15):
    ligand = read_ligandPdb_file(ligand_pdb)
    ligand_atom_num = ligand.GetNumAtoms()

    structure, atom_objects, atom_positions, atom_indices = read_proteinPdb_file(protein_pdb)
    

    ligand_pos = ligand.GetConformers()[0].GetPositions()
    
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p, protein_atoms_num = proteinGraph_exp(atom_objects, atom_positions, ligand_pos, distance_threshold=15)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+ligand_atom_num], dim=-1)
    edge_index_inter = inter_graph(ligand, ligand_pos, atom_positions, distance_threshold=15)
    if edge_index_intra.max() >= x.shape[0] or edge_index_inter.max() >= x.shape[0]:
        x = None
        edge_index_intra = None
        edge_index_inter = None
        pos = None
        split = None
        return x, edge_index_intra, edge_index_inter, pos, split
    
    pos = torch.concat([torch.from_numpy(ligand_pos), torch.from_numpy(atom_positions)], dim=0)
    split = torch.cat([torch.zeros((ligand_atom_num,)), torch.ones((protein_atoms_num,))], dim=0)

    return x, edge_index_intra, edge_index_inter, pos, split



   
# ligand_pdb = '/home/data1/BGM/mutationEffect/ensemble/ligand/axitinib_4wa9.pdb'
# protein_pdbqt = '/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_F317L_3d5c7/Abl1_F317L_3d5c7_model_1_ptm_r0_seed3_axitinib_4wa9_axitinib_4wa9_vina_out.pdbqt'
# pdbGraphs_af(ligand_pdb, protein_pdbqt)

# ligand_pdb = '/home/data1/BGM/mutationEffect/ensemble/ligand/axitinib_4wa9.pdb'
# protein_pdb = '/home/data1/BGM/mutationEffect/ABL_AlphaFold/V299L_dc3ca/V299L_dc3ca_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb'
# pdbGraphs_exp(ligand_pdb, protein_pdb)
