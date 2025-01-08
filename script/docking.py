#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

from vina import Vina
import pandas as pd
import MDAnalysis as mda
import os
import glob
from pymol import cmd
import subprocess

import os
from pymol import cmd

def ensure_directory_exists(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def align_and_save_proteins(protein1_path, protein2_path, aligned_protein1_path, aligned_protein2_path):
    cmd.load(protein1_path, "protein1")
    cmd.load(protein2_path, "protein2")

    cmd.align("protein1", "protein2")

    cmd.save(aligned_protein1_path, "protein1")
    cmd.save(aligned_protein2_path, "protein2")

    cmd.delete("protein1")
    cmd.delete("protein2")

def extract_first_conformation(input_pdbqt, output_pdbqt):
    with open(input_pdbqt, 'r') as f:
        data = f.readlines()
    first_conformation_lines = []
    read_next_conformation = False
    for line in data:
        if line.startswith("MODEL"):
            read_next_conformation = True
        elif line.startswith("ENDMDL"):
            break
        elif read_next_conformation:
            first_conformation_lines.append(line)
    first_conformation = ''.join(first_conformation_lines)
        
    with open(output_pdbqt, 'w') as temp_f:
        temp_f.write(first_conformation)

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt):
    command = ['prepare_receptor4', '-r', input_pdb, '-o', output_pdbqt]
    subprocess.run(command)
    print('Successfully written {}'.format(output_pdbqt))
    
def get_avgCoordinates(proteinPath):
    residue_numbers = [20, 41, 131, 152]
    u = mda.Universe(proteinPath)
    ca_coordinates = []
    for residue in u.residues:
        if residue.resid in residue_numbers:
            ca_atom = residue.atoms.select_atoms("name CA")
            if len(ca_atom) > 0:
                ca_coordinates.append(ca_atom.center_of_geometry())
    average_coordinates = sum(ca_coordinates) / len(ca_coordinates)
    return average_coordinates

def start_vina(proteinPath, ligandPath, avgCoordinates, autoDockFile):
    v = Vina(sf_name='vina', cpu=8)
    v.set_receptor(proteinPath)
    v.set_ligand_from_file(ligandPath)
    v.compute_vina_maps(center=[avgCoordinates[0], avgCoordinates[1], avgCoordinates[2]], box_size=[28, 28, 28])
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses(autoDockFile, n_poses=5, overwrite=True)
    docked_energy = v.energies(n_poses=1,  energy_range=3.0)
    total_DockedEnergy = docked_energy[0]
    return total_DockedEnergy

wild = 'WILD_8dc32'
wildPath = glob.glob('../ABL_AlphaFold/{}/{}_unrelaxed_rank_001_alphafold2_ptm_model_*.pdb'.format(wild, wild))[0]

ligandPath = '../data/kpi2.csv'
ligandDf = pd.read_csv(ligandPath)
ligandPath = []
for kpii, pdbid in zip(ligandDf['kpi'].tolist(), ligandDf['pdbID'].tolist()):
    name = kpii + '_' +pdbid
    output_pdbqt = 'ligand/{}_temp.pdbqt'.format(name)
    ligandPath.append(output_pdbqt)

mutationList = pd.read_csv('protein.csv')['protein'].tolist()
mutationPath = []
for name in mutationList:
    mutpath = glob.glob('../ABL_AlphaFold/{}/{}_unrelaxed_rank_001_alphafold2_ptm_model_*.pdb'.format(name, name))[0]
    
    tempAlignedFile = 'aligned/{}'.format(name)
    ensure_directory_exists(tempAlignedFile)
    aligned_wild_path = '{}/{}_aligned.pdb'.format(tempAlignedFile, wild)
    aligned_mutation_path = '{}/{}_aligned.pdb'.format(tempAlignedFile, name)
    convert_aligned_wild_path = '{}/{}_aligned.pdbqt'.format(tempAlignedFile, wild)
    convert_aligned_mutation_path = '{}/{}_aligned.pdbqt'.format(tempAlignedFile, name)

    align_and_save_proteins(wildPath, mutpath, aligned_wild_path, aligned_mutation_path)
    convert_pdb_to_pdbqt(aligned_wild_path, convert_aligned_wild_path)
    convert_pdb_to_pdbqt(aligned_mutation_path, convert_aligned_mutation_path)
    
    wildAvg_coor = get_avgCoordinates(convert_aligned_wild_path)
    mutAvg_coor = get_avgCoordinates(convert_aligned_mutation_path)
   
    dockEnergyDic = {'wild_vina': [], 'wildDockEnergy': [], 'mut_vina': [], 'mutDockEnergy': [], 'relativeEnergy': []}
    for ligPath in ligandPath:
        ligName = ligPath.split('/')[-1].split('.')[0]
        tempAlignedFile = 'docked/{}'.format(name)
        ensure_directory_exists(tempAlignedFile)

    
        autoDockFile_wild = '{}/{}_{}_vina_output.pdbqt'.format(tempAlignedFile, wild, ligName)
        wildEnergy = start_vina(convert_aligned_wild_path, ligPath, wildAvg_coor, autoDockFile_wild)
        autoDockFile_mutation = '{}/{}_{}_vina_output.pdbqt'.format(tempAlignedFile, name, ligName)
        mutEnergy = start_vina(convert_aligned_mutation_path, ligPath, mutAvg_coor, autoDockFile_mutation)
        relativeEnergy = mutEnergy - wildEnergy
        dockEnergyDic['wild_vina'].append(autoDockFile_wild)
        dockEnergyDic['wildDockEnergy'].append(wildEnergy)
        dockEnergyDic['mut_vina'].append(autoDockFile_mutation)
        dockEnergyDic['mutDockEnergy'].append(mutEnergy)
        dockEnergyDic['relativeEnergy'].append(relativeEnergy)
        
    df = pd.DataFrame(dockEnergyDic)
    dfPath = 'docked/{}/{}_dockEnergyAll.csv'.format(name, name)
    df.to_csv(dfPath, index=False)
    if os.path.exists(dfPath):
        print('writing to file : ' + dfPath)