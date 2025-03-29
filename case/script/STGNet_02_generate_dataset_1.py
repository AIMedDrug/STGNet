# -*- coding: iso-8859-1 -*-
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from utils import *
import pickle
import os
import ast
from multiprocessing import Pool

def splitData_dat(filePath1, filePath2, filePath3, filePath4):
    allKeys = []
    allData = {}
    with open(filePath3, "r") as dat_file:
        for line in dat_file:
            line = line.strip() 
            elements = line.split(" ") 
            keyName = elements[0]
            allKeys.append(keyName)
            allData[keyName] = elements
    
    random.shuffle(allKeys)
    allData = {key: allData[key] for key in allKeys}

    mutationName = pd.read_csv(filePath2)['mutation'].tolist()
    kpi = pd.read_csv(filePath1)['kpi'].tolist()

    
    for kpii in kpi:
        testKpiName = kpii
        dataTrain = []
        dataTest = []
        
        for key in allKeys:
            if key.split('_')[1] == testKpiName:
                dataTest.append(allData[key])
            else:
                dataTrain.append(allData[key])
           
        kpiPath = '{}/{}'.format(filePath4, testKpiName)
        if not os.path.exists(kpiPath):
            os.makedirs(kpiPath)
        validation_size = max(1, len(dataTrain) // 10)
        dataVal = random.sample(dataTrain, validation_size)
        dataTrain = [item for item in dataTrain if item not in dataVal]
        

        savePath = '{}/dataTrain.dat'.format(kpiPath)
        with open(savePath, 'w') as dat_file:
            for row in dataTrain:
                line = ' '.join(row)
                dat_file.write(line + '\n')
        print('writing ' + savePath + ' finished!')
        
        savePath = '{}/dataTest.dat'.format(kpiPath)
        with open(savePath, 'w') as dat_file:
            for row in dataTest:
                line = ' '.join(row)
                dat_file.write(line + '\n')   
        print('writing ' + savePath + ' finished!')  

        savePath = '{}/dataVal.dat'.format(kpiPath)
        with open(savePath, 'w') as dat_file:
            for row in dataVal:
                line = ' '.join(row)
                dat_file.write(line + '\n')
        print('writing ' + savePath + ' finished!')
        print('\n')  


def process_single_mutation(args):
    element1, element2, relative_dock, delta_y, exp, seqVoc, sequenceDic, smilesDic = args
    if exp == 'nd' or delta_y == 'nd':
        return None
    kpi_exp = element1.split('_')[1]
    kpi_af = element2.split('_')[-1]
    mut1 = element1.split('_')[0]
    mut2 = element2.split('_')[0]

    mutVari_exp, _ = makeVariablee(sequenceDic[mut1], seqVoc)
    mutvec_exp = ','.join(map(str, mutVari_exp))

    muVari_af, _ = makeVariablee(sequenceDic[mut2], seqVoc)
    mutvec_af = ','.join(map(str, muVari_af))


    smi_exp = smilesDic[kpi_exp]
    smi_af = smilesDic[kpi_af]


    mutSeq_exp = sequenceDic[mut1]
    mutSeq_af = sequenceDic[mut2]
    
    line = f"{element1} {element2} {mutvec_exp} {mutvec_exp} {smi_exp} {smi_af} {mutSeq_exp} {mutSeq_af} {relative_dock} {delta_y} {exp}"
    return line

def transToDat_All(filePath1, filePath2, filePath3, savePath):
    vocab_path = '../data/seq_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        seqVoc = pickle.load(f) 

    kpiDf = pd.read_csv(filePath2)
    smilesDic = {} 
    for kpi, smiles in zip(kpiDf['kpi'].tolist(), kpiDf['smiles'].tolist()):
        if smiles != 'nd':
            smilesDic[kpi] = smiles
    
    muSeqDf = pd.read_csv(filePath3)
    sequenceDic = {}
    for name, seq in zip(muSeqDf['mutationName'].tolist(), muSeqDf['sequence'].tolist()):
        sequenceDic[name] = seq
    
    resisDf = pd.read_csv(filePath1)
    element1 = resisDf['element1']
    element2 = resisDf['element2']
    relative_dock = resisDf['relative_dock']
    delta_y = resisDf['delta_y']
    exp = resisDf['exp']
    
    args_list = [(ele1, ele2, rey, deltay, ex, seqVoc, sequenceDic, smilesDic) for ele1, ele2, rey, deltay, ex in zip(element1, element2, relative_dock, delta_y, exp)]
    with Pool(18) as pool:
        results = pool.map(process_single_mutation, args_list)
    with open(savePath, "w") as dat_file:
        for result in results:
            if result is not None:
                dat_file.write(result + '\n')
    print('writing ' + savePath + ' finished!')
    



if __name__ == '__main__':
    filePath1 = '../data/resistance_top2_twin_muts_tuple_noGauss.csv'
    filePath2 = '../data/kpi2-inter.csv'
    filePath3 = '../data/mutationSeq.csv'
    savePath = '../data/multiMutation_top2_twin_muts_tuple_inter/dataAll.dat'
    transToDat_All(filePath1, filePath2, filePath3, savePath)
    
    filePath1 = '../data/kpi2-inter.csv'
    filePath2 = '../data/mutationName.csv'
    filePath3 = '../data/multiMutation_top2_twin_muts_tuple_inter/dataAll.dat'
    filePath4 = '../data/multiMutation_top2_twin_muts_tuple_inter'
    splitData_dat(filePath1, filePath2, filePath3, filePath4)


   


