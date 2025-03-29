import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch
from utils import *
from featurizer import *

def getInData(data):
    temp = data.split(',')
    newData = [int(da) for da in temp]
    return newData

def getFloData(data):
    temp = data.split(',')
    newData = [float(da) for da in temp]
    return newData

def getDockData(data1, data2):
    wildTemp = data1.split(',')
    mutTemp = data2.split(',')
    com_list = wildTemp + mutTemp
    dock = [round(float(da), 3) for da in com_list]
    return dock

def getProcessData(label, kpii):
    
    if label == 'train':
        allFoldPath = '../data/multiMutation_top2_twin_muts_tuple_inter/{}/dataTrain.dat'.format(kpii)
    elif label == 'val':
        allFoldPath = '../data/multiMutation_top2_twin_muts_tuple_inter/{}/dataVal.dat'.format(kpii)
    else:
        allFoldPath = '../data/multiMutation_top2_twin_muts_tuple_inter/{}/dataTest.dat'.format(kpii)



    resultElement1 = []
    resultElement2 = []
    resultMutVecExp = []
    resultMutVecAf = []
    resultSmiExp = []
    resultSmiAf = []
    resultMutSeqExp = []
    resultMutSeqAf = []
    resultRelDock = []
    resultDelY = []
    resultExp = []
    with open(allFoldPath, "r") as dat_file:
        for line in dat_file:
            line = line.strip() 
            elements = line.split(" ")
            resultElement1.append(elements[0])
            resultElement2.append(elements[1])

            mutVecExp = getInData(elements[2])
            resultMutVecExp.append(mutVecExp)

            mutVecAf = getInData(elements[3])
            resultMutVecAf.append(mutVecAf)

            resultSmiExp.append(elements[4])
            resultSmiAf.append(elements[5])

            resultMutSeqExp.append(elements[6])
            resultMutSeqAf.append(elements[7])

            resultRelDock.append(elements[8])

            resultDelY.append(elements[9])

            resultExp.append(elements[10])

    return resultElement1, resultElement2, resultMutVecExp, resultMutVecAf, resultSmiExp, resultSmiAf, resultMutSeqExp, resultMutSeqAf, resultRelDock, resultDelY, resultExp


if __name__ == '__main__':
    filePath = '../data/kpi2-inter.csv'
    kpi = pd.read_csv(filePath)['kpi'].tolist()
    subPath = '../data/multiMutation_top2_twin_muts_tuple_inter'
    blosum62_vectors = getBlosum62_vectors()

    for idx, kpii in enumerate(kpi):
        processed_data_file_train = '{}/{}/processed/data_train.pt'.format(subPath, kpii)
        processed_data_file_val = '{}/{}/processed/data_val.pt'.format(subPath, kpii)
        processed_data_file_test = '{}/{}/processed/data_test.pt'.format(subPath, kpii)

        if not os.path.isfile(processed_data_file_val):
            valElement1, valElement2, valMutVecExp, valMutVecAf, valSmiExp, valSmiAf, valMutSeqExp, valMutSeqAf, valRelDock, valDelY, valExp = getProcessData('val', kpii)
            val_data = formDatasetTwin_muts_inter('{}/{}'.format(subPath, kpii), 'data_val', valElement1, valElement2, valMutVecExp, valMutVecAf, valSmiExp, valSmiAf, valMutSeqExp, valMutSeqAf, valRelDock, valDelY, valExp, blosum62_vectors)    
            del valElement1, valElement2, valMutVecExp, valMutVecAf, valSmiExp, valSmiAf, valMutSeqExp, valMutSeqAf, valRelDock, valDelY, valExp, val_data
        gc.collect()
        torch.cuda.empty_cache()

        if not os.path.isfile(processed_data_file_test):
            testElement1, testElement2, testMutVecExp, testMutVecAf, testSmiExp, testSmiAf, testMutSeqExp, testMutSeqAf, testRelDock, testDelY, testExp = getProcessData('test', kpii)
            test_data = formDatasetTwin_muts_inter('{}/{}'.format(subPath, kpii), 'data_test', testElement1, testElement2, testMutVecExp, testMutVecAf, testSmiExp, testSmiAf, testMutSeqExp, testMutSeqAf, testRelDock, testDelY, testExp, blosum62_vectors)
            del testElement1, testElement2, testMutVecExp, testMutVecAf, testSmiExp, testSmiAf, testMutSeqExp, testMutSeqAf, testRelDock, testDelY, testExp, test_data
        gc.collect()
        torch.cuda.empty_cache()

        if not os.path.isfile(processed_data_file_train):
            trainElement1, trainElement2, trainMutVecExp, trainMutVecAf, trainSmiExp, trainSmiAf, trainMutSeqExp, trainMutSeqAf, trainRelDock, trainDelY, trainExp = getProcessData('train', kpii)
            train_data = formDatasetTwin_muts_inter('{}/{}'.format(subPath, kpii), 'data_train', trainElement1, trainElement2, trainMutVecExp, trainMutVecAf, trainSmiExp, trainSmiAf, trainMutSeqExp, trainMutSeqAf, trainRelDock, trainDelY, trainExp, blosum62_vectors)
            del trainElement1, trainElement2, trainMutVecExp, trainMutVecAf, trainSmiExp, trainSmiAf, trainMutSeqExp, trainMutSeqAf, trainRelDock, trainDelY, trainExp, train_data
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
