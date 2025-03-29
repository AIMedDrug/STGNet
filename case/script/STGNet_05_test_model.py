from model import *
from utils import *
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import gc
import torch
import time
import torch.optim as optim


def test(testDataLoader, comModel, device):
    expNameResult = []
    afNameResult = []
    expResult = []
    relativeResult = []
    deltayResult = []
    yPredResult = []

    comModel.eval()
    for batch_idx, data in enumerate(testDataLoader):
        data_mol_exp = data[0]
        data_mol_af = data[1]
        data_pro_exp = data[2]
        data_pro_af = data[3]
        data_inter_exp = data[4]
        data_inter_af = data[5]

        x1 = data_mol_exp.x.to(device)
        edge_index1 = data_mol_exp.edge_index.to(device)
        edge_attr1 = data_mol_exp.edge_attr.to(device)
        batch1 = data_mol_exp.batch.to(device)

        x2 = data_mol_af.x.to(device)
        edge_index2 = data_mol_af.edge_index.to(device)
        edge_attr2 = data_mol_af.edge_attr.to(device)
        batch2 = data_mol_af.batch.to(device)

        ligand_exp_input = (x1, edge_index1, edge_attr1, batch1)
        ligand_af_input = (x2, edge_index2, edge_attr2, batch2)


        exp_feature = data_pro_exp.x.to(device)
        exp_edge_index = data_pro_exp.edge_index.to(device)
        exp_weight = data_pro_exp.edge_weight.unsqueeze(1).to(device)
        exp_batch = data_pro_exp.batch.to(device)

        af_feature = data_pro_af.x.to(device)
        af_edge_index = data_pro_af.edge_index.to(device)
        af_weight = data_pro_af.edge_weight.unsqueeze(1).to(device)
        af_batch = data_pro_af.batch.to(device)

        map_exp_input = (exp_feature, exp_edge_index, exp_weight, exp_batch)
        map_af_input = (af_feature, af_edge_index, af_weight, af_batch)

        exp_x = data_inter_exp.x.to(torch.float).to(device)
        exp_edge_index_intra = data_inter_exp.edge_index_intra.to(torch.long).to(device)
        exp_edge_index_inter = data_inter_exp.edge_index_inter.to(torch.long).to(device)
        exp_pos = data_inter_exp.pos.to(device)
        exp_split = data_inter_exp.split.to(device)
        exp_batch = data_inter_exp.batch.to(device)

        af_x = data_inter_af.x.to(torch.float).to(device)
        af_edge_index_intra = data_inter_af.edge_index_intra.to(torch.long).to(device)
        af_edge_index_inter = data_inter_af.edge_index_inter.to(torch.long).to(device)
        af_pos = data_inter_af.pos.to(device)
        af_split = data_inter_af.split.to(device)
        af_batch = data_inter_af.batch.to(device)

        inter_exp_input =(exp_x, exp_edge_index_intra, exp_edge_index_inter, exp_pos, exp_split, exp_batch)
        inter_af_input = (af_x, af_edge_index_intra, af_edge_index_inter, af_pos, af_split, af_batch)
        
        deltayValue = data_pro_af.deltaY.to(device)
        expValue = data_pro_exp.exp.to(device)

        y_pred = comModel(ligand_exp_input, ligand_af_input,
                          map_exp_input, map_af_input,
                          inter_exp_input, inter_af_input,
                          device
                          )

        expName = data_pro_exp.expName
        afName = data_pro_af.afName
        relativeDock = data_pro_af.relative_dock

        expNameResult.append(expName)
        afNameResult.append(afName)
        expResult.append(expValue.tolist())
        relativeResult.append(relativeDock.tolist())
        deltayResult.append(deltayValue.tolist())
        yPredResult.append(y_pred.tolist())
    
    expNameResult = flatteningResult(expNameResult)
    afNameResult = flatteningResult(afNameResult)
    expResult = flatteningResult(expResult)
    relativeResult = flatteningResult(relativeResult)
    deltayResult = flatteningResult(deltayResult)
    yPredResult = flatteningResult(yPredResult)

    gc.collect()
    torch.cuda.empty_cache()
    return expNameResult, afNameResult, expResult, relativeResult, deltayResult, yPredResult


# 2-GCN

gpu_index = 1 

if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_index}')
else:
    device = torch.device('cpu')

args = {
    'ligand_feature_dim': 75,
    'map_feature_dim': 56,
    'trans_feature': 287, 
    'inter_feature': 64, 
    'num_layer': 2,
    'num_heads': 2,
    'hidden_dim': 64,
    'fc_middle_dim': 16,
    'dim': 64,
    'middle_dim': 64,
    'output_dim': 16,
    'dropout': 0.2,

}

batch_size = 16

df = pd.read_csv('../data/kpi2-inter.csv')['kpi'].tolist()



for idx, kpii in enumerate(df):
    curr_kpi = kpii

    trainData = formDatasetTwin_muts_inter(root='../data/multiMutation_top2_twin_muts_tuple_inter/{}'.format(curr_kpi), dataset='data_train')
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    valData = formDatasetTwin_muts_inter(root='../data/multiMutation_top2_twin_muts_tuple_inter/{}'.format(curr_kpi), dataset='data_val')
    valLoader = DataLoader(valData, batch_size=batch_size, shuffle=True)
    testData = formDatasetTwin_muts_inter(root='../data/multiMutation_top2_twin_muts_tuple_inter/{}'.format(curr_kpi), dataset='data_test')
    testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False)



    from model import *
    comModel = SiameseNetworkCombinationInter(args, device).to(device)
    comModel.load_state_dict(torch.load(f'../results/pth/{curr_kpi}_model_weights.pth'))

    
    expNameResult, afNameResult, expResult, relativeResult, deltayResult, yPredResult = test(testLoader, comModel, device)

    yResult = {}
    yResult['expName'] = expNameResult
    yResult['afName'] = afNameResult
    yResult['exp'] = expResult
    yResult['relativeDock'] = relativeResult
    yResult['deltaY'] = deltayResult
    yResult['pred'] = yPredResult
    
    savePath = f'../results/single/{curr_kpi}_yResult.csv'
    df = pd.DataFrame(yResult)
    df.to_csv(savePath, index=False)

    gc.collect()
    torch.cuda.empty_cache()


