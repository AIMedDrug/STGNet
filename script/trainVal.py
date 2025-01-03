
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import gc
import torch
import time
import torch.optim as optim
import sys
sys.path.append('models')
from model import *
sys.path.append('utils')
from utils import *

def train(epoch, trainDataLoader, comModel, criterion, optimizer, device):
    total_loss = 0
    n_batches = 0
    
    comModel.train()
    for batch_idx, data in enumerate(trainDataLoader):
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

        y_pred = comModel(ligand_exp_input, ligand_af_input,
                          map_exp_input, map_af_input,
                          inter_exp_input, inter_af_input,
                          device
                          )

        loss = criterion(y_pred.type(torch.DoubleTensor), deltayValue.type(torch.DoubleTensor))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss = total_loss + loss.data
        n_batches += 1
    
    gc.collect()
    torch.cuda.empty_cache()
 
    avgLoss = total_loss / n_batches
    print('Epoch {} --------------------- {}'.format(epoch, avgLoss))
    return avgLoss.item()

def val(epoch, valDataLoader, comModel, criterion, device):
    total_loss = 0
    n_batches = 0

    comModel.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(valDataLoader):
            gc.collect()
            torch.cuda.empty_cache()

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

            y_pred = comModel(ligand_exp_input, ligand_af_input,
                            map_exp_input, map_af_input,
                            inter_exp_input, inter_af_input,
                            device
                            )

            loss = criterion(y_pred.type(torch.DoubleTensor), deltayValue.type(torch.DoubleTensor))
            total_loss += loss.data
            n_batches += 1
    
    gc.collect()
    torch.cuda.empty_cache()
    
    avg_loss = total_loss / n_batches
    print('Epoch {} --------------------- Val Loss: {}'.format(epoch, avg_loss))
    return avg_loss.item()

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

epoches = 50
batch_size = 16


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

df = pd.read_csv('../data/initial/tki.csv')['tki'].tolist()



trainLoss = {}
valLoss = {}
curr_kpi = 'axitinib'

trainData = formDatasetTwin_muts_inter(root='../data/filterData/top5/{}'.format(curr_kpi), dataset='data_train')
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
valData = formDatasetTwin_muts_inter(root='../data/filterData/top5/{}'.format(curr_kpi), dataset='data_val')
valLoader = DataLoader(valData, batch_size=batch_size, shuffle=True)
testData = formDatasetTwin_muts_inter(root='../data/filterData/top5/{}'.format(curr_kpi), dataset='data_test')
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False)


sys.path.append('models')
from model import *
comModel = SiameseNetworkCombinationInter(args, device).to(device)
learning_rate=0.0001
optimizer = optim.Adam(comModel.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = torch.nn.MSELoss()

lossTrain = []
lossVal = []

start_time = time.time()
for idx, epoch in enumerate(range(epoches)):
    avgLoss_train = train(epoch, trainLoader, comModel, criterion, optimizer, device)
    avgLoss_val = val(epoch, valLoader, comModel, criterion, device)
    # scheduler.step(avgLoss_val)
    # scheduler.step()
    lossTrain.append(avgLoss_train)
    lossVal.append(avgLoss_val)
    if idx == 0:
        epoch_time = time.time()
        temp_time = epoch_time - start_time
        print(f'The first training epoch is {temp_time} s')
        print('\n')

model_save_path = f'../result/top5/stgNet/pth/{curr_kpi}_model_weights.pth'
torch.save(comModel.state_dict(), model_save_path)

end_time = time.time()
total_time = end_time - start_time
print(f'The training epoches are {total_time} s')

keyname0 = 'loss_{}'.format(curr_kpi)
trainLoss[keyname0] = lossTrain
valLoss[keyname0] = lossVal



expNameResult, afNameResult, expResult, relativeResult, deltayResult, yPredResult = test(testLoader, comModel, device)

yResult = {}
yResult['expName'] = expNameResult
yResult['afName'] = afNameResult
yResult['exp'] = expResult
yResult['relativeDock'] = relativeResult
yResult['deltaY'] = deltayResult
yResult['pred'] = yPredResult

savePath = f'../result/top5/stgNet/single/{curr_kpi}_{epoches}_lr={learning_rate}_yResult.csv'
df = pd.DataFrame(yResult)
df.to_csv(savePath, index=False)

savePath = f'../result/top5/stgNet/single/{curr_kpi}_Trainloss_{epoches}_lr={learning_rate}.csv'
df = pd.DataFrame(trainLoss)
df.to_csv(savePath, index=False)

savePath = f'../result/top5/stgNet/single/{curr_kpi}_Valloss_{epoches}_lr={learning_rate}.csv'
df = pd.DataFrame(valLoss)
df.to_csv(savePath, index=False)


