import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import csv
import random
import os
import numpy as np
import csv


mutationName = pd.read_csv('../data/initial/mutationName.csv')['mutation'].tolist()
kpiName = pd.read_csv('../data/initial/tki.csv')['tki'].tolist()

trueResults = {}
with open('../data/initial/resistance.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for idx, row in enumerate(reader):
        name = row[1] + '_' + row[0]
        temp = []
        if row[2] == 'nd':
            temp.append(0)
        else:
            temp.append(1)
        temp.append(row[2])
        trueResults[name] = temp
    


rmseDic = {}
maeDic = {}
rvalueDic = {}
spearDic = {}
kpi = 'axitinib'
path = f'../result/top5/stgNet/single/{kpi}_50_lr=0.0001_yResult.csv'
df = pd.read_csv(path)
sorted_data = df.sort_values(by=['expName', 'afName'])

reformData = {'expName': [], 'afName': [], 'exp': [], 'alterPred': []}
reformData_filter = {'expName': [], 'afName': [], 'exp': [], 'alterPred_avg': []}
for mutName in mutationName:
    filtered_df = sorted_data[sorted_data.iloc[:, 1].str.startswith(f'{mutName}') & sorted_data.iloc[:, 1].str.endswith(f'{kpi}')]
    if filtered_df.empty:
        continue
    expName = filtered_df['expName'].tolist()
    afName = filtered_df['afName'].tolist()
    exp = filtered_df['exp'].tolist()
    relativeDock = filtered_df['relativeDock'].astype(float).tolist()
    pred = filtered_df['pred'].astype(float).tolist()
    alterPred = [x + y for x, y in zip(relativeDock, pred)]

    reformData['expName'].extend(expName)
    reformData['afName'].extend(afName)
    reformData['exp'].extend(exp)
    reformData['alterPred'].extend(alterPred)

    expAvg = sum(exp) / len(exp)
    alterPredA = alterPred
    alterAvg = sum(alterPredA) / len(alterPredA)
    expNameAvg = expName[0].split('_')[0] + '_' + expName[0].split('_')[1]
    afNameAvg = afName[0].split('_')[0] + '_' + afName[0].split('_')[-1]
    reformData_filter['expName'].append(expNameAvg)
    reformData_filter['afName'].append(afNameAvg)
    reformData_filter['exp'].append(expAvg)
    reformData_filter['alterPred_avg'].append(alterAvg)

dfNew = pd.DataFrame(reformData)
dfNew.to_csv(f'../preds/top5/{kpi}_results.csv', index=False)

dfNew_filter = pd.DataFrame(reformData_filter)
dfNew_filter.to_csv(f'../preds/top5/{kpi}_results_filter.csv', index=False)



trueData = dfNew_filter['exp'].astype(float).tolist()
predData = dfNew_filter['alterPred_avg'].astype(float).tolist()

trueData = np.array(trueData)
predData = np.array(predData)
rmse = np.sqrt(mean_squared_error(trueData, predData))
rmseDic[kpi] = rmse
mae = mean_absolute_error(trueData, predData)
maeDic[kpi] = mae
pearson_corr, _ = pearsonr(trueData, predData)
rvalueDic[kpi]= pearson_corr
spearman_corr, _ = spearmanr(trueData, predData)
spearDic[kpi] = spearman_corr


writePath = f'../preds/top5/a-rmseAll-01.csv'
with open(writePath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
    for key, value in rmseDic.items():
        writer.writerow([key, value])
print('Writting to path: ' + writePath)

writePath = f'../preds/top5/a-maeAll-01.csv'
with open(writePath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
    for key, value in maeDic.items():
        writer.writerow([key, value])
print('Writting to path: ' + writePath)

writePath = f'../preds/top5/a-rValueAll-01.csv'
with open(writePath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
    for key, value in rvalueDic.items():
        writer.writerow([key, value])
print('Writting to path: ' + writePath)

writePath = f'../preds/top5/a-spearAll-01.csv'
with open(writePath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
    for key, value in spearDic.items():
        writer.writerow([key, value])
print('Writting to path: ' + writePath)



