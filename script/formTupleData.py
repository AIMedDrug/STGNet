import pandas as pd
import csv


mutationList = pd.read_csv('../data/initial/mutationName.csv')['mutation'].tolist()
kpiList = pd.read_csv('../data/initial/tki.csv')['tki'].tolist()

trueDic = {}
with open('../data/initial/resistance.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for idx, row in enumerate(reader):
        key = row[1] + '_' + row[0]
        trueDic[key] = row[2]

notSatisfied = []
allData = {'element1': [], 'element2': [], 'relative_dock': [], 'delta_y': [], 'exp': []}
for idx_mut, mutName in enumerate(mutationList):
    
    for firstKpi in kpiList:
        expKeyName = mutName + '_' + firstKpi
        expValue = trueDic[expKeyName]
        path = f'../data/dockEnergy/Abl1_{mutName}_relative_meanGaussian.csv'
        df = pd.read_csv(path)
        count = 0
        for secondKpi in kpiList:
            filtered_df = df[df.iloc[:, 0].str.startswith(f'{mutName}') & df.iloc[:, 0].str.endswith(f'{secondKpi}')]
            mutList = filtered_df.iloc[:, 0].tolist()
            relativeDockList = filtered_df.iloc[:, 1].tolist() 
            meanGauList = filtered_df.iloc[:, 2].tolist()
            if expValue == 'nd' or relativeDockList[0] == 'nd':
                for idx, (mut, deltaDock) in enumerate(zip(mutList, relativeDockList)):
                    allData['element1'].append(expKeyName + '_' + str(count))
                    allData['element2'].append(mut)
                    allData['relative_dock'].append('nd')
                    allData['delta_y'].append('nd')
                    allData['exp'].append('nd')
                    count = count + 1
                    if idx >= 5:
                        break
            else:

                relativeDockList = list(map(float, relativeDockList))
                expValue = round(float(expValue),2)
                meanGauList = list(map(float, meanGauList))
  
                within_tolerance = []
                for mut, x, meanG, in zip(mutList, relativeDockList, meanGauList):
                    distance = abs(x - meanG)
                    within_tolerance.append((mut, x, distance))
                sorted_within_tolerance = sorted(within_tolerance, key=lambda x: x[2])
                closest_5 = sorted_within_tolerance[:5]
                if len(closest_5) == 5:
                    for item in closest_5:
                        mutValue = round(float(item[1]), 3)
                        deltay = expValue - mutValue
                        allData['element1'].append(expKeyName + '_' + str(count))
                        allData['element2'].append(item[0])
                        allData['relative_dock'].append(mutValue)
                        allData['delta_y'].append(deltay)
                        allData['exp'].append(expValue)
                        count = count + 1
                else:
                    print(f'{mutName}_{kpi}')
                    notSatisfied.append(f'{mutName}_{kpi}')
                    


print(len(allData['element1']))



df = pd.DataFrame(allData)
path = '../data/filterData/top5_twin_muts_tuple.csv'
df.to_csv(path, index=False)