import matplotlib.pyplot as plt
import pandas as pd

maePath = '../results/metrics/a-maeAll-01.csv'
df = pd.read_csv(maePath)
keys = df['Key'].tolist()
values = df['Value'].tolist()
plt.figure(figsize=(10,6))
plt.subplots_adjust(top=0.92, bottom=0.19, left=0.11, right=0.98)
plt.title('MAE', fontsize=27)
plt.bar(keys, values)
plt.yticks(fontsize=27)
plt.xticks(fontsize=27, rotation=20)
plt.grid(color='grey', alpha=0.8, linestyle='-')
savePath = '../results/metrics/maeAll.png'
plt.savefig(savePath, dpi=300)


rmsePath = '../results/metrics/a-rmseAll-01.csv'
df = pd.read_csv(rmsePath)
keys = df['Key'].tolist()
values = df['Value'].tolist()
plt.figure(figsize=(10,6))
plt.subplots_adjust(top=0.92, bottom=0.19, left=0.11, right=0.98)
plt.title('RMSE', fontsize=27)
plt.bar(keys, values)
plt.yticks(fontsize=27)
plt.xticks(fontsize=27, rotation=20)
plt.grid(color='grey', alpha=0.8, linestyle='-')
savePath = '../results/metrics/rmseAll.png'
plt.savefig(savePath, dpi=300)

pearPath = '../results/metrics/a-rValueAll-01.csv'
df = pd.read_csv(pearPath)
keys = df['Key'].tolist()
values = df['Value'].tolist()
plt.figure(figsize=(10,6))
plt.subplots_adjust(top=0.92, bottom=0.19, left=0.11, right=0.98)
plt.title('Pearson coefficients', fontsize=27)
plt.bar(keys, values)
plt.yticks(fontsize=27)
plt.xticks(fontsize=27, rotation=20)
plt.grid(color='grey', alpha=0.8, linestyle='-')
savePath = '../results/metrics/pearsonAll.png'
plt.savefig(savePath, dpi=300)

spearPath = '../results/metrics/a-spearAll-01.csv'
df = pd.read_csv(spearPath)
keys = df['Key'].tolist()
values = df['Value'].tolist()
plt.figure(figsize=(10,6))
plt.subplots_adjust(top=0.92, bottom=0.19, left=0.11, right=0.98)
plt.title('Pearson coefficients', fontsize=27)
plt.bar(keys, values)
plt.yticks(fontsize=27)
plt.xticks(fontsize=27, rotation=20)
plt.grid(color='grey', alpha=0.8, linestyle='-')
savePath = '../results/metrics/spearmanAll.png'
plt.savefig(savePath, dpi=300)