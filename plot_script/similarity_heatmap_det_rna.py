import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
 


rna_data = pd.read_csv("./datasets/plot_similarity/control_yes_dose_gene_24h.csv")
det_data = pd.read_csv("./datasets/plot_similarity/control_no_dose_gene_24h.csv")

rna_data = rna_data.sample(n=100, replace=False)
det_data = det_data.sample(n=100, replace=False)

data = pd.concat([det_data, rna_data], axis=0)

s = cosine_similarity(data.iloc[:,1:].values, data.iloc[:,1:].values)
a = pd.DataFrame(s, columns=data["index"], index=data["index"])

f, ax1 = plt.subplots(figsize = (30,30),nrows=1)
#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签               
sns.heatmap(data=a, annot=False, annot_kws={'size':8,'weight':'bold', 'color':'black'})
ax1.set_xlabel('')
ax1.set_ylabel('')
plt.xticks(rotation = 60, fontsize=8, ticks=[])
plt.yticks(fontsize=8, ticks=[])
plt.title("cell line: MCF7, pert_time: 24 h")
# plt.title("cell line: MCF7, drug: BRD-K21680192, pert_time: 24h")
plt.show()


