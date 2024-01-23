import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


adata_MCF7 = sc.read("./datasets/sciplex3/adata_MCF7_test.h5ad")
adata_MCF7.uns["log1p"] = {'base': None}
sc.pp.highly_variable_genes(adata_MCF7, n_top_genes=2000, subset=False)

# 删除其它基因
adata_MCF7 = adata_MCF7[:, adata_MCF7.var.highly_variable].copy()

adata_MCF7.obs["product_name"] = [x.split(" ")[0] for x in adata_MCF7.obs["product_name"]]
adata_MCF7.obs.loc[
        adata_MCF7.obs["product_name"].str.contains("Vehicle"), "product_name"
] = "control"

adata_MCF7_control = adata_MCF7[adata_MCF7.obs.product_name.isin(["control"])]
print(adata_MCF7_control.obs.shape)

# print(adata_MCF7_control.X.A)

data = pd.DataFrame(adata_MCF7_control.X.A, index=adata_MCF7_control.obs_names, 
                    columns=adata_MCF7_control.var_names)

data = data.sample(n=200, replace=False)

s = cosine_similarity(data.values, data.values)
a = pd.DataFrame(s, columns=data.index, index=data.index)

f, ax1 = plt.subplots(figsize = (30,30),nrows=1)
#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签               
sns.heatmap(data=a, annot=False, annot_kws={'size':8,'weight':'bold', 'color':'black'})
ax1.set_xlabel('')
ax1.set_ylabel('')
plt.xticks(rotation = 60, fontsize=8, ticks=[])
plt.yticks(rotation = 0, fontsize=8, ticks=[])
plt.title("cell line: MCF7, pert_time: 6h")
# plt.title("cell line: MCF7, drug: BRD-K21680192, pert_time: 24h")
plt.show()
