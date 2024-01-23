import pickle
import numpy as np
import scanpy as sc
import pandas as pd
from helper import rank_genes_groups_by_cov


adata = sc.read('./datasets/preprocess/sciplex4/sciplex4_filtered_genes_for_split.h5ad')


adata = adata[adata.obs.control.isin([1]) | adata.obs.dose.isin([10000])]


# 药物对一个 cell line 中的 978 + 1032 个基因影响最大的 50 基因
rank_genes_groups_by_cov(
    adata,
    groupby="cov_drug",
    covariate="cell_type",
    control_group="control",
    key_added="all_DEGs",
)

# print(adata.uns["all_DEGs"].keys())
# print(adata.uns["all_DEGs"]["MCF7_SRT3025"])
# print(len(adata.uns["all_DEGs"]["MCF7_SRT3025"]))
# print(adata.var_names)
# idx = np.where(adata.var_names == "IGF1")
# print(idx[0].item())


colunms_name = [i for i in range(50)]
deg_gene = pd.DataFrame(columns=colunms_name)
for key, value in adata.uns["all_DEGs"].items():
    idx = np.where(adata.var_names.isin(value))
    idx = idx[0].tolist()

    temp = pd.DataFrame([idx], index=[key], columns=colunms_name)
    deg_gene = pd.concat([deg_gene, temp])

deg_gene.to_csv("./datasets/preprocess/sciplex4/chemcpa_deg_gene.csv")

print(adata.var.highly_variable.value_counts())
sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=False)
print(adata.var.highly_variable.value_counts())
idx = np.where(adata.var.highly_variable)[0]
print(len(idx.tolist()[:200]))

pickle.dump(
    idx.tolist()[:200],
    open("./datasets/preprocess/sciplex4/chemcpa_mse_hvg_idx.pkl", "wb"),
)
