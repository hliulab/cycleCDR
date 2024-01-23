import os
import sys
import numpy as np
import scanpy as sc
from scanpy import AnnData

sys.path.append(os.getcwd())
# from preprocessing.sciplex.helper import rank_genes_groups_by_cov


adata: AnnData = sc.read("./datasets/row/sciplex4/sciplex4.h5ad")

# 均一化 dose
adata.obs["dose_val"] = adata.obs.dose.astype(float) / np.max(
    adata.obs.dose.astype(float)
)


# 除去 control 组只有 17 中药物
adata.obs.loc[adata.obs["product_name"].str.contains("Vehicle"), "dose_val"] = 1.0

adata.obs["product_name"] = [x.split(" ")[0] for x in adata.obs["product_name"]]
adata.obs.loc[
    adata.obs["product_name"].str.contains("Vehicle"), "product_name"
] = "control"

adata.obs["condition"] = adata.obs.product_name.copy()

adata.obs["drug_dose_name"] = (
    adata.obs.condition.astype(str) + "_" + adata.obs.dose_val.astype(str)
)
adata.obs["cov_drug_dose_name"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.drug_dose_name.astype(str)
)
adata.obs["cov_drug"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)

adata.obs["control"] = [
    1 if x == "control_1.0" else 0 for x in adata.obs.drug_dose_name.to_numpy()
]

adata = adata[(adata.obs.control == 1) | (adata.obs.dose.isin([10000]))].copy()

# 剔除 adata 中基因表达值全为 0 的基因
print("剔除 adata 中基因表达值全为 0 的基因")
print(adata.X.A.shape)
print(adata.var.shape)
adata = adata[:, adata.X.sum(axis=0) > 0]
print(adata.X.shape)
print(adata.var.shape)

# 剔除 adata 中基因表达大于 0 的个数小于 300 的样本
print("剔除 adata 中基因表达大于 0 的个数小于 300 的样本")
print(adata.X.shape)
print(adata.obs.shape)
adata = adata[adata.X.getnnz(axis=1) >= 300, :]
print(adata.X.shape)
print(adata.obs.shape)

# 剔除 adata 中基因表达大于 0 的次数小于 200 的基因
print("剔除 adata 中基因表达大于 0 的次数小于 200 的基因")
print(adata.X.shape)
print(adata.var.shape)
adata = adata[:, adata.X.getnnz(axis=0) >= 200]
print(adata.X.A.shape)
print(adata.var.shape)

# 剔除 adata 中基因表达大于 0 的个数小于 300 的样本
print("剔除 adata 中基因表达大于 0 的个数小于 300 的样本")
print(adata.X.shape)
print(adata.obs.shape)
adata = adata[adata.X.getnnz(axis=1) >= 300, :]
print(adata.X.shape)
print(adata.obs.shape)

print(adata.obs.dose.value_counts())

# # 剔除 adata 中样本数等于 1 的 cov_drug
# print("剔除 adata 中样本数等于 1 的 cov_drug")
# cov_drug_list = ["K562_Bisindolylmaleimide", "K562_Flavopiridol"]
# print(adata.X.shape)
# print(adata.obs.shape)
# adata = adata[~adata.obs.cov_drug.isin(cov_drug_list), :]
# print(adata.X.shape)
# print(adata.obs.shape)

print("normalize ing ...")

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

# print("开始筛选高变基因")

# # 根据药物对一个 cell line 的基因的影响进行排序
# rank_genes_groups_by_cov(
#     adata,
#     groupby="cov_drug",
#     covariate="cell_type",
#     control_group="control",
#     key_added="all_DEGs",
#     n_genes=200,
# )


# de_gene = set()
# for key, value in adata.uns["all_DEGs"].items():
#     idx = np.where(adata.var_names.isin(value))
#     idx = idx[0].tolist()

#     de_gene = de_gene.union(set(idx))

# de_gene = list(de_gene)
# print("高变基因总数:", len(de_gene))

# # 保留 de_gene 中的基因
# adata = adata[:, de_gene]


sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)

# 删除其它基因
# adata = adata[:, (adata.var.in_lincs) | (adata.var.highly_variable)].copy()
adata = adata[:, (adata.var.highly_variable)].copy()

print(adata.X.A.shape)
print(adata.var.shape)
print(adata.var.head())
print(adata.X.A)

sc.write("./datasets/preprocess/sciplex4/sciplex4_filtered_genes.h5ad", adata)

print("finished")
