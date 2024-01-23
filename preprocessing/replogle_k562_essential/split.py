import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import os
import sys

sys.path.append(os.getcwd())
from preprocessing.replogle_k562_essential.helper import rank_genes_groups_by_cov


adata = sc.read_h5ad("./datasets/row/replogle_k562_essential/perturb_processed.h5ad")

adata.obs["cov_pert"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)

print(adata.obs.shape)
print(adata.X.A.shape)
print(adata.var.shape)

pert_names = set()
for i in range(adata.obs.shape[0]):
    temp = adata.obs.iloc[i]["condition"].split('+')
    for j in temp:
        if j != 'ctrl':
            pert_names.add(j)

pert_names = list(pert_names)
# print(len(pert_names))

pert_symbol_to_enst = {}
for i in range(len(pert_names)):
    temp = adata.var[adata.var['gene_name'] == pert_names[i]]

    if temp.shape[0] == 1:
        pert_symbol_to_enst[pert_names[i]] = temp.index.to_numpy()[0]


# print(len(pert_symbol_to_enst.keys()))
# not_in_pert = []
# for i in range(len(pert_names)):
#     if pert_names[i] not in pert_symbol_to_enst.keys():
#         not_in_pert.append(pert_names[i])

# for t in not_in_pert:
#     adata = adata[
#         adata.obs['condition'].apply(lambda x, t=t: False if t in x else True)
#     ].copy()
# exit()

adata.var["is_pert"] = adata.var.gene_name.isin(pert_symbol_to_enst.keys())

exist_gene = adata.var[adata.var['is_pert'].isin([True])].gene_name.tolist()

exist_condition = []
conditions = adata.obs.condition.unique().tolist()
conditions.remove('ctrl')
for i in conditions:
    if i.split('+')[0] in exist_gene:
        exist_condition.append(i)
temp = exist_condition + ['ctrl']
adata = adata[adata.obs['condition'].isin(temp)].copy()
print(adata.obs.condition.value_counts())

# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)

# 在 var 中增加了 4 列, 其中一列标记了是否为高变基因
sc.pp.highly_variable_genes(adata, n_top_genes=1141, subset=False)

adata = adata[:, (adata.var.is_pert) | (adata.var.highly_variable)].copy()
print("-------------------------------------")
print(adata.obs.shape)
print(adata.X.A.shape)
print(adata.var.shape)

pert = adata.obs.condition.unique().tolist()

adata.obs["split_ood_finetuning"] = "train"

valid_and_test_pert = [
    'RPL23A+ctrl',
    'EIF3B+ctrl',
    'SUPT16H+ctrl',
    'FAU+ctrl',
    'RTF1+ctrl',
    'MED12+ctrl',
    'PRPF31+ctrl',
    'PLRG1+ctrl',
    'SNRNP70+ctrl',
    'PSMC4+ctrl',
    'MED21+ctrl',
    'SNRPE+ctrl',
    'CNOT1+ctrl',
    'SARS+ctrl',
    'PAF1+ctrl',
    'RPL17+ctrl',
    'POLR2H+ctrl',
    'MED4+ctrl',
    'SLU7+ctrl',
    'SNRPD1+ctrl',
    'MED30+ctrl',
    'PSMD2+ctrl',
    'CHMP3+ctrl',
    'HSPA5+ctrl',
    'SNRPA1+ctrl',
    'MED9+ctrl',
    'SFPQ+ctrl',
    'HSPA9+ctrl',
    'INTS2+ctrl',
    'PHB2+ctrl',
    'SUPT5H+ctrl',
    'EIF2S1+ctrl',
    'EIF2S3+ctrl',
    'SNRPF+ctrl',
    'PHB+ctrl',
    'SUPT6H+ctrl',
    'AQR+ctrl',
    'GATA1+ctrl',
    'SRSF3+ctrl',
]

adata.obs.loc[
    adata.obs["condition"].isin(valid_and_test_pert), "split_ood_finetuning"
] = "valid"

valid_control_idx = sc.pp.subsample(
    adata[(adata.obs["control"] == 1)], 0.2, copy=True
).obs.index
adata.obs.loc[valid_control_idx, "split_ood_finetuning"] = "valid"

test_control_idx = sc.pp.subsample(
    adata[adata.obs["split_ood_finetuning"] == "valid"],
    0.5,
    copy=True,
).obs.index
adata.obs.loc[test_control_idx, "split_ood_finetuning"] = "test"

adata.var["position"] = [i + 1 for i in range(adata.var.shape[0])]

# 替换 rank_genes_groups_cov_all 为 index
# ens_to_index = {}
# for i in range(adata.var.shape[0]):
#     temp = adata.var[adata.var["position"] == i + 1]
#     ens_to_index[temp.index.to_numpy()[0]] = temp["position"].to_numpy()[0]


# 药物对一个 cell line 中的 2000 个基因影响最大的 50 基因
rank_genes_groups_by_cov(
    adata,
    groupby="cov_pert",
    covariate="cell_type",
    control_group="control",
    key_added="rank_genes_groups_cov_all",
)

colunms_name = [i for i in range(50)]
deg_gene = pd.DataFrame(columns=colunms_name)
deg_dict = {}
for condition_name, values in adata.uns["rank_genes_groups_cov_all"].items():
    idx = np.where(adata.var_names.isin(values))
    idx = idx[0].tolist()

    deg_dict[condition_name] = idx

    temp = pd.DataFrame([idx], index=[condition_name], columns=colunms_name)
    deg_gene = pd.concat([deg_gene, temp])

deg_gene.to_csv("./datasets/preprocess/replogle_k562_essential/deg_gene.csv")

adata.uns["rank_genes_groups_cov_all"] = deg_dict

adata.write_h5ad(
    './datasets/preprocess/replogle_k562_essential/pre_perturb_processed.h5ad'
)

pert_index = {}
for t in pert_symbol_to_enst.keys():
    pert_index[t] = adata.var[adata.var['gene_name'] == t]["position"].to_numpy()[0]

with open('./datasets/preprocess/replogle_k562_essential/pert_index.pkl', 'wb') as f:
    pickle.dump(pert_index, f)

# print(pert_index)

print(adata.obs["split_ood_finetuning"].value_counts())
adata_treat_train = adata[adata.obs["split_ood_finetuning"] == "train"].copy()
print(adata_treat_train.obs.cell_type.value_counts())
adata_treat_test = adata[adata.obs["split_ood_finetuning"] == "test"].copy()
print(adata_treat_test.obs.cell_type.value_counts())
print(adata.X.A.shape)


print(adata.var.highly_variable.value_counts())
sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=False)
print(adata.var.highly_variable.value_counts())
idx = np.where(adata.var.highly_variable)[0]
print(len(idx.tolist()[:200]))

with open("./datasets/preprocess/replogle_k562_essential/mse_hvg_idx.pkl", "wb") as f:
    pickle.dump(idx.tolist()[:200], f)
