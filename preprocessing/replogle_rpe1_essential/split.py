import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import os
import sys

sys.path.append(os.getcwd())
from preprocessing.replogle_rpe1_essential.helper import rank_genes_groups_by_cov


adata = sc.read_h5ad("./datasets/row/replogle_rpe1_essential/perturb_processed.h5ad")

adata.obs["cov_pert"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)

print(adata.obs.shape)
print(adata.X.A.shape)
print(adata.var.shape)

pert_names = set()
conditions = adata.obs.condition.unique().tolist()
conditions.remove('ctrl')
for i in range(len(conditions)):
    temp = conditions[i].split('+')
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
for i in conditions:
    if i.split('+')[0] in exist_gene:
        exist_condition.append(i)
temp = exist_condition + ['ctrl']
adata = adata[adata.obs['condition'].isin(temp)].copy()
print(adata.obs.condition.value_counts())

# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)

# 在 var 中增加了 4 列, 其中一列标记了是否为高变基因
sc.pp.highly_variable_genes(adata, n_top_genes=945, subset=False)

adata = adata[:, (adata.var.is_pert) | (adata.var.highly_variable)].copy()
print("-------------------------------------")
print(adata.obs.shape)
print(adata.X.A.shape)
print(adata.var.shape)

# pert = adata.obs.condition.unique().tolist()

adata.obs["split_ood_finetuning"] = "train"

valid_and_test_pert = [
    'EIF3B+ctrl',
    'EXOSC9+ctrl',
    'RPL8+ctrl',
    'DDX21+ctrl',
    'SFPQ+ctrl',
    'EIF3E+ctrl',
    'BDP1+ctrl',
    'PUF60+ctrl',
    'RPS7+ctrl',
    'RPL12+ctrl',
    'RPL18A+ctrl',
    'RPS9+ctrl',
    'RPS8+ctrl',
    'RPL34+ctrl',
    'SART1+ctrl',
    'ARCN1+ctrl',
    'NOL11+ctrl',
    'RPL39+ctrl',
    'RPL35A+ctrl',
    'RPS18+ctrl',
    'RPL23A+ctrl',
    'DARS+ctrl',
    'RPS6+ctrl',
    'RPL6+ctrl',
    'CACTIN+ctrl',
    'RPL35+ctrl',
    'PRPF8+ctrl',
    'WEE1+ctrl',
    'SLU7+ctrl',
    'EIF2S2+ctrl',
    'RPL3+ctrl',
    'EIF2B4+ctrl',
    'SCFD1+ctrl',
    'IPO13+ctrl',
    'RPS19+ctrl',
    'BUD13+ctrl',
    'RPL23+ctrl',
    'RPL38+ctrl',
    'IGBP1+ctrl',
    'RPS23+ctrl',
    'AARS+ctrl',
    'CDC5L+ctrl',
    'RPL36+ctrl',
    'BET1+ctrl',
    'RPL17+ctrl',
    'SF3B2+ctrl',
    'PSMD8+ctrl',
    'PSMD11+ctrl',
    'SNRPA1+ctrl',
    'RPL18+ctrl',
    'PSMA2+ctrl',
    'RPL30+ctrl',
    'EIF2S3+ctrl',
    'RAC3+ctrl',
    'PRPF31+ctrl',
    'SNIP1+ctrl',
    'RPL21+ctrl',
    'GPKOW+ctrl',
    'RPL15+ctrl',
    'RPL14+ctrl',
    'CNOT1+ctrl',
    'SRSF3+ctrl',
    'SNRNP70+ctrl',
    'MED22+ctrl',
    'STX5+ctrl',
    'SUPT16H+ctrl',
    'SNRPE+ctrl',
    'SRP68+ctrl',
    'ETF1+ctrl',
    'SNRPD1+ctrl',
    'DHX8+ctrl',
    'SNRNP200+ctrl',
    'FTSJ3+ctrl',
    'PSMB6+ctrl',
    'PSMB1+ctrl',
    'SNRPD3+ctrl',
    'SMU1+ctrl',
    'EIF2S1+ctrl',
    'PSMC4+ctrl',
    'PSMD12+ctrl',
    'TRAPPC8+ctrl',
    'POLR3D+ctrl',
    'PSMB2+ctrl',
    'COPB1+ctrl',
    'PSMC3+ctrl',
    'PRPF6+ctrl',
    'PSMC5+ctrl',
    'PSMB5+ctrl',
    'PSMA4+ctrl',
    'PSMC2+ctrl',
    'PSMD14+ctrl',
    'PSMD6+ctrl',
    'PSMA3+ctrl',
    'PSMB3+ctrl',
    'PSMC1+ctrl',
    'PSMA6+ctrl',
    'PSMD1+ctrl',
    'HSPA5+ctrl',
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

deg_gene.to_csv("./datasets/preprocess/replogle_rpe1_essential/deg_gene.csv")

adata.uns["rank_genes_groups_cov_all"] = deg_dict

adata.write_h5ad(
    './datasets/preprocess/replogle_rpe1_essential/pre_perturb_processed.h5ad'
)

pert_index = {}
for i in range(adata.var.shape[0]):
    gene = adata.var.iloc[i].gene_name
    pert_index[gene] = adata.var.iloc[i].position

# print(len(pert_index.keys()))
# print(len(adata.obs.condition.unique().tolist()))
# print('MED11' in pert_index.keys())
# exit()
j = 0
for i in exist_condition:
    g = i.split('+')[0]
    if g not in pert_index.keys():
        j += 1
        print(g)
        print('---------------------------------')


with open('./datasets/preprocess/replogle_rpe1_essential/pert_index.pkl', 'wb') as f:
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

with open("./datasets/preprocess/replogle_rpe1_essential/mse_hvg_idx.pkl", "wb") as f:
    pickle.dump(idx.tolist()[:200], f)
