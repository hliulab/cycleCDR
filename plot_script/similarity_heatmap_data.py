import scanpy as sc
import numpy as np
import pandas as pd
from rdkit import Chem


full = True

if full:
    adata_in = "./datasets/l1000/lincs_full_pp.h5ad"


adata = sc.read(adata_in)

# 获取药物 id
pert_id_unique = pd.Series(np.unique(adata.obs.pert_id))
print(f"# of unique perturbations: {len(pert_id_unique)}")

reference_df = pd.read_csv(
    "./datasets/row/GSE92742_Broad_LINCS_pert_info.txt", delimiter="\t"
)
# canonical_smiles 存储了药物的 smiles
reference_df = reference_df.loc[
    reference_df.pert_id.isin(pert_id_unique), ["pert_id", "canonical_smiles"]
]

# 计算 L1000 数据集中的药物, 没有 smiles 数据的数量
cond = ~pert_id_unique.isin(reference_df.pert_id)
print(
    f"From {len(pert_id_unique)} total drugs, {cond.sum()} were not part of the reference dataframe."  # noqa: E501
)

# 合并数据
adata.obs = adata.obs.reset_index().merge(reference_df, how="left").set_index("index")

# 剔除没有 smiles 的数据
adata.obs.loc[:, "canonical_smiles"] = adata.obs.canonical_smiles.astype("str")
invalid_smiles = adata.obs.canonical_smiles.isin(["-666", "restricted", "nan"])
print(
    f"Among {len(adata)} observations, {100*invalid_smiles.sum()/len(adata):.2f}% ({invalid_smiles.sum()}) have an invalid SMILES string"  # noqa: E501
)
adata = adata[~invalid_smiles]

# 删除不合理的 smile 数据
def check_smiles(smiles):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        print("invalid SMILES")
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:  # noqa: E722
            print("invalid chemistry")
            return False
    return True


def remove_invalid_smiles(
    dataframe, smiles_key: str = "SMILES", return_condition: bool = False
):
    unique_drugs = pd.Series(np.unique(dataframe[smiles_key]))
    valid_drugs = unique_drugs.apply(check_smiles)
    print(f"A total of {(~valid_drugs).sum()} have invalid SMILES strings")
    _validation_map = dict(zip(unique_drugs, valid_drugs))
    cond = dataframe[smiles_key].apply(lambda x: _validation_map[x])
    if return_condition:
        return cond
    dataframe = dataframe[cond].copy()
    return dataframe


cond = remove_invalid_smiles(
    adata.obs, smiles_key="canonical_smiles", return_condition=True
)
adata = adata[cond]

# 获取 control
adata_control = adata[adata.obs.control.isin([1])]
# adata_control = adata_control[adata_control.obs.cell_id.isin(["MCF7"])]
# adata_control_1 = adata_control[~adata_control.obs.pert_dose.isin([-666])]

adata_control_24h = adata_control[adata_control.obs.pert_time.isin([24])]
print(adata_control_24h.obs.cell_id.value_counts())
# adata_control_24h.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/control_yes_dose_info_24h.csv")
# pd.DataFrame(adata_control_24h.X.A, index=adata_control_24h.obs.index, columns=adata_control_24h.var_names).to_csv("./datasets/plot_similarity/control_yes_dose_gene_24h.csv")  # noqa: E501

# adata_control_2 = adata_control[adata_control.obs.pert_dose.isin([-666])]
# adata_control_24h = adata_control_2[adata_control_2.obs.pert_time.isin([24])]
# adata_control_24h.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/control_no_dose_info_24h.csv")
# pd.DataFrame(adata_control_24h.X.A, index=adata_control_24h.obs.index, columns=adata_control_24h.var_names).to_csv("./datasets/plot_similarity/control_no_dose_gene_24h.csv")  # noqa: E501

exit()

# det pert_time 24
adata_control = adata[adata.obs.cell_id.isin(["MCF7"])]
k = adata_control[adata_control.obs.pert_dose.isin([-666])]
k = k[~k.obs.det_plate.isin(["nan"])]
k = k[k.obs.rna_plate.isin(["nan"])]
k = k[k.obs.pert_time.isin([24])]
print(k.obs.iloc[:,:-9])
k.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/det_no_dose_info_24.csv")
pd.DataFrame(k.X.A, index=k.obs.index, columns=k.var_names).to_csv("./datasets/plot_similarity/det_no_dose_gene_24.csv")  # noqa: E501


t = adata_control[~adata_control.obs.pert_dose.isin([-666])]
t = t[t.obs.pert_id.isin(["BRD-K21680192"])]
t = t[~t.obs.det_plate.isin(["nan"])]
t = t[t.obs.rna_plate.isin(["nan"])]
t = t[t.obs.pert_dose.isin([10])]
t = t[t.obs.pert_time.isin([24])]
print(t.obs.iloc[:,:-9])
# print(t.obs.pert_dose.value_counts())
t.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/det_yes_dose_info_24.csv")
f = pd.DataFrame(t.X.A, index=t.obs.index, columns=t.var_names)
f.to_csv("./datasets/plot_similarity/det_yes_dose_gene_24.csv")

# pert_time 3
adata_control = adata[adata.obs.cell_id.isin(["MCF7"])]
k = adata_control[adata_control.obs.pert_dose.isin([-666])]
k = k[~k.obs.det_plate.isin(["nan"])]
k = k[k.obs.rna_plate.isin(["nan"])]
k = k[k.obs.pert_time.isin([3])]
print(k.obs.iloc[:,:-9])
k.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/det_no_dose_info_3.csv")
pd.DataFrame(k.X.A, index=k.obs.index, columns=k.var_names).to_csv("./datasets/plot_similarity/det_no_dose_gene_3.csv")  # noqa: E501


t = adata_control[~adata_control.obs.pert_dose.isin([-666])]
t = t[t.obs.pert_id.isin(["BRD-K21680192"])]
t = t[~t.obs.det_plate.isin(["nan"])]
t = t[t.obs.rna_plate.isin(["nan"])]
t = t[t.obs.pert_dose.isin([10])]
t = t[t.obs.pert_time.isin([3])]
print(t.obs.iloc[:,:-9])
# print(t.obs.pert_dose.value_counts())
t.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/det_yes_dose_info_3.csv")
f = pd.DataFrame(t.X.A, index=t.obs.index, columns=t.var_names)
f.to_csv("./datasets/plot_similarity/det_yes_dose_gene_3.csv")


# rna pert_time 24
adata_control = adata[adata.obs.cell_id.isin(["MCF7"])]
k = adata_control[adata_control.obs.pert_dose.isin([-666])]
k = k[~k.obs.rna_plate.isin(["nan"])]
k = k[k.obs.det_plate.isin(["nan"])]
k = k[k.obs.pert_time.isin([24])]
print(k.obs.iloc[:,:-9])
k.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/rna_no_dose_info_24.csv")
pd.DataFrame(k.X.A, index=k.obs.index, columns=k.var_names).to_csv("./datasets/plot_similarity/rna_no_dose_gene_24.csv")

t = adata_control[~adata_control.obs.pert_dose.isin([-666])]
t = t[t.obs.pert_id.isin(["BRD-K21680192"])]
t = t[~t.obs.rna_plate.isin(["nan"])]
t = t[t.obs.det_plate.isin(["nan"])]
t = t[t.obs.pert_dose.isin([10])]
t = t[t.obs.pert_time.isin([24])]
print(t.obs.iloc[:,:-9])
# print(t.obs.pert_dose.value_counts())
t.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/rna_yes_dose_info_24.csv")
f = pd.DataFrame(t.X.A, index=t.obs.index, columns=t.var_names)
f.to_csv("./datasets/plot_similarity/rna_yes_dose_gene_24.csv")


# rna pert_time 6
adata_control = adata[adata.obs.cell_id.isin(["MCF7"])]
k = adata_control[adata_control.obs.pert_dose.isin([-666])]
k = k[~k.obs.rna_plate.isin(["nan"])]
k = k[k.obs.det_plate.isin(["nan"])]
k = k[k.obs.pert_time.isin([6])]
print(k.obs.iloc[:,:-9])
k.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/rna_no_dose_info_6.csv")
pd.DataFrame(k.X.A, index=k.obs.index, columns=k.var_names).to_csv("./datasets/plot_similarity/rna_no_dose_gene_6.csv")

t = adata_control[~adata_control.obs.pert_dose.isin([-666])]
t = t[t.obs.pert_id.isin(["BRD-K21680192"])]
t = t[~t.obs.rna_plate.isin(["nan"])]
t = t[t.obs.det_plate.isin(["nan"])]
t = t[t.obs.pert_dose.isin([10])]
t = t[t.obs.pert_time.isin([6])]
print(t.obs.iloc[:,:-9])
# print(t.obs.pert_dose.value_counts())
t.obs.iloc[:,:-9].to_csv("./datasets/plot_similarity/rna_yes_dose_info_6.csv")
f = pd.DataFrame(t.X.A, index=t.obs.index, columns=t.var_names)
f.to_csv("./datasets/plot_similarity/rna_yes_dose_gene_6.csv")


