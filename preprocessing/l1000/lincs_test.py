import scanpy as sc
import numpy as np
import pandas as pd
from rdkit import Chem

# from scipy import sparse


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
    f"From {len(pert_id_unique)} total drugs, {cond.sum()} were not part of the reference dataframe."
)

# 合并数据
adata.obs = adata.obs.reset_index().merge(reference_df, how="left").set_index("index")

# 剔除没有 smiles 的数据
adata.obs.loc[:, "canonical_smiles"] = adata.obs.canonical_smiles.astype("str")
invalid_smiles = adata.obs.canonical_smiles.isin(["-666", "restricted", "nan"])
print(
    f"Among {len(adata)} observations, {100*invalid_smiles.sum()/len(adata):.2f}% ({invalid_smiles.sum()}) have an invalid SMILES string"
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
        except Exception:
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

print(adata.obs.pert_dose.value_counts())
print(adata.obs.pert_dose_unit.value_counts())


# 获取 cell line 原始基因图谱
# print("det 验证开始")
# cell_ids = pd.Series(np.unique(adata.obs.cell_id))
# det_plates = pd.Series(np.unique(adata.obs.det_plate))
# det_wells = pd.Series(np.unique(adata.obs.det_well))

# for i, cell_id in cell_ids.iteritems():
#     for j, det_plate in det_plates.iteritems():
#         for h, det_well in det_wells.iteritems():
#             temp = adata[adata.obs.cell_id.isin([cell_id])]
#             temp = temp[temp.obs.det_plate.isin([det_plate])]
#             if temp.obs.empty:
#                 continue
#             temp = temp[temp.obs.det_well.isin([det_well])]
#             if temp.obs.empty:
#                 continue

#             no_dose = temp[temp.obs.pert_dose.isin([-666])]
#             yes_dose = temp[~temp.obs.pert_dose.isin([-666])]

#             if not no_dose.obs.empty and not yes_dose.obs.empty:
#                 print("cell_id", cell_id)
#                 print("det_plate", det_plate)
#                 print("det_well", det_well)
#                 print("no_dose:", no_dose.obs.shape)
#                 print("yes_dose:", yes_dose.obs.shape)
#                 print("---------------------------------")


# print("++++++++++++++++++++++++++++++++++++++++++++")
# print("rna 验证开始")
# cell_ids = pd.Series(np.unique(adata.obs.cell_id))
# rna_plates = pd.Series(np.unique(adata.obs.rna_plate))
# rna_wells = pd.Series(np.unique(adata.obs.rna_well))

# for i, cell_id in cell_ids.iteritems():
#     for j, rna_plate in rna_plates.iteritems():
#         for h, rna_well in rna_wells.iteritems():
#             temp = adata[adata.obs.cell_id.isin([cell_id])]
#             temp = temp[temp.obs.rna_plate.isin([rna_plate])]
#             if temp.obs.empty:
#                 continue
#             temp = temp[temp.obs.rna_well.isin([rna_well])]
#             if temp.obs.empty:
#                 continue

#             no_dose = temp[temp.obs.pert_dose.isin([-666])]
#             yes_dose = temp[~temp.obs.pert_dose.isin([-666])]

#             if not no_dose.obs.empty and not yes_dose.obs.empty:
#                 print("cell_id", cell_id)
#                 print("rna_plate", rna_plate)
#                 print("rna_well", rna_well)
#                 print("no_dose:", no_dose.obs.shape)
#                 print("yes_dose:", yes_dose.obs.shape)
#                 print("---------------------------------")
