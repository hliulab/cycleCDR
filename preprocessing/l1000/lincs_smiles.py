import scanpy as sc
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


full = True

if full:
    adata_in = "./datasets/preprocess/l1000/lincs_full_pp.h5ad"


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

# BRD-M74254599 没有图结构, 需要删除
adata = adata[adata.obs.pert_id != "BRD-M74254599"].copy()

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


# 获取 24 h 测序的未受到干扰的 cell line
adata_24h = adata[adata.obs.pert_time.isin([24])]

# 获取 24h 的 control 数据
adata_control_24h = adata_24h[adata_24h.obs.control.isin([1])]

# 获取 24h 的 control 的 cell_id
cell_ids_in_control_24h = np.unique(adata_control_24h.obs.cell_id).tolist()

# 提取 24 h 的未经过药物处理的基因图谱
adata_control_24h_df = adata_control_24h.to_df()
adata_control_24h_df["cell_id"] = adata_control_24h.obs.cell_id
cell_line_control_24h = pd.DataFrame(columns=adata.var_names)
for cell_id, df in tqdm(adata_control_24h_df.groupby("cell_id"), desc="control:"):
    gene_avg = df.iloc[:, :-1].mean()
    temp = pd.DataFrame(
        gene_avg.to_numpy().reshape(1, -1, order='F'),
        columns=adata.var_names,
        index=[cell_id],
    )
    cell_line_control_24h = pd.concat([cell_line_control_24h, temp], ignore_index=False)


cell_line_control_24h.to_csv("./datasets/preprocess/l1000/l1000_control_24h.csv")


# 提取 pert_time=24h, pert_dose=10 的 cell line
adata_treat_24h_10 = adata_24h[adata_24h.obs.pert_dose.isin([10])]
colunms_name = list(adata.var_names.copy()).extend(["cell_id", "pert_id", "pert_dose"])
cell_line_treat_24h_10 = pd.DataFrame(columns=colunms_name)
for i, cell_id in tqdm(enumerate(cell_ids_in_control_24h), desc="treat:"):
    adata_pert = adata_treat_24h_10[adata_treat_24h_10.obs.cell_id.isin([cell_id])]
    adata_pert_df = adata_pert.to_df()
    adata_pert_df["pert_id"] = adata_pert.obs.pert_id
    for pert_id, df in adata_pert_df.groupby("pert_id"):
        pert_gene = df.iloc[:, :-1].mean()
        temp_df = pd.DataFrame(
            pert_gene.to_numpy().reshape(1, -1, order='F'), columns=adata.var_names
        )
        temp_df["cell_id"] = [cell_id]
        temp_df["pert_id"] = [pert_id]
        temp_df["pert_dose"] = [10]
        cell_line_treat_24h_10 = pd.concat(
            [cell_line_treat_24h_10, temp_df], ignore_index=False
        )


cell_line_treat_24h_10.to_csv("./datasets/preprocess/l1000/l1000_treat_24h_10.csv")


# 提取 smiles
drug_df = adata.obs.loc[:, ["pert_id", "canonical_smiles"]]
drug_df = drug_df.drop_duplicates()
drug_df.to_csv("./datasets/preprocess/l1000/drug.csv")
