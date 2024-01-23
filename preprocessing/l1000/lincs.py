import pickle
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import numpy as np
from scipy import sparse


full = True

filename = "./datasets/row/lincs_full.h5ad"
adata = sc.read(filename)

print(adata.obs_keys())
adata_treat = adata[adata.obs.pert_dose.isin([10])]
print(adata_treat.obs.pert_dose.value_counts())
print(adata_treat.obs.pert_dose_unit.value_counts())
# exit()

adata.obs["condition"] = adata.obs["pert_iname"]
adata.obs["condition"] = adata.obs["condition"].str.replace("/", "|")

adata.obs["cell_type"] = adata.obs["cell_id"]
adata.obs["dose_val"] = adata.obs["pert_dose"]
adata.obs["cov_drug_dose_name"] = (
    adata.obs.cell_type.astype(str)
    + "_"
    + adata.obs.condition.astype(str)
    + "_"
    + adata.obs.dose_val.astype(str)
)
adata.obs["cov_drug_name"] = (
    adata.obs.cell_type.astype(str) + "_" + adata.obs.condition.astype(str)
)
adata.obs["eval_category"] = adata.obs["cov_drug_name"]
adata.obs["control"] = (adata.obs["condition"] == "DMSO").astype(int)

control = adata[adata.obs.control.isin([1])]

# 同一个药物, 实验次数小于 5 次的被删除
drug_abundance = adata.obs.condition.value_counts()
suff_drug_abundance = drug_abundance.index[drug_abundance > 5]
adata = adata[adata.obs.condition.isin(suff_drug_abundance)].copy()


colunms_name = [i for i in range(50)]
deg_gene = pd.DataFrame(columns=colunms_name)

# 筛选差异基因
de_genes = {}
# 转化为 Dataframe, adata_df 存放细胞系的基因图谱, 横: cell_id, 纵: gene
adata_df = adata.to_df()
adata_df["condition"] = adata.obs.condition
adata_df["pert_id"] = adata.obs.pert_id
dmso = adata_df[adata_df.condition == "DMSO"].iloc[:, :-2].mean()
for cond, df in tqdm(adata_df.groupby("condition")):
    if cond != "DMSO":
        drug_mean = df.iloc[:, :-2].mean()
        # argsort 默认排序是从小到大，这里是筛选和 control 组差异最大的 50 个基因
        de_50_idx = np.argsort(abs(drug_mean - dmso))[-50:]
        de_genes[cond] = drug_mean.index[de_50_idx].to_numpy()

        pert_id = df.iloc[:, -1].unique()[0]
        temp = pd.DataFrame([de_50_idx.tolist()], index=[pert_id], columns=colunms_name)
        deg_gene = pd.concat([deg_gene, temp])

deg_gene.to_csv("./datasets/preprocess/l1000/deg_gene.csv")


def extract_drug(cond):
    split = cond.split("_")
    if len(split) == 2:
        return split[-1]

    return "_".join(split[1:])


# 保存每个药物影响最大的 50 个基因
adata.uns["rank_genes_groups_cov"] = {
    cat: de_genes[extract_drug(cat)]
    # eval_category = cell_id + "_" + pert_iname
    for cat in adata.obs.eval_category.unique()
    if extract_drug(cat) != "DMSO"
}


try:
    del adata.uns["rank_genes_groups"]  # 暂时未用到
except:  # noqa: E722
    print("All good.")


adata.X = sparse.csr_matrix(adata.X)

adata_out = "./datasets/preprocess/l1000/lincs_full_pp.h5ad"
sc.write(adata_out, adata)
print("save finish!")


sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

# 在 var 中增加了 4 列, 其中一列标记了是否为高变基因
sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=False)

idx = np.where(adata.var.highly_variable)[0]
print(idx.tolist())

pickle.dump(idx.tolist(), open("./datasets/preprocess/l1000/mse_hvg_idx.pkl", "wb"))
