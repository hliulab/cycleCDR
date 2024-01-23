import math
import numpy as np
import pandas as pd
from numpy import mean, median

# from sklearn.metrics import r2_score


def compute_pearsonr(y_true, y_pred):
    p = np.corrcoef(y_true, y_pred)

    if math.isnan(p[0, 1]):
        return 0.0
    return p[0, 1]


control = pd.read_csv("./datasets/preprocess/dixit/control_test.csv")
# adata_cpa = sc.read(
#     './datasets/preprocess/sciplex3/sciplex3_filtered_genes_for_split.h5ad'
# )
# control = adata_cpa[
#     adata_cpa.obs.split_ood_finetuning.isin(["test"]) & adata_cpa.obs.control.isin([1])
# ]

# # 将 control 转为 dataframe
# control_dataframe = pd.DataFrame(control.X.toarray())
# control_dataframe.columns = adata_cpa.var.index

# # 将 adata_cpa.obs.cell_type 放入 control 中
# control_dataframe["cell_type"] = control.obs.cell_type.to_numpy()


treat = pd.read_csv("./datasets/preprocess/dixit/treat_test.csv")

de_gene_idx = pd.read_csv("./datasets/preprocess/dixit/deg_gene.csv")
de_gene_idx = de_gene_idx.set_index("Unnamed: 0")

# all_gene_r2_list = []
# de_gene_r2_list = []
k562_all_gene_pearsonr_list = []
k562_de_gene_pearsonr_list = []
rpe1_all_gene_pearsonr_list = []
rpe1_de_gene_pearsonr_list = []

# 遍历 data 中的每一行
for cov_drug, treat_df in treat.groupby("cov_pert"):
    de_idx = de_gene_idx.loc[cov_drug].to_numpy()

    cell_type = treat_df.cell_type.unique()[0]

    # print(control)

    # control_df = control_dataframe[control_dataframe.cell_type.isin([cell_type])]
    control_df = control[control.index.isin(treat_df.index)]

    treat_df = treat_df.drop(["cell_type", "condition", "cov_pert"], axis=1)
    control_df = control_df.drop(["cell_type"], axis=1)

    if len(de_idx) != 50:
        print(de_idx)
        print(cov_drug)
        exit()

    try:
        treat_de_df = treat_df.iloc[:, de_idx]
    except Exception:
        print(de_idx)
        print(cov_drug)
        exit()
    control_de_df = control_df.iloc[:, de_idx]

    treat_mean = mean(treat_df.to_numpy(), axis=0)
    control_mean = mean(control_df.to_numpy(), axis=0)

    # r2 = r2_score(treat_mean, control_mean)

    pearsonr = compute_pearsonr(treat_mean, control_mean)

    if cov_drug.split("_")[0] == "K562":
        k562_all_gene_pearsonr_list.append(pearsonr)
    elif cov_drug.split("_")[0] == "rpe1":
        rpe1_all_gene_pearsonr_list.append(pearsonr)
    else:
        raise ValueError("cell type error: ", cov_drug.split("_")[0])

    treat_de_mean = mean(treat_de_df.to_numpy(), axis=0)
    control_de_mean = mean(control_de_df.to_numpy(), axis=0)

    # de_r2 = r2_score(treat_de_mean, control_de_mean)

    de_pearsonr = compute_pearsonr(treat_de_mean, control_de_mean)

    if cov_drug.split("_")[0] == "K562":
        k562_de_gene_pearsonr_list.append(de_pearsonr)
    elif cov_drug.split("_")[0] == "rpe1":
        rpe1_de_gene_pearsonr_list.append(de_pearsonr)
    else:
        raise ValueError("cell type error: ", cov_drug.split("_")[0])

    # all_gene_r2_list.append(r2)
    # de_gene_r2_list.append(de_r2)
    # all_gene_pearsonr_list.append(pearsonr)
    # de_gene_pearsonr_list.append(de_pearsonr)


# print("all gene mean: ", mean(all_gene_r2_list))
# print("all gene median: ", median(all_gene_r2_list))
# print("deg gene mean: ", mean(de_gene_r2_list))
# print("deg gene median: ", median(de_gene_r2_list))
print("K562 all gene pearsonr mean: ", mean(k562_all_gene_pearsonr_list))
print("K562 all gene pearsonr median: ", median(k562_all_gene_pearsonr_list))
print("K562 deg gene pearsonr mean: ", mean(k562_de_gene_pearsonr_list))
print("K562 deg gene pearsonr median: ", median(k562_de_gene_pearsonr_list))
print("rpe1 all gene pearsonr mean: ", mean(rpe1_all_gene_pearsonr_list))
print("rpe1 all gene pearsonr median: ", median(rpe1_all_gene_pearsonr_list))
print("rpe1 deg gene pearsonr mean: ", mean(rpe1_de_gene_pearsonr_list))
print("rpe1 deg gene pearsonr median: ", median(rpe1_de_gene_pearsonr_list))
