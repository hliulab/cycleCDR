import math
import numpy as np
import pandas as pd
from numpy import mean, median
from sklearn.metrics import r2_score, explained_variance_score

def compute_pearsonr(y_true, y_pred):
    p = np.corrcoef(y_true, y_pred)

    if math.isnan(p[0, 1]):
        return 0.0
    return p[0, 1]


control = pd.read_csv("./datasets/preprocess/l1000/l1000_control_24h.csv")
control = control.set_index("Unnamed: 0")


treat = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_test.csv")
treat = treat.drop(columns=["Unnamed: 0"])

deg_gene = pd.read_csv("./datasets/preprocess/l1000/deg_gene.csv")
deg_gene = deg_gene.set_index("Unnamed: 0")
print(deg_gene.shape)
deg_gene = deg_gene[~deg_gene.index.duplicated(keep='first')]
print(deg_gene.shape)

all_r2_list = []
deg_gene_r2_list = []
all_pearsonr_list = []
deg_gene_pearsonr_list = []
all_explained_variance_list = []
deg_gene_explained_variance_list = []

# 遍历 data 中的每一行
for i in range(len(treat)):
    temp_treat = treat.iloc[i, :]
    cell_id = temp_treat["cell_id"]
    pert_id = temp_treat["pert_id"]
    temp_treat = temp_treat.drop(["cell_id", "pert_id", "pert_dose"])
    temp_treat = temp_treat.to_list()

    temp_control = control.loc[cell_id]
    temp_control = temp_control.to_list()

    explained_variance = explained_variance_score(temp_treat, temp_control)
    all_explained_variance_list.append(explained_variance)

    res = r2_score(temp_treat, temp_control)

    all_r2_list.append(res)

    pearsonr = compute_pearsonr(temp_treat, temp_control)
    all_pearsonr_list.append(pearsonr)

    if pert_id in deg_gene.index:
        deg_gene_idx = deg_gene.loc[pert_id].to_list()
        treat_deg = [temp_treat[i] for i in deg_gene_idx]
        control_deg = [temp_control[i] for i in deg_gene_idx]

        deg_gene_r2 = r2_score(treat_deg, control_deg)
        explained_variance_de = explained_variance_score(treat_deg, control_deg)

        deg_gene_explained_variance_list.append(explained_variance_de)
        deg_gene_r2_list.append(deg_gene_r2)

        deg_gene_pearsonr = compute_pearsonr(treat_deg, control_deg)
        deg_gene_pearsonr_list.append(deg_gene_pearsonr)


print("all r2 mean: ", mean(all_r2_list))
print("all r2 median: ", median(all_r2_list))
print("deg r2 mean: ", mean(deg_gene_r2_list))
print("deg r2 median: ", median(deg_gene_r2_list))
print("all explained_variance mean: ", mean(all_explained_variance_list))
print("all explained_variance median: ", median(all_explained_variance_list))
print("deg explained_variance_de mean: ", mean(deg_gene_explained_variance_list))
print("deg explained_variance_de median: ", median(deg_gene_explained_variance_list))


print("all pearsonr mean: ", mean(all_pearsonr_list))
print("all pearsonr median: ", median(all_pearsonr_list))
print("deg pearsonr mean: ", mean(deg_gene_pearsonr_list))
print("deg pearsonr median: ", median(deg_gene_pearsonr_list))
