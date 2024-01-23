import math
import pickle
import numpy as np
import pandas as pd
from numpy import mean
from sklearn.metrics import r2_score, explained_variance_score
from sqlalchemy import column


def compute_pearsonr(y_true, y_pred):
    p = np.corrcoef(y_true, y_pred)

    if math.isnan(p[0, 1]):
        return 0.0
    return p[0, 1]


control = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/control_test.csv")

treat = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/treat_test.csv")

de_gene_idx = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/deg_gene.csv")
de_gene_idx = de_gene_idx.set_index("Unnamed: 0")

# all_gene_r2_list = []
# de_gene_r2_list = []
rpe1_all_gene_r2_list = []
rpe1_de_gene_r2_list = []
rpe1_all_gene_pearsonr_list = []
rpe1_de_gene_pearsonr_list = []
all_gene_explained_variance_list = []
de_gene_explained_variance_list = []

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

    r2 = r2_score(treat_mean, control_mean)
    explained_variance = explained_variance_score(treat_mean, control_mean)

    pearsonr = compute_pearsonr(treat_mean, control_mean)

    if cov_drug.split("_")[0] == "rpe1":
        rpe1_all_gene_pearsonr_list.append(pearsonr)
        rpe1_all_gene_r2_list.append(r2)
        all_gene_explained_variance_list.append(explained_variance)
    else:
        raise ValueError("cell type error: ", cov_drug.split("_")[0])

    treat_de_mean = mean(treat_de_df.to_numpy(), axis=0)
    control_de_mean = mean(control_de_df.to_numpy(), axis=0)

    de_r2 = r2_score(treat_de_mean, control_de_mean)
    de_explained_variance = explained_variance_score(treat_de_mean, control_de_mean)

    de_pearsonr = compute_pearsonr(treat_de_mean, control_de_mean)

    if cov_drug.split("_")[0] == "rpe1":
        rpe1_de_gene_pearsonr_list.append(de_pearsonr)
        rpe1_de_gene_r2_list.append(de_r2)
        de_gene_explained_variance_list.append(de_explained_variance)
    else:
        raise ValueError("cell type error: ", cov_drug.split("_")[0])

with open(
    "results/plot_data/rep1/23370bbf2ccc3f92a4896e52098eb0cf/cycleCDR_rep1_cuda:0.pkl",
    "rb",
) as f:
    data1 = pickle.load(f)

print(data1["test_res"]["pred_treats_dict"].keys())
# exit()

with open(
    "results/plot_data/rep1/23370bbf2ccc3f92a4896e52098eb0cf/cycleCDR_rep1_cuda:0.pkl",
    "rb",
) as f:
    data2 = pickle.load(f)

treats_r2_cpa_list = list(data1["test_res"]["treats_r2_cpa_dict"].values())
treats_r2_cpa_list.extend(list(data2["test_res"]["treats_r2_cpa_dict"].values()))

treats_r2_cpa_de_list = list(data1["test_res"]["treats_r2_cpa_de_dict"].values())
treats_r2_cpa_de_list.extend(list(data2["test_res"]["treats_r2_cpa_de_dict"].values()))

treats_explained_variance_cpa_list = list(
    data1["test_res"]["treats_explained_variance_cpa_dict"].values()
)
treats_explained_variance_cpa_list.extend(
    list(data2["test_res"]["treats_explained_variance_cpa_dict"].values())
)

treats_explained_variance_cpa_de_list = list(
    data1["test_res"]["treats_explained_variance_cpa_de_dict"].values()
)
treats_explained_variance_cpa_de_list.extend(
    list(data2["test_res"]["treats_explained_variance_cpa_de_dict"].values())
)


# rpe1_all_gene_r2_list
# rpe1_de_gene_r2_list
# all_gene_explained_variance_list
# de_gene_explained_variance_list

dataframe = pd.DataFrame(columns=["group", "baseline", "cycleCDR"])

for i in range(len(rpe1_all_gene_r2_list)):
    # print(type(rpe1_all_gene_r2_list[i]))
    # print(type(treats_r2_cpa_list[i]))
    # exit()
    temp = pd.DataFrame(
        [["r2 all", rpe1_all_gene_r2_list[i], treats_r2_cpa_list[i]]],
        columns=["group", "baseline", "cycleCDR"],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)

for i in range(len(rpe1_de_gene_r2_list)):
    temp = pd.DataFrame(
        [["r2 degs", rpe1_de_gene_r2_list[i], treats_r2_cpa_de_list[i]]],
        columns=["group", "baseline", "cycleCDR"],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)

for i in range(len(all_gene_explained_variance_list)):
    temp = pd.DataFrame(
        [["EV all", all_gene_explained_variance_list[i], treats_explained_variance_cpa_list[i]]],
        columns=["group", "baseline", "cycleCDR"],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)

for i in range(len(de_gene_explained_variance_list)):
    temp = pd.DataFrame(
        [["EV degs", de_gene_explained_variance_list[i], treats_explained_variance_cpa_de_list[i]]],
        columns=["group", "baseline", "cycleCDR"],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)

dataframe.to_excel("./results/plot_data/rep1/rep1.xlsx")
