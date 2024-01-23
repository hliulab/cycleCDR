import pandas as pd
from numpy import mean, median
from sklearn.metrics import r2_score, explained_variance_score


control = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test.csv")
treat = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test.csv")

de_gene_idx = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_deg_gene.csv")
de_gene_idx = de_gene_idx.set_index("Unnamed: 0")

all_gene_r2_list = []
de_gene_r2_list = []
all_gene_explained_variance_list = []
de_gene_explained_variance_list = []

# 遍历 data 中的每一行
for cov_drug, treat_df in treat.groupby("cov_drug"):
    de_idx = de_gene_idx.loc[cov_drug].to_numpy()

    control_df = control[control.index.isin(treat_df.index)]
    # control_df = control[control.cell_type.isin([cov_drug.split("_")[0]])]

    treat_df = treat_df.drop(["cell_type", "SMILES", "cov_drug"], axis=1)
    control_df = control_df.drop(["cell_type"], axis=1)

    if len(de_idx) != 50:
        print(de_idx)
        print(cov_drug)
        exit()

    treat_de_df = treat_df.iloc[:, de_idx]
    control_de_df = control_df.iloc[:, de_idx]

    treat_mean = mean(treat_df.to_numpy(), axis=0)
    control_mean = mean(control_df.to_numpy(), axis=0)

    all_gene_r2 = r2_score(treat_mean, control_mean)
    all_gene_explained_variance = explained_variance_score(treat_mean, control_mean)

    treat_de_mean = mean(treat_de_df.to_numpy(), axis=0)
    control_de_mean = mean(control_de_df.to_numpy(), axis=0)

    de_gene_r2 = r2_score(treat_de_mean, control_de_mean)
    de_gene_explained_variance = explained_variance_score(treat_de_mean, control_de_mean)

    all_gene_r2_list.append(all_gene_r2)
    de_gene_r2_list.append(max(de_gene_r2, 0))
    all_gene_explained_variance_list.append(all_gene_explained_variance)
    de_gene_explained_variance_list.append(max(de_gene_explained_variance, 0))


print("all mean: ", mean(all_gene_r2_list))
print("all median: ", median(all_gene_r2_list))
print("deg mean: ", mean(de_gene_r2_list))
print("deg median: ", median(de_gene_r2_list))
print("all explained variance mean: ", mean(all_gene_explained_variance_list))
print("all explained variance median: ", median(all_gene_explained_variance_list))
print("deg explained variance mean: ", mean(de_gene_explained_variance_list))
print("deg explained variance median: ", median(de_gene_explained_variance_list))

# print(treat.iloc[0, :20])
# print(control.iloc[0, :20])
