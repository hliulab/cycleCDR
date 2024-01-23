import pickle
import pandas as pd
import scanpy as sc
import torch


cell_type = "rpe1"
pert = "SART1+ctrl"
control = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/control_test.csv")
treat = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/treat_test.csv")

deg = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/deg_gene.csv")
deg = deg.set_index("Unnamed: 0")

control = control[control.cell_type == cell_type]
cov_pert = cell_type + "_" + pert
print(treat.cov_pert.unique())
treat = treat[treat.cov_pert == cov_pert]

deg = deg.loc[cov_pert, :].to_list()
# print(deg)

control = control.iloc[:, deg[:8]].mean(axis=0)
treat = treat.iloc[:, deg[:8]].mean(axis=0)
print("control")

print(control)
print("treat")
print(treat)


# print(treat)
# print(deg)


adata = sc.read('./datasets/row/replogle_rpe1_essential/perturb_processed.h5ad')
# print(adata.var)


print(deg[:8])
genes = adata.var.iloc[deg[:8]].gene_name.tolist()


print(genes)

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

data = data1["test_res"]["pred_treats_dict"][cov_pert]
data.extend(data2["test_res"]["pred_treats_dict"][cov_pert])


data = torch.stack(data).mean(axis=0)

print(data[deg[:8]].numpy().tolist())
