import pickle
import pandas as pd
import scanpy as sc
import torch


cell_type = "K562"
pert = "SLU7+ctrl"
control = pd.read_csv("./datasets/preprocess/replogle_k562_essential/control_test.csv")
treat = pd.read_csv("./datasets/preprocess/replogle_k562_essential/treat_test.csv")

deg = pd.read_csv("./datasets/preprocess/replogle_k562_essential/deg_gene.csv")
deg = deg.set_index("Unnamed: 0")

control = control[control.cell_type == cell_type]
cov_pert = cell_type + "_" + pert
print(treat.cov_pert.unique())
treat = treat[treat.cov_pert == cov_pert]

deg = deg.loc[cov_pert, :].to_list()
# print(deg)

control = control.iloc[:, deg[:8]].mean(axis=0)
treat = treat.iloc[:, deg[:8]]
print("control")

print(control)
print("treat")
for i in range(treat.shape[0]):
    print(treat.iloc[i, 1])


# print(treat)
# print(deg)


adata = sc.read('./datasets/row/replogle_k562_essential/perturb_processed.h5ad')
# print(adata.var)


print(deg[:8])
genes = adata.var.iloc[deg[:8]].gene_name.tolist()


print(genes)

with open(
    "results/plot_data/rep1k562/b8ffed0e929ad182953f302bf3e82084/cycleCDR_rep1k562_cuda:0.pkl",
    "rb",
) as f:
    data1 = pickle.load(f)

print(data1["test_res"]["pred_treats_dict"].keys())
# exit()

with open(
    "results/plot_data/rep1k562/b8ffed0e929ad182953f302bf3e82084/cycleCDR_rep1k562_cuda:0.pkl",
    "rb",
) as f:
    data2 = pickle.load(f)

data = data1["test_res"]["pred_treats_dict"][cov_pert]
data.extend(data2["test_res"]["pred_treats_dict"][cov_pert])


data = torch.stack(data)
for i in range(data.shape[0]):
    print(data[i, 1].item())

# print(data[deg[:8]].numpy().tolist())
