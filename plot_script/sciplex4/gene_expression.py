import pickle
import pandas as pd
import scanpy as sc
import torch
from pyensembl import EnsemblRelease


cell_type = "K562"
drug = "Quisinostat"
control = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test.csv")
treat = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test.csv")

deg = pd.read_csv("./datasets/preprocess/sciplex4/chemcpa_deg_gene.csv")
deg = deg.set_index("Unnamed: 0")

control = control[control.cell_type == cell_type]
cov_drug = cell_type + "_" + drug
treat = treat[treat.cov_drug == cov_drug]

deg = deg.loc[cov_drug, :].to_list()
# print(deg)

control = control.iloc[:, deg[:8]].mean(axis=0)
treat = treat.iloc[:, deg[:8]].mean(axis=0)
print("control")

print(control)
print("treat")
print(treat)


# print(treat)
# print(deg)


adata = sc.read('./datasets/preprocess/sciplex4/sciplex4_filtered_genes_for_smiles.h5ad')

data = EnsemblRelease(77)
print(deg[:8])
ensemb_list = adata.var.iloc[deg[:8]].gene.tolist()
ensemb_dict = {}
for ensemb in ensemb_list:
    try:
        ensemb_dict[ensemb] = data.gene_by_id(ensemb.split(".")[0]).gene_name
    except Exception:
        break

print(ensemb_dict)
for k, v in ensemb_dict.items():
    print(v, end=" ")

print()

with open(
    "results/plot_data/sciplex4/3906c2066d6fc368996fac7d07af9377/cycleCDR_sciplex4_cuda:0.pkl",
    "rb",
) as f:
    data1 = pickle.load(f)

print(data1["test_res"]["pred_treats_dict"].keys())
# exit()

with open(
    "results/plot_data/sciplex4/3906c2066d6fc368996fac7d07af9377/cycleCDR_sciplex4_cuda:0.pkl",
    "rb",
) as f:
    data2 = pickle.load(f)

data = data1["test_res"]["pred_treats_dict"][cov_drug]
data.extend(data2["test_res"]["pred_treats_dict"][cov_drug])


data = torch.stack(data).mean(axis=0)

print(data[deg[:8]].numpy().tolist())
