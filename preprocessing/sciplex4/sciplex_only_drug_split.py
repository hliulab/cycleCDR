import os
import sys

# import numpy as np
import scanpy as sc

sys.path.append(os.getcwd())


adata = sc.read('./datasets/preprocess/sciplex4/sciplex4_filtered_genes_for_smiles.h5ad')

ood_drugs = [
    "Dacinostat",
    "CUDC-907",
    "Quisinostat",
    "Panobinostat",
    "Givinostat",
]

adata.obs["split_ood_finetuning"] = "train"

# ood
adata.obs.loc[adata.obs.condition.isin(ood_drugs), "split_ood_finetuning"] = "valid"

validation_cond = (adata.obs.split_ood_finetuning == "train") & (
    adata.obs.control.isin([1])
)
val_idx = sc.pp.subsample(adata[validation_cond], 0.12, copy=True).obs.index
adata.obs.loc[val_idx, "split_ood_finetuning"] = "valid"

validation_cond = adata.obs.split_ood_finetuning.isin(["valid"])
val_idx = sc.pp.subsample(adata[validation_cond], 0.5, copy=True).obs.index
adata.obs.loc[val_idx, "split_ood_finetuning"] = "test"

adata_train = adata[adata.obs.split_ood_finetuning.isin(["train"])]
print(adata_train.obs.control.value_counts())

adata_valid = adata[adata.obs.split_ood_finetuning.isin(["valid"])]
print(adata_valid.obs.control.value_counts())

adata_ood = adata[adata.obs.split_ood_finetuning.isin(["test"])]
print(adata_ood.obs.control.value_counts())

sc.write("./datasets/preprocess/sciplex4/sciplex4_filtered_genes_for_split.h5ad", adata)
