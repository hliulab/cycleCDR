import os
import sys

# import numpy as np
import scanpy as sc

sys.path.append(os.getcwd())


adata = sc.read('./datasets/preprocess/sciplex3/sciplex3_filtered_genes_for_smiles.h5ad')

validation_drugs = [
    "Alvespimycin",
    "Luminespib",
    "Epothilone",
    "Flavopiridol",
    "Quisinostat",
    "Abexinostat",
    "Panobinostat",
    "AR-42",
    "Trichostatin",
    "M344",
    "Resminostat",
    "Belinostat",  # ood
    "Mocetinostat",  # no_ood
    "Pracinostat",  # no_ood
    "Entinostat",  # no_ood
    "Tucidinostat",  # no_ood
    "Tacedinaline",  # no_ood
    "Patupilone",  # no_ood
    "GSK1070916",  # no_ood
    "JNJ-26854165",  # no_ood
    "TAK-901",  # no_ood
    "Dasatinib",  # no_ood
]


ood_drugs = [
    "Dacinostat",  # ood
    "CUDC-907",  # ood
    "Givinostat",  # ood
    "CUDC-101",  # ood
    "Pirarubicin",  # ood
    "Hesperadin",  # ood
    "Tanespimycin",  # ood
    "Trametinib",  # ood
    "Raltitrexed",  # no_ood
]

additional_validation_drugs = [
    "YM155",  # apoptosis
    "Barasertib",  # cell cycle
    "Fulvestrant",  # nuclear receptor
    "Nintedanib",  # tyrosine
    "Rigosertib",  # tyrosine
    "BMS-754807",  # tyrosine
    "KW-2449",  # tyrosine
    "Crizotinib",  # tyrosin
    "ENMD-2076",  # cell cycle
    "Alisertib",  # cell cycle
    "JQ1",  # epigenetic
]

validation_drugs.extend(additional_validation_drugs)

adata.obs["split_ood_finetuning"] = "train"

# ood
adata.obs.loc[adata.obs.condition.isin(ood_drugs), "split_ood_finetuning"] = "ood"

# print(adata.obs.dose.value_counts())
# exit()

# test
validation_cond = (adata.obs.condition.isin(validation_drugs)) & (
    adata.obs.dose.isin([10000])
)
val_idx = sc.pp.subsample(adata[validation_cond], 0.28, copy=True).obs.index
adata.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = adata.obs.split_ood_finetuning == "train"
val_idx = sc.pp.subsample(adata[validation_cond], 0.07, copy=True).obs.index
adata.obs.loc[val_idx, "split_ood_finetuning"] = "test"

validation_cond = (adata.obs.split_ood_finetuning == "train") & (
    adata.obs.control.isin([1])
)
val_idx = sc.pp.subsample(adata[validation_cond], 0.05, copy=True).obs.index
adata.obs.loc[val_idx, "split_ood_finetuning"] = "test"

adata_train = adata[adata.obs.split_ood_finetuning.isin(["train"])]
print(adata_train.obs.control.value_counts())

adata_valid = adata[adata.obs.split_ood_finetuning.isin(["test"])]
print(adata_valid.obs.control.value_counts())

adata_ood = adata[adata.obs.split_ood_finetuning.isin(["ood"])]
print(adata_ood.obs.control.value_counts())

sc.write("./datasets/preprocess/sciplex3/sciplex3_filtered_genes_for_split.h5ad", adata)
