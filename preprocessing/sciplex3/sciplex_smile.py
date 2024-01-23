import scanpy as sc
import pandas as pd

# import numpy as np
from rdkit import Chem


adata_cpa = sc.read("./datasets/preprocess/sciplex3/sciplex3_filtered_genes.h5ad")
# adata_cpa = sc.read("./datasets/sciplex3/sciplex3_genes_lincs.h5ad")

adata_cpi = pd.read_csv(
    "./datasets/row/trapnell_drugs_smiles.csv", names=["condition", "SMILES", "pathway"]
)

drug_dict = dict(zip(adata_cpi.condition, adata_cpi.SMILES))

adata_cpa.obs["condition"] = adata_cpa.obs["condition"].cat.rename_categories(
    {"(+)-JQ1": "JQ1"}
)

adata_cpa.obs["SMILES"] = adata_cpa.obs.condition.map(drug_dict)

adata_cpa.obs["SMILES"] = (
    adata_cpa.obs["SMILES"].astype("category").cat.rename_categories({"": "CS(C)=O"})
)

# print(adata_cpa.obs.loc[adata_cpa.obs.condition == "control", "SMILES"].value_counts())

adata_cpa.obs.SMILES = adata_cpa.obs.SMILES.apply(Chem.CanonSmiles)
print(adata_cpa.obs_keys())

adata_cpa.write("./datasets/preprocess/sciplex3/sciplex3_filtered_genes_for_smiles.h5ad")
