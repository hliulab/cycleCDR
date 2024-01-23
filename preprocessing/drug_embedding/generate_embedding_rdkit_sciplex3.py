import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from joblib import Parallel, delayed
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


dataset_name = "./datasets/row/trapnell_cpa.h5ad"
adata = sc.read(dataset_name)
smiles_list = list(adata.obs.SMILES.value_counts().index)

print(f"Number of smiles strings: {len(smiles_list)}")

generator = MakeGenerator(("RDKit2D",))
# for name, numpy_type in generator.GetColumns():
#     print(f"{name}({numpy_type.__name__})")


n_jobs = 16
data = Parallel(n_jobs=n_jobs)(
    delayed(generator.process)(smiles)
    for smiles in tqdm(smiles_list, position=0, leave=True)
)


embedding = np.array(data)

# Check `nans` and `infs`
drug_idx, feature_idx = np.where(np.isnan(embedding))
print(f"drug_idx:\n {drug_idx}")
print(f"feature_idx:\n {feature_idx}")

drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))

drug_idx = np.concatenate((drug_idx, drug_idx_infs))
feature_idx = np.concatenate((feature_idx, feature_idx_infs))


# Set values to `0`
embedding[drug_idx, feature_idx] = 0


# Save
save_df = pd.DataFrame(
    data=embedding,
    index=smiles_list,
    columns=[f"latent_{i}" for i in range(embedding.shape[1])],
)

# Drop first feature from generator (RDKit2D_calculated)
save_df = save_df.drop(columns=["latent_0"])

# Drop columns with 0 standard deviation
threshold = 0.01
columns = [f"latent_{idx+1}" for idx in np.where(save_df.std() <= threshold)[0]]
print(f"Deleting columns with std<={threshold}: {columns}")
save_df = save_df.drop(columns=columns)


normalized_df = (save_df - save_df.mean()) / save_df.std()


model_name = "rdkit2D"
fname = "./datasets/preprocess/sciplex3/rdkit2D_embedding.parquet"
# with open(fname, "wb") as f:
normalized_df.to_parquet(fname)

print(normalized_df.shape)
