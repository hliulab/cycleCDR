import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


dataset_name = "./datasets/l1000/drug.csv"
smiles_df = pd.read_csv(dataset_name)
smiles_list = smiles_df["canonical_smiles"].values
drug_id_list = smiles_df["pert_id"].values
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
df = pd.DataFrame(
    data=embedding,
    index=drug_id_list,
    columns=[f"latent_{i}" for i in range(embedding.shape[1])],
)

# Drop first feature from generator (RDKit2D_calculated)
df.drop(columns=["latent_0"], inplace=True)

# Drop columns with 0 standard deviation
threshold = 0.01
columns = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
print(f"Deleting columns with std<={threshold}: {columns}")
df.drop(columns=columns, inplace=True)


normalized_df = (df - df.mean()) / df.std()


model_name = "rdkit2D"
fname = f"./datasets/l1000/rdkit2D_embedding.parquet"
normalized_df.to_parquet(fname)



