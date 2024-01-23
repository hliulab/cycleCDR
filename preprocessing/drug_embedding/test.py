import pandas as pd
import torch


df = pd.read_parquet("./datasets/l1000/rdkit2D_embedding.parquet")
print(df)