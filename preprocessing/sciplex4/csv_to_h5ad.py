import numpy as np
import pandas as pd
import scanpy as sc
import scipy

obs_df = pd.read_csv("./datasets/row/sciplex4/celltypedata.csv", index_col=0)
obs_df = obs_df.drop(columns=["cell"])
print(obs_df.head())
print(obs_df.columns)

X_df = pd.read_csv("./datasets/row/sciplex4/expression_data.csv", index_col=0)
print(X_df.head())

X = scipy.sparse.csr_matrix(X_df.to_numpy().T.astype(np.float32))
print(X.A.shape)

var_df = X_df.iloc[:, :1].copy()
var_df.columns = ["gene"]
# 将 var_df 的 index 的值赋值给 var_df 的 gene 列
var_df["gene"] = var_df.index
var_df.index = var_df.index.astype(str)
print(var_df)

adata = sc.AnnData(X=X, obs=obs_df, var=var_df)
print(adata)
sc.write("./datasets/row/sciplex4/sciplex4.h5ad", adata)
