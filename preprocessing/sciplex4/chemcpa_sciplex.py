import numpy as np
import scanpy as sc
import pandas as pd
from scanpy import AnnData


adata_cpa = sc.read(
    './datasets/preprocess/sciplex4/sciplex4_filtered_genes_for_split.h5ad'
)
print(adata_cpa.obs.shape)
print(adata_cpa.obs.split_ood_finetuning.value_counts())
exit()

adata_cpa = adata_cpa[adata_cpa.obs.control.isin([1]) | adata_cpa.obs.dose.isin([10000])]

# 剔除 adata 中基因表达大于 0 的个数小于 50 的样本
print("剔除 adata 中基因表达大于 0 的个数小于 50 的样本")
print(adata_cpa.X.shape)
print(adata_cpa.obs.shape)
adata_cpa = adata_cpa[adata_cpa.X.getnnz(axis=1) >= 50, :]
print(adata_cpa.X.shape)
print(adata_cpa.obs.shape)
print(adata_cpa.obs.split_ood_finetuning.value_counts())

# adata_cpa = adata_cpa[adata_cpa.obs.split_ood_finetuning.isin(['test'])]
# print(adata_cpa.obs.control.value_counts())

rng = np.random.default_rng(100)


def split_adata(adata: AnnData, split_size=1000):
    adata_res = []
    one_index = adata.obs.shape[0] // split_size
    if one_index * split_size < adata.obs.shape[0]:
        one_index += 1
    for i in range(one_index):
        if i != one_index - 1:
            adata_res.append(
                adata[adata.obs.iloc[i * split_size : (i + 1) * split_size].index]
            )
        else:
            adata_res.append(adata[adata.obs.iloc[i * split_size :].index])

    return adata_res


set_names = ["train", "valid", "test"]

for set_name in set_names:
    if set_name == "test":
        adata_control = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["test"]))
            & (adata_cpa.obs["control"].isin([1]))
        ]
        adata_treat = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["test"]))
            & (adata_cpa.obs.control.isin([0]))
        ]

    elif set_name == "valid":
        adata_control = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["valid"]))
            & (adata_cpa.obs["control"].isin([1]))
        ]
        adata_treat = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["valid"]))
            & (adata_cpa.obs.control.isin([0]))
        ]

    elif set_name == "train":
        adata_control = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["train"]))
            & (adata_cpa.obs["control"].isin([1]))
        ]
        adata_treat = adata_cpa[
            (adata_cpa.obs["split_ood_finetuning"].isin(["train"]))
            & (adata_cpa.obs.control.isin([0]))
        ]

    # sciplex3 的 dose 是以 nm 为单位的，10 um = 10000 nm
    adata_treat = adata_treat[adata_treat.obs.dose.isin([10000])]

    cell_ids = np.unique(adata_treat.obs.cell_type).tolist()

    for cell_id in cell_ids:
        cell_control = adata_control[adata_control.obs.cell_type == cell_id]
        cell_treat = adata_treat[adata_treat.obs.cell_type == cell_id]

        treat_length = cell_treat.obs.shape[0]

        row_sequence = np.arange(cell_control.X.A.shape[0])

        if cell_treat.obs.shape[0] - row_sequence.shape[0] > 0:
            temp = rng.choice(
                row_sequence,
                cell_treat.obs.shape[0] - row_sequence.shape[0],
                replace=True,
            )
            row_sequence = np.concatenate((row_sequence, temp))

        treat_columns = [i for i in range(cell_treat.X.A.shape[1])]
        treat_columns.extend(["cell_type", "SMILES", "cov_drug"])
        control_columns = [i for i in range(cell_treat.X.A.shape[1])]
        control_columns.extend(["cell_type"])

        split_size = 1000
        cell_control = split_adata(cell_control, split_size)
        cell_treat = split_adata(cell_treat, split_size)

        treat_df = []
        control_df = []

        temp_treat_df = []
        temp_control_df = []

        for i in range(treat_length):
            index = i - ((i // split_size) * split_size)

            treat_row = cell_treat[i // split_size].X.A[index, :].tolist()
            smile = cell_treat[i // split_size].obs.SMILES.iloc[index]
            cov_drug = cell_treat[i // split_size].obs.cov_drug.iloc[index]
            treat_row.extend([cell_id, smile, cov_drug])
            treat_temp = pd.DataFrame([treat_row], columns=treat_columns)
            temp_treat_df.append(treat_temp)

            control_index = row_sequence[i]
            control_index_temp = control_index - (
                (control_index // split_size) * split_size
            )
            control_row = (
                cell_control[control_index // split_size]
                .X.A[control_index_temp, :]
                .tolist()
            )
            control_row.extend([cell_id])
            control_temp = pd.DataFrame([control_row], columns=control_columns)

            temp_control_df.append(control_temp)

            if len(temp_treat_df) == 1000:
                temp_treat_df = pd.concat(temp_treat_df, ignore_index=True)
                temp_control_df = pd.concat(temp_control_df, ignore_index=True)
                treat_df.append(temp_treat_df)
                control_df.append(temp_control_df)
                temp_treat_df = []
                temp_control_df = []

        if len(temp_treat_df) != 0:
            temp_treat_df = pd.concat(temp_treat_df, ignore_index=True)
            temp_control_df = pd.concat(temp_control_df, ignore_index=True)
            treat_df.append(temp_treat_df)
            control_df.append(temp_control_df)

        treat_df = pd.concat(treat_df, ignore_index=True)
        control_df = pd.concat(control_df, ignore_index=True)

        treat_df.to_csv(
            f"./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_{set_name}_{cell_id}.csv",
            index=False,
        )
        control_df.to_csv(
            f"./datasets/preprocess/sciplex4/chemcpa_trapnell_control_{set_name}_{cell_id}.csv",
            index=False,
        )

        print(f"Done {set_name} {cell_id}")
        print(treat_df.shape, control_df.shape)

    print(f"Done {set_name}")
