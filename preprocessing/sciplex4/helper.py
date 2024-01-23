import numpy as np
from scanpy import AnnData


def rank_genes_groups_by_cov(
    adata: AnnData,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):
    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        control_group_cov = "_".join([cov_cat, control_group])

        adata_cov = adata[adata.obs[covariate] == cov_cat]

        adata_control = adata_cov[adata_cov.obs[groupby] == control_group_cov].copy()
        adata_treat = adata_cov[adata_cov.obs[groupby] != control_group_cov].copy()

        control = adata_control.X.A.mean(axis=0)

        adata_treat_df = adata_treat.to_df()
        adata_treat_df["cov_drug"] = adata_treat.obs.cov_drug

        for cond, df in adata_treat_df.groupby("cov_drug", observed=False):
            drug_mean = df.iloc[:, :-1].mean()
            # argsort 默认排序是从小到大，这里是筛选和 control 组差异最大的 50 个基因
            de_idx = np.argsort(abs(drug_mean - control))[-n_genes:]

            gene_dict[cond] = drug_mean.index[de_idx].to_numpy().tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return return_dict
