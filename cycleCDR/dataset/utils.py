from scanpy import AnnData


def split_adata(adata: AnnData, split_size=5000):
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
