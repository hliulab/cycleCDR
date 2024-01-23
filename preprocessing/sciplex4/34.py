import os
import sys
import scanpy as sc
import anndata as ad
from scanpy import AnnData
from pyensembl import EnsemblRelease


sys.path.append(os.getcwd())


adata4: AnnData = sc.read("./datasets/row/sciplex4/sciplex4.h5ad")
# release 77 uses human reference genome GRCh38
data = EnsemblRelease(77)

ensemb_list = adata4.var_names.tolist()
ensemb_dict = {}
for ensemb in ensemb_list:
    try:
        ensemb_dict[ensemb] = data.gene_by_id(ensemb.split(".")[0]).gene_name
    except Exception:
        break

# 去掉 adata4 中 var 的 index 不在 ensemb_dict 中的key
print(adata4.X.shape)
adata4 = adata4[:, adata4.var.gene.isin(ensemb_dict.keys())]
# 根据 ensemb_dict 将 adata4 中 var 的 gene 列的值转换为 emsembl_dict 中对应 key 的 value
adata4.var["gene"] = adata4.var.gene.map(ensemb_dict)
# 将 adatas 中的 var 的 gene 列设置为 index
adata4.var = adata4.var.set_index("gene")
# 去掉 adata4 中 var 的 index 重复的行
adata4 = adata4[:, ~adata4.var.index.duplicated(keep="first")]
# print(adata4.var_names)
print(adata4.X.shape)

# adatas = []
# for i in range(5):
#     print(i)
#     adatas.append(sc.read("./datasets/row/" + f"sciplex_raw_chunk_{i}.h5ad"))
adata4_gene = adata4.var.index.tolist()

adata3: AnnData = sc.read("sciplex3.h5ad")
adata3 = adata3[:, adata3.var.index.isin(adata4_gene)]
adata3_gene = adata3.var.index.tolist()
adata4 = adata4[:, adata4.var.index.isin(adata3_gene)]
print(adata3.X.shape)
print(adata4.X.shape)
# exit()

# adata3: AnnData = ad.concat(adatas, join="outer", label="batch", index_unique="-")
# sc.write("sciplex3.h5ad", adata3)
print(adata3.X.shape)
print(adata3.var)
adata3.obs = adata3.obs.drop(columns=['product_dose', 'size_factor', 'batch'])


# adata4 中 obs 中的 columns 只保留 adata3 中 obs 中的 columns
adata4.obs = adata4.obs[adata3.obs.columns]
adata3.obs["dose_character"] = adata3.obs.dose_character.astype(str)
adata3.obs["dose_pattern"] = adata3.obs.dose_pattern.astype(str)
# adata3.obs["root_node"] = adata3.obs.root_node.astype(str)

adata4.obs["dose_character"] = adata4.obs.dose_character.astype(str)
adata4.obs["dose_pattern"] = adata4.obs.dose_pattern.astype(str)
# adata4.obs["root_node"] = adata4.obs.root_node.astype(str)

adata: AnnData = ad.concat(
    [adata3, adata4], join="outer", label="batch", index_unique="-"
)
sc.write("sciplex.h5ad", adata)
print(adata.obs.shape)
print(adata.var.shape)
print(adata.X.shape)
print(adata.obs_keys())
print(adata.var_keys())
