import scanpy as sc
from numpy import mean
from sklearn.metrics import r2_score


adata_ref = sc.read_h5ad("./datasets/row/dixit/perturb_processed.h5ad")

adata_ref.obs["cov_pert"] = (
    adata_ref.obs.cell_type.astype(str) + "_" + adata_ref.obs.condition.astype(str)
)

print(adata_ref.obs.cov_pert.value_counts())

cov_perts = adata_ref.obs.cov_pert.unique().tolist()
cov_perts = set(cov_perts)
cov_perts.remove('K562_ctrl')
cov_perts = list(cov_perts)

k562_control = adata_ref[adata_ref.obs.cov_pert.isin(['K562_ctrl'])].X.A
k562_control_mean = mean(k562_control, axis=0)

r2_dict = {}

for cov_pert in cov_perts:
    treat = adata_ref[adata_ref.obs.cov_pert.isin([cov_pert])].X.A
    treat_mean = mean(treat, axis=0)
    if cov_pert.split("_")[0] == 'K562':
        r2 = r2_score(treat_mean, k562_control_mean)

    r2_dict[cov_pert] = r2

k562_pert = []
rpe1_pert = []
for key in r2_dict.keys():
    if key.split("_")[0] == 'K562':
        k562_pert.append("_".join(key.split("_")[1:]))

k562_pert = set(k562_pert)

# new_r2_dict = {}
# for pert in union_pert:
#     new_r2_dict['K562_' + pert] = r2_dict['K562_' + pert]
#     new_r2_dict['rpe1_' + pert] = r2_dict['rpe1_' + pert]


r2_dict = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)
# print(len(new_r2_dict))
# print(len(union_pert))
# for i in new_r2_dict:
#     print(i)
for pert in r2_dict:
    print(pert)

