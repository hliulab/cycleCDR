import scanpy as sc
from numpy import mean
from sklearn.metrics import r2_score


adata_ref = sc.read_h5ad("./datasets/row/replogle_rpe1_essential/perturb_processed.h5ad")

adata_ref.obs["cov_pert"] = (
    adata_ref.obs.cell_type.astype(str) + "_" + adata_ref.obs.condition.astype(str)
)

print(adata_ref.obs.cov_pert.value_counts())

cov_perts = adata_ref.obs.cov_pert.unique().tolist()
cov_perts = set(cov_perts)
cov_perts.remove('rpe1_ctrl')

cov_perts = list(cov_perts)

rpe1_control = adata_ref[adata_ref.obs.cov_pert.isin(['rpe1_ctrl'])].X.A
rpe1_control_mean = mean(rpe1_control, axis=0)

r2_dict = {}

for cov_pert in cov_perts:
    treat = adata_ref[adata_ref.obs.cov_pert.isin([cov_pert])].X.A
    treat_mean = mean(treat, axis=0)
    if cov_pert.split("_")[0] == 'rpe1':
        r2 = r2_score(treat_mean, rpe1_control_mean)

    r2_dict[cov_pert] = r2

rpe1_pert = []
rpe1_pert = []
for key in r2_dict.keys():
    if key.split("_")[0] == 'rpe1':
        rpe1_pert.append("_".join(key.split("_")[1:]))

rpe1_pert = set(rpe1_pert)

# new_r2_dict = {}
# for pert in union_pert:
#     new_r2_dict['rpe1_' + pert] = r2_dict['rpe1_' + pert]
#     new_r2_dict['rpe1_' + pert] = r2_dict['rpe1_' + pert]


r2_dict = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)
# print(len(new_r2_dict))
# print(len(union_pert))
# for i in new_r2_dict:
#     print(i)
for pert in r2_dict:
    print(pert)

