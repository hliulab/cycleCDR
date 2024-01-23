import scanpy as sc
from numpy import mean
from sklearn.metrics import r2_score

adata_ref = sc.read_h5ad("./datasets/preprocess/sciplex4/sciplex4_filtered_genes.h5ad")

# print(adata_ref.obs.cov_drug.value_counts())

cov_drugs = adata_ref.obs.cov_drug.unique().tolist()
cov_drugs = set(cov_drugs)
cov_drugs.remove('MCF7_control')
cov_drugs.remove('A549_control')
cov_drugs.remove('K562_control')
cov_drugs = list(cov_drugs)

mcf7_control = adata_ref[adata_ref.obs.cov_drug.isin(['MCF7_control'])].X.A
mcf7_control_mean = mean(mcf7_control, axis=0)
a549_control = adata_ref[adata_ref.obs.cov_drug.isin(['A549_control'])].X.A
a549_control_mean = mean(a549_control, axis=0)
k562_control = adata_ref[adata_ref.obs.cov_drug.isin(['K562_control'])].X.A
k562_control_mean = mean(k562_control, axis=0)

r2_dict = {}

for cov_drug in cov_drugs:
    treat = adata_ref[adata_ref.obs.cov_drug.isin([cov_drug])].X.A
    treat_mean = mean(treat, axis=0)
    if cov_drug.split("_")[0] == 'K562':
        r2 = r2_score(treat_mean, k562_control_mean)
    elif cov_drug.split("_")[0] == 'MCF7':
        r2 = r2_score(treat_mean, mcf7_control_mean)
    elif cov_drug.split("_")[0] == 'A549':
        r2 = r2_score(treat_mean, a549_control_mean)
    else:
        raise ValueError(f"{cov_drug.split('_')[0]} is not in ['K562', 'MCF7', 'A549']")

    r2_dict[cov_drug] = r2

mcf7_drug = []
k562_drug = []
a549_drug = []
for key in r2_dict.keys():
    if key.split("_")[0] == 'K562':
        k562_drug.append("_".join(key.split("_")[1:]))
    elif key.split("_")[0] == 'MCF7':
        mcf7_drug.append("_".join(key.split("_")[1:]))
    elif key.split("_")[0] == 'A549':
        a549_drug.append("_".join(key.split("_")[1:]))
    else:
        raise ValueError(f"{key.split('_')[0]} is not in ['K562', 'MCF7', 'A549']")

k562_drug = set(k562_drug)
mcf7_drug = set(mcf7_drug)
a549_drug = set(a549_drug)


# 交集
union_drug = list(k562_drug.intersection(mcf7_drug).intersection(a549_drug))

# new_r2_dict = {}
# for pert in union_pert:
#     new_r2_dict['K562_' + pert] = r2_dict['K562_' + pert]
#     new_r2_dict['rpe1_' + pert] = r2_dict['rpe1_' + pert]


# new_r2_dict = sorted(new_r2_dict.items(), key=lambda x: x[1], reverse=True)
# print(len(new_r2_dict))
# print(len(union_pert))
# for i in new_r2_dict:
#     print(i)
for drug in union_drug:
    if (
        r2_dict['K562_' + drug] < 0.59
        and r2_dict['MCF7_' + drug] < 0.59
        and r2_dict['A549_' + drug] < 0.59
    ):
        print(
            drug
            + " : "
            + str(r2_dict['K562_' + drug])
            + " "
            + str(r2_dict['MCF7_' + drug])
            + " "
            + str(r2_dict['A549_' + drug])
        )
