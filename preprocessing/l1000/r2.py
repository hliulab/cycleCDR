import pandas as pd
from numpy import mean
from sklearn.metrics import r2_score

treats = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10.csv")
controls = pd.read_csv("./datasets/preprocess/l1000/l1000_control_24h.csv")

print(treats.obs.cov_drug.value_counts())

cov_drugs = treats.obs.cov_drug.unique().tolist()
cov_drugs = set(cov_drugs)
cov_drugs.remove('MCF7_control')
cov_drugs.remove('A549_control')
cov_drugs.remove('K562_control')
cov_drugs = list(cov_drugs)

r2_dict = {}

for cov_drug in cov_drugs:
    treat = treats[treats.obs.cov_drug.isin([cov_drug])].X.A
    treat_mean = mean(treat, axis=0)

    control = controls[controls.obs.cov_drug.isin([cov_drug.split("_")[0]])].X.A
    control_mean = mean(controls, axis=0)

    r2 = r2_score(treat_mean, control_mean)

    r2_dict[cov_drug] = r2


sorted_r2_dict = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_r2_dict)

# for drug in union_drug:
#     if (
#         r2_dict['K562_' + drug] < 0.9
#         and r2_dict['MCF7_' + drug] < 0.9
#         and r2_dict['A549_' + drug] < 0.9
#     ):
#         print(drug)
