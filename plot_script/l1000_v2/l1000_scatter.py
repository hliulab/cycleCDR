import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


control = pd.read_csv("./datasets/l1000/l1000_control_24h.csv")
control.set_index("Unnamed: 0", inplace=True)


treat = pd.read_csv("./datasets/l1000/l1000_treat_24h_10_test.csv")
treat = treat.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

baseline_r2_dict = {}

# 遍历 data 中的每一行
for i in range(len(treat)):
    temp_treat = treat.iloc[i, :]
    cell_id = temp_treat["cell_id"]
    pert_id = temp_treat["pert_id"]
    temp_treat = temp_treat.drop(["cell_id", "pert_id", "pert_dose"])
    temp_treat = temp_treat.to_list()

    temp_control = control.loc[cell_id]
    temp_control = temp_control.to_list()

    res = r2_score(temp_treat, temp_control)

    baseline_r2_dict["{}{}".format(cell_id, pert_id)] = res



with open("./results/plot_data/cycleCDR_l1000_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cycleCDR_l1000_cuda%3A1.pkl", 'rb') as f:
    data2 = pickle.load(f)

treats_r2_cpa_dict = data1["test_res"]["treats_r2_cpa_dict"]
treats_r2_cpa_dict.update(data2["test_res"]["treats_r2_cpa_dict"])
# print(list(treats_r2_cpa_dict.keys())[0])
# print(list(baseline_r2_dict.keys())[0])
# exit()

x = []
y = []
for id in treats_r2_cpa_dict.keys():
    if id not in baseline_r2_dict.keys():
        print(id)
        continue
    x.append(baseline_r2_dict[id])
    y.append(treats_r2_cpa_dict[id])


x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
ax.scatter(x, y, s=2)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='dotted', c='k')
ax.set_title("L1000")
ax.set_xlabel("Baseline R2")
ax.set_ylabel("cycleCDR R2")
plt.show()
