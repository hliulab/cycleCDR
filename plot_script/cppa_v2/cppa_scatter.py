import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


control = pd.read_csv("./datasets/cppa/control_perturbed_data_test.csv")
control = control.set_index("id")


treat = pd.read_csv("./datasets/cppa/treat_perturbed_data_test.csv")

baseline_r2_dict = {}

# 遍历 data 中的每一行
for i in range(len(treat)):
    temp_treat = treat.iloc[i, :]
    id = temp_treat["id"]
    temp_treat = temp_treat.drop(["id"])
    temp_treat = temp_treat.to_list()

    temp_control = control.iloc[i, :]
    temp_control = temp_control.to_list()

    res = r2_score(temp_treat, temp_control)

    baseline_r2_dict[id] = res


with open("./results/plot_data/cycleCDR_cppa_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cycleCDR_cppa_cuda%3A1.pkl", 'rb') as f:
    data2 = pickle.load(f)

treats_r2_cpa_dict = data1["test_res"]["treats_r2_cpa_dict"]
treats_r2_cpa_dict.update(data2["test_res"]["treats_r2_cpa_dict"])

x = []
y = []
for id in treats_r2_cpa_dict.keys():
    x.append(baseline_r2_dict[id])
    y.append(treats_r2_cpa_dict[id])


x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
ax.scatter(x, y, s=7)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='dotted', c='k')
ax.set_title("RPPA", fontsize=15)
ax.set_xlabel("Baseline r2 score", fontsize=15)
ax.set_ylabel("cycleCDR r2 score", fontsize=15)
plt.show()
