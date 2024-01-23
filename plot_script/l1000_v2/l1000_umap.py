import pickle
from numpy import mean, median
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open("./results/plot_data/cyclecpa_l1000_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cyclecpa_l1000_cuda%3A1.pkl", 'rb') as f:
    data2 = pickle.load(f)

test_treats_r2_cpa_dict = []
for cell_drug in data1["test_treats_r2_cpa_dict"].keys():
    test_treats_r2_cpa_dict.append(data1["test_treats_r2_cpa_dict"][cell_drug])

for cell_drug in data2["test_treats_r2_cpa_dict"].keys():
    test_treats_r2_cpa_dict.append(data2["test_treats_r2_cpa_dict"][cell_drug])
print("mean all:", mean(test_treats_r2_cpa_dict))
print("median all:", median(test_treats_r2_cpa_dict))

test_treats_r2_cpa_de_dict = []
for cell_drug in data1["test_treats_r2_cpa_de_dict"].keys():
    test_treats_r2_cpa_de_dict.append(data1["test_treats_r2_cpa_de_dict"][cell_drug])

for cell_drug in data2["test_treats_r2_cpa_de_dict"].keys():
    test_treats_r2_cpa_de_dict.append(data2["test_treats_r2_cpa_de_dict"][cell_drug])
print("mean degs:", mean(test_treats_r2_cpa_de_dict))
print("median degs:", median(test_treats_r2_cpa_de_dict))

test_true_treats = []
for cell_drug in data1["test_true_treats_dict"].keys():
    for value in data1["test_true_treats_dict"][cell_drug]:
        test_true_treats.append(value.tolist())

test_pred_treats = []
for cell_drug in data2["test_pred_treats_dict"].keys():
    for value in data2["test_pred_treats_dict"][cell_drug]:
        test_pred_treats.append(value.tolist())

test_true_treats = np.array(test_true_treats)
test_pred_treats = np.array(test_pred_treats)

data = np.concatenate((test_true_treats, test_pred_treats), axis=0)

target = ["red" for _ in range(len(test_true_treats))] + [
    "grey" for _ in range(len(test_pred_treats))
]


reducer = umap.UMAP(
    random_state=100,
    spread=0.5,
    min_dist=0.05,
    n_neighbors=5,
    learning_rate=1.1,
    n_components=2,
)
embedding = reducer.fit_transform(data)

test_true = embedding[: len(test_true_treats)]
test_pred = embedding[len(test_true_treats) :]

test_true_df = pd.DataFrame(test_true, columns=["x", "true_y"])
# print(test_true_df)

test_pred_df = pd.DataFrame(test_pred, columns=["x", "pred_y"])
# print(test_pred_df)

# 以 0 列为基准，对 test_pred_df 和 test_true_df 进行合并
df = pd.merge(test_pred_df, test_true_df, on="x", how="outer")
# print(df)
df.to_excel("./results/plot_data/umap_l1000_2.xlsx")

data = pd.read_excel("./results/plot_data/umap_l1000_2.xlsx")
# print(data)

# x = np.array(data["x"])
# y = np.array(data["pred_y"])
# y += np.array(data["true_y"])
pred = []
true_list = []
y = []
c = []
for i in range(len(data)):
    if not np.isnan(data["pred_y"][i]):
        # print()
        pred.append((data["x"][i], data["pred_y"][i]))
        # y.append(data["pred_y"][i])
        c.append("#ff7f0e")
    else:
        # y.append(data["true_y"][i])
        true_list.append((data["x"][i], data["true_y"][i]))
        c.append("#1f77b4")

pred = np.array(pred)
true_list = np.array(true_list)
# y = np.array(y)
fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
# ax.scatter(x, y, s=10, c=c, marker="8")
l1 = ax.scatter(pred[:, 0], pred[:, 1], c="#f8766d", s=4, marker="8")
l2 = ax.scatter(true_list[:, 0], true_list[:, 1], c="#c77cff", s=4, marker="8")
# ax.set_title("L1000")
ax.set_xlabel("UMAP1", labelpad=8)
ax.set_ylabel("UMAP2", labelpad=8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(
    handles=[l1, l2],
    labels=['Predicted expression profile', 'True expression profile'],
    loc='upper center',
    frameon=False,
    fontsize=10,
    # prop={'size': 12},
    ncol=2,
    bbox_to_anchor=(0.5, 1.1),
)
plt.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.11)
plt.show()
