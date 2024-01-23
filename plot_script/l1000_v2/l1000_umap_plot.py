import pickle
from tkinter import font
from matplotlib.pylab import f
from numpy import mean, median
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("./results/plot_data/umap_l1000.xlsx")
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
        pred.append(
            (
                data["x"][i],
                data["pred_y"][i],
            )
        )
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
l1 = ax.scatter(pred[:, 0], pred[:, 1], c="#ff7e7e", s=5, marker="8")
l2 = ax.scatter(true_list[:, 0], true_list[:, 1], c="#8080ff", s=5, marker="8")
# ax.set_title("L1000")
ax.set_xlabel("UMAP1", labelpad=8, fontsize=12, fontfamily='Cambria')
ax.set_ylabel("UMAP2", labelpad=8, fontsize=12, fontfamily='Cambria')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(
    handles=[l1, l2],
    labels=['Predicted expression profile', 'True expression profile'],
    loc='upper center',
    frameon=False,
    fontsize=11,
    # fontfamily='Cambria',
    # prop={'size': 12},
    ncol=2,
    bbox_to_anchor=(0.5, 1.1),
)
plt.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.11)
plt.show()
