import pickle

# from numpy import mean, median
import umap
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt


with open("./results/plot_data/cycleCDR_sciplex3_cuda%3A0_no_pretrain.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cycleCDR_sciplex3_cuda%3A1_no_pretrain.pkl", 'rb') as f:
    data2 = pickle.load(f)

test_true_treats = []
for cell_drug in data1["test_res"]["true_treats_dict"].keys():
    for value in data1["test_res"]["true_treats_dict"][cell_drug]:
        test_true_treats.append(value.tolist())

test_pred_treats = []
for cell_drug in data2["test_res"]["pred_treats_dict"].keys():
    for value in data2["test_res"]["pred_treats_dict"][cell_drug]:
        test_pred_treats.append(value.tolist())

test_true_controls = []
for cell_drug in data1["test_res"]["true_controls_dict"].keys():
    for value in data1["test_res"]["true_controls_dict"][cell_drug]:
        test_true_controls.append(value.tolist())

test_true_treats = np.array(test_true_treats)
test_pred_treats = np.array(test_pred_treats)
test_true_controls = np.array(test_true_controls)

data = np.concatenate((test_true_treats, test_pred_treats, test_true_controls), axis=0)

# target = ["red" for _ in range(len(test_true_treats))] + [
#     "grey" for _ in range(len(test_pred_treats))
# ]


reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)

test_true = embedding[: len(test_true_treats)]
test_pred = embedding[len(test_true_treats): len(test_true_treats) + len(test_pred_treats)]
test_true_controls = embedding[len(test_true_treats) + len(test_pred_treats):]

type1 = axes.scatter(test_true[:, 0], test_true[:, 1], s=3, c='red')
type2 = axes.scatter(test_pred[:, 0], test_pred[:, 1], s=3, c='grey')
type3 = axes.scatter(test_true_controls[:, 0], test_true_controls[:, 1], s=3, c='blue')

axes.legend((type1, type2, type3), ('true', 'pred', 'control'), loc='upper right')

plt.show()


# test_true_df = pd.DataFrame(test_true, columns=["x", "true_y"])
# # print(test_true_df)

# test_pred_df = pd.DataFrame(test_pred, columns=["x", "pred_y"])
# # print(test_pred_df)

# # 以 0 列为基准，对 test_pred_df 和 test_true_df 进行合并
# df = pd.merge(test_pred_df, test_true_df, on="x", how="outer")
# # print(df)
# df.to_excel("./results/plot_data/umap_sciplex3.xlsx")


# with open("./results/plot_data/cyclecpa_sciplex3_cuda%3A0_no_gat_gan_nopretrain.pkl", 'rb') as f:
#     data1 = pickle.load(f)

# with open("./results/plot_data/cyclecpa_sciplex3_cuda%3A1_no_gat_gan_nopretrain.pkl", 'rb') as f:
#     data2 = pickle.load(f)

# no_pretrained_test_treats_r2_cpa = []
# for cell_drug in data1["test_treats_r2_cpa_dict"].keys():
#     no_pretrained_test_treats_r2_cpa.append(
#         data1["test_treats_r2_cpa_dict"][cell_drug])

# for cell_drug in data2["test_treats_r2_cpa_dict"].keys():
#     no_pretrained_test_treats_r2_cpa.append(
#         data2["test_treats_r2_cpa_dict"][cell_drug])
# print("no pretrained mean all:", mean(no_pretrained_test_treats_r2_cpa))
# print("no pretrained median all:", median(no_pretrained_test_treats_r2_cpa))


# no_pretrained_test_treats_r2_cpa_de = []
# for cell_drug in data1["test_treats_r2_cpa_de_dict"].keys():
#     no_pretrained_test_treats_r2_cpa_de.append(
#         data1["test_treats_r2_cpa_de_dict"][cell_drug])

# for cell_drug in data2["test_treats_r2_cpa_de_dict"].keys():
#     no_pretrained_test_treats_r2_cpa_de.append(
#         data2["test_treats_r2_cpa_de_dict"][cell_drug])

# print("no pretrained mean degs:", mean(no_pretrained_test_treats_r2_cpa_de))
# print("no pretrained median degs:", median(
#     no_pretrained_test_treats_r2_cpa_de))
