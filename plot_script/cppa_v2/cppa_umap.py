import pickle
from numpy import mean, median
import umap
import numpy as np
import pandas as pd


with open("./results/plot_data/cyclecpa_cppa_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cyclecpa_cppa_cuda%3A1.pkl", 'rb') as f:
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

target = ["red" for _ in range(len(test_true_treats))] + ["grey" for _ in range(len(test_pred_treats))]


reducer = umap.UMAP(random_state=42, spread=1, min_dist=0.1)
embedding = reducer.fit_transform(data)

test_true = embedding[:len(test_true_treats)]
test_pred = embedding[len(test_true_treats):]

test_true_df = pd.DataFrame(test_true, columns=["x", "true_y"])
print(test_true_df)

test_pred_df = pd.DataFrame(test_pred, columns=["x", "pred_y"])
print(test_pred_df)

# 以 0 列为基准，对 test_pred_df 和 test_true_df 进行合并
df = pd.merge(test_pred_df, test_true_df, on="x", how="outer")
# print(df)
df.to_excel("./results/plot_data/umap_cppa.xlsx")

