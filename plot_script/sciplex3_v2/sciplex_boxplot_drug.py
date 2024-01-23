# 调用seaborn
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 调用seaborn自带数据集
# df = sns.load_dataset('iris')

with open("./results/plot_data/cyclecpa_sciplex3_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/cyclecpa_sciplex3_cuda%3A1.pkl", 'rb') as f:
    data2 = pickle.load(f)

df = pd.DataFrame(columns=["drug", "r2"])

test_treats_r2_cpa_dict = []
for cell_drug in data1["test_treats_r2_cpa_dict"].keys():
    drug = cell_drug.split("_")[1]

    temp = pd.DataFrame({"drug": drug,
                   "r2": data1["test_treats_r2_cpa_dict"][cell_drug]}, index=[0])

    # df = df.append({"drug": drug,
    #                "r2": data1["test_treats_r2_cpa_dict"][cell_drug]}, ignore_index=True)
    df = pd.concat([df, temp], ignore_index=True)


for cell_drug in data2["test_treats_r2_cpa_dict"].keys():
    drug = cell_drug.split("_")[1]

    # df = df.append({"drug": drug,
    #                "r2": data2["test_treats_r2_cpa_dict"][cell_drug]}, ignore_index=True)
    temp = pd.DataFrame({"drug": drug,
                   "r2": data2["test_treats_r2_cpa_dict"][cell_drug]}, index=[0])
    df = pd.concat([df, temp], ignore_index=True)

# print(df)
# exit()
ax = sns.boxplot(x='drug', y='r2', data=df, width=0.3)
# Add jitter with the swarmplot function 添加散点分布
ax = sns.swarmplot(x='drug', y='r2', data=df, color="grey")

plt.show()
