# 调用seaborn
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 调用seaborn自带数据集
# df = sns.load_dataset('iris')

with open("./results/plot_data/sciplex3/c25ffb647f84c9869fc817871c8b5295/cycleCDR_sciplex3_cuda%3A0.pkl", 'rb') as f:
    data1 = pickle.load(f)

with open("./results/plot_data/sciplex3/c25ffb647f84c9869fc817871c8b5295/cycleCDR_sciplex3_cuda%3A1.pkl", 'rb') as f:
    data2 = pickle.load(f)

df = pd.DataFrame(columns=["cell_type", "r2", "group"])

for cell_drug in data1["test_res"]["treats_r2_cpa_de_dict"].keys():
    cell_type = cell_drug.split("_")[0]

    # df = df.append({"cell_type": cell_type,
    #                "r2": data1["test_treats_r2_cpa_de_dict"][cell_drug],
    #                "group": "DEGs gene"}, ignore_index=True)
    temp = pd.DataFrame(
        {
            "cell_type": cell_type,
            "r2": data1["test_res"]["treats_r2_cpa_de_dict"][cell_drug],
            "group": "DEGs gene",
        },
        index=[0],
    )
    df = pd.concat([df, temp], ignore_index=True)


for cell_drug in data1["test_res"]["treats_r2_cpa_dict"].keys():
    cell_type = cell_drug.split("_")[0]

    # df = df.append({"cell_type": cell_type,
    #                "r2": data1["test_treats_r2_cpa_dict"][cell_drug],
    #                "group": "all gene"}, ignore_index=True)
    temp = pd.DataFrame(
        {
            "cell_type": cell_type,
            "r2": data1["test_res"]["treats_r2_cpa_dict"][cell_drug],
            "group": "all gene",
        },
        index=[0],
    )

    df = pd.concat([df, temp], ignore_index=True)


for cell_drug in data2["test_res"]["treats_r2_cpa_de_dict"].keys():
    cell_type = cell_drug.split("_")[0]

    # df = df.append({"cell_type": cell_type,
    #                "r2": data2["test_treats_r2_cpa_de_dict"][cell_drug],
    #                "group": "DEGs gene"}, ignore_index=True)
    temp = pd.DataFrame(
        {
            "cell_type": cell_type,
            "r2": data2["test_res"]["treats_r2_cpa_de_dict"][cell_drug],
            "group": "DEGs gene",
        },
        index=[0],
    )
    df = pd.concat([df, temp], ignore_index=True)


for cell_drug in data2["test_res"]["treats_r2_cpa_dict"].keys():
    cell_type = cell_drug.split("_")[0]

    # df = df.append({"cell_type": cell_type,
    #                "r2": data2["test_treats_r2_cpa_dict"][cell_drug],
    #                "group": "all gene"}, ignore_index=True)
    temp = pd.DataFrame(
        {
            "cell_type": cell_type,
            "r2": data2["test_res"]["treats_r2_cpa_dict"][cell_drug],
            "group": "all gene",
        },
        index=[0],
    )
    df = pd.concat([df, temp], ignore_index=True)

cell_types = df.cell_type.unique().tolist()
for cell_type in cell_types:
    temp = df[df.cell_type == cell_type]
    # print(temp)
    all_gene = temp[temp.group == "all gene"]
    de_gene = temp[temp.group == "DEGs gene"]

    for i in range(all_gene.shape[0]):
        drug = all_gene.iloc[i, 0]
        all_r2 = all_gene.iloc[i, 1]
        de_r2 = de_gene.iloc[i, 1]
        print("{}\t{}\t{}".format(drug, all_r2, de_r2))  # noqa: UP032

exit()

ax = sns.boxplot(x='cell_type', y='r2', hue="group", data=df, width=0.3)
# Add jitter with the swarmplot function 添加散点分布
# ax = sns.swarmplot(x='cell_type', y='r2', data=df, color="grey")
# plt.title("Figure 5")
plt.legend(loc='upper right').set_title("")

# plt.xticks(rotation = -25, fontsize=10)
plt.xlabel("")
plt.show()
