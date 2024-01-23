# 调用seaborn
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 调用seaborn自带数据集
# df = sns.load_dataset('iris')

with open(
    "./results/plot_data/l1000/6adf97f83acf6453d4a6a4b1070f3754/cycleCDR_l1000_cuda%3A0.pkl",
    'rb',
) as f:
    data1 = pickle.load(f)

with open(
    "./results/plot_data/l1000/6adf97f83acf6453d4a6a4b1070f3754/cycleCDR_l1000_cuda%3A1.pkl",
    'rb',
) as f:
    data2 = pickle.load(f)

dataframe = pd.DataFrame(columns=["drug", "r2", "group"])

for cell_drug in data1["test_res"]["treats_r2_cpa_de_dict"].keys():
    index = cell_drug.find("BRD")
    drug = cell_drug[index:]

    # dataframe = dataframe.append(
    #     {
    #         "drug": drug,
    #         "r2": data1["test_treats_r2_cpa_de_dict"][cell_drug],
    #         "group": "DEGs gene",
    #     },
    #     ignore_index=True,
    # )
    temp = pd.DataFrame(
        {
            "drug": drug,
            "r2": data1["test_res"]["treats_r2_cpa_de_dict"][cell_drug],
            "group": "DEGs gene",
        },
        index=[0],
    )
    # print(drug, data1["test_res"]["treats_r2_cpa_dict"][cell_drug], data1["test_res"]["treats_r2_cpa_de_dict"][cell_drug])
    dataframe = pd.concat([dataframe, temp], ignore_index=True)


for cell_drug in data1["test_res"]["treats_r2_cpa_dict"].keys():
    index = cell_drug.find("BRD")
    drug = cell_drug[index:]

    # dataframe = dataframe.append(
    #     {
    #         "drug": drug,
    #         "r2": data1["test_treats_r2_cpa_dict"][cell_drug],
    #         "group": "all gene",
    #     },
    #     ignore_index=True,
    # )
    temp = pd.DataFrame(
        {
            "drug": drug,
            "r2": data1["test_res"]["treats_r2_cpa_dict"][cell_drug],
            "group": "all gene",
        },
        index=[0],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)


for cell_drug in data2["test_res"]["treats_r2_cpa_de_dict"].keys():
    index = cell_drug.find("BRD")
    drug = cell_drug[index:]

    # dataframe = dataframe.append(
    #     {
    #         "drug": drug,
    #         "r2": data2["test_treats_r2_cpa_de_dict"][cell_drug],
    #         "group": "DEGs gene",
    #     },
    #     ignore_index=True,
    # )
    temp = pd.DataFrame(
        {
            "drug": drug,
            "r2": data2["test_res"]["treats_r2_cpa_de_dict"][cell_drug],
            "group": "DEGs gene",
        },
        index=[0],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)


for cell_drug in data2["test_res"]["treats_r2_cpa_dict"].keys():
    index = cell_drug.find("BRD")
    drug = cell_drug[index:]

    # dataframe = dataframe.append(
    #     {
    #         "drug": drug,
    #         "r2": data2["test_treats_r2_cpa_dict"][cell_drug],
    #         "group": "all gene",
    #     },
    #     ignore_index=True,
    # )
    temp = pd.DataFrame(
        {
            "drug": drug,
            "r2": data2["test_res"]["treats_r2_cpa_dict"][cell_drug],
            "group": "all gene",
        },
        index=[0],
    )
    dataframe = pd.concat([dataframe, temp], ignore_index=True)

# print(df)
# exit()

drug = dataframe.drug.value_counts().index.tolist()[:10]

# brd_to_drug = {}
# brd_to_drug["BRD-K21680192"] = "Plk1 inhibitor"
# brd_to_drug["BRD-K64800655"] = "BMS-387032"
# brd_to_drug["BRD-K67868012"] = "BI 2536"
# brd_to_drug["BRD-K99545815"] = "GSK461364"
# brd_to_drug["BRD-K68174511"] = "RO3280"
# brd_to_drug["BRD-K74710236"] = "HMN-214"
# brd_to_drug["BRD-K60230970"] = "GW843682X"
# brd_to_drug["BRD-K72703948"] = "BI 6727"
# brd_to_drug["BRD-K42918627"] = "Rigosertib"
# brd_to_drug["BRD-K19416115"] = "Volasertib"

# df 中的 drug 列替换为 brd_to_drug 中的值


dataframe = dataframe[dataframe.drug.isin(drug)]
# dataframe.drug = dataframe.drug.replace(brd_to_drug)

for d in drug:
    temp = dataframe[dataframe.drug == d]
    # print(temp)
    all_gene = temp[temp.group == "all gene"]
    de_gene = temp[temp.group == "DEGs gene"]

    for i in range(all_gene.shape[0]):
        drug = all_gene.iloc[i, 0]
        all_r2 = all_gene.iloc[i, 1]
        de_r2 = de_gene.iloc[i, 1]
        print("{}\t{}\t{}".format(drug, all_r2, de_r2))  # noqa: UP032

exit()

# dataframe.to_csv("aaa.csv", index=False)

# print(dataframe)
# exit()

ax = sns.boxplot(x='drug', y='r2', hue="group", data=dataframe, width=0.5)
# Add jitter with the swarmplot function 添加散点分布
# ax = sns.swarmplot(x='drug', y='r2', data=df, color="grey")

plt.legend(loc='lower right').set_title("")

plt.xticks(rotation=88, fontsize=11, fontweight='bold')
plt.xlabel("")
plt.ylabel("R2", fontsize=11, fontweight='bold')
plt.subplots_adjust(bottom=0.27, right=0.9, top=0.9)
# plt.title("Figure 4")
plt.show()

# plt.savefig("./results/plots/l1000_boxplot_drug.pdf")
