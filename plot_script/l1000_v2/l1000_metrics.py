import pickle
from numpy import mean, median


with open(
    "./results/plot_data/l1000/6adf97f83acf6453d4a6a4b1070f3754/cycleCDR_l1000_cuda:0.pkl",
    'rb',
) as f:
    data1 = pickle.load(f)

with open(
    "./results/plot_data/l1000/6adf97f83acf6453d4a6a4b1070f3754/cycleCDR_l1000_cuda:1.pkl",
    'rb',
) as f:
    data2 = pickle.load(f)

treats_r2_cpa_list = list(data1["test_res"]["treats_r2_cpa_dict"].values())
treats_r2_cpa_list.extend(list(data2["test_res"]["treats_r2_cpa_dict"].values()))

treats_r2_cpa_de_list = list(data1["test_res"]["treats_r2_cpa_de_dict"].values())
treats_r2_cpa_de_list.extend(list(data2["test_res"]["treats_r2_cpa_de_dict"].values()))

treats_explained_variance_list = list(
    data1["test_res"]["treats_explained_variance_cpa_dict"].values()
)
treats_explained_variance_list.extend(
    list(data2["test_res"]["treats_explained_variance_cpa_dict"].values())
)

treats_explained_variance_de_list = list(
    data1["test_res"]["treats_explained_variance_cpa_de_dict"].values()
)
treats_explained_variance_de_list.extend(
    list(data2["test_res"]["treats_explained_variance_cpa_de_dict"].values())
)


print("r2 all mean", mean(treats_r2_cpa_list))
print("r2 degs mean", mean(treats_r2_cpa_de_list))
print("explained_variance all mean", mean(treats_explained_variance_list))
print("explained_variance degs mean", mean(treats_explained_variance_de_list))
print("r2 all median", median(treats_r2_cpa_list))
print("r2 degs median", median(treats_r2_cpa_de_list))
print("explained_variance all median", median(treats_explained_variance_list))
print("explained_variance degs median", median(treats_explained_variance_de_list))
