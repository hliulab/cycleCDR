import pandas as pd
from numpy import mean, median
from sklearn.metrics import r2_score, explained_variance_score


control = pd.read_csv("./datasets/preprocess/cppa/control_perturbed_data_test.csv")
control = control.set_index("id")


treat = pd.read_csv("./datasets/preprocess/cppa/treat_perturbed_data_test.csv")

all_r2_list = []
all_explained_variance_list = []

# 遍历 data 中的每一行
for i in range(len(treat)):
    temp_treat = treat.iloc[i, :]
    temp_treat = temp_treat.drop(["id"])
    temp_treat = temp_treat.to_list()

    temp_control = control.iloc[i, :]
    temp_control = temp_control.to_list()

    r2 = r2_score(temp_treat, temp_control)

    all_r2_list.append(r2)

    explained_variance = explained_variance_score(temp_treat, temp_control)

    all_explained_variance_list.append(explained_variance)


print("all mean: ", mean(all_r2_list))
print("all median: ", median(all_r2_list))
print("all explained_variance mean: ", mean(all_explained_variance_list))
print("all explained_variance median: ", median(all_explained_variance_list))
