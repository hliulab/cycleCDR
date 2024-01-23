import pandas as pd

train_data = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_train.csv")
print("treat train shape:", train_data.shape)

# print(train_data)
print("treat train 药物个数:", len(list(train_data["pert_id"].unique())))
print("treat train 细胞个数:", len(list(train_data["cell_id"].unique())))

valid_data = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_valid.csv")
print("treat valid shape:", valid_data.shape)

print("treat valid 药物个数:", len(list(valid_data["pert_id"].unique())))
print("treat valid 细胞个数:", len(list(valid_data["cell_id"].unique())))

test_data = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_test.csv")
print("treat test shape:", test_data.shape)

print("treat test 药物个数:", len(list(test_data["pert_id"].unique())))
print("treat test 细胞个数:", len(list(test_data["cell_id"].unique())))

pert_list = list(train_data["pert_id"].unique())
pert_list.extend(list(valid_data["pert_id"].unique()))
pert_list.extend(list(test_data["pert_id"].unique()))
pert_list = list(set(pert_list))
print("treat总共药物数量:", len(pert_list))

cell_list = list(train_data["cell_id"].unique())
cell_list.extend(list(valid_data["cell_id"].unique()))
cell_list.extend(list(test_data["cell_id"].unique()))
cell_list = list(set(cell_list))
print("treat 总共细胞系数量:", len(cell_list))

# --------------------------

control_data = pd.read_csv("./datasets/preprocess/l1000/l1000_control_24h.csv")
print("总样本量:", train_data.shape[0] * 2 + valid_data.shape[0] * 2+ test_data.shape[0] * 2)
print("train 样本量:", train_data.shape[0] * 2)
print("valid 样本量:", valid_data.shape[0] * 2)
print("test 样本量:", test_data.shape[0] * 2)

