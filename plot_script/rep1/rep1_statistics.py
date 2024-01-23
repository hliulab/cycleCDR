import pandas as pd

train_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/treat_train.csv")
print("train shape:", train_data.shape)

# print(train_data)
print("train 药物个数:", len(list(train_data["condition"].unique())))
print("train 细胞系个数:", len(list(train_data["cell_type"].unique())))

valid_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/treat_valid.csv")
print("valid shape:", valid_data.shape)

print("valid 药物个数:", len(list(valid_data["condition"].unique())))
print("valid 细胞系个数:", len(list(valid_data["cell_type"].unique())))

test_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/treat_test.csv")
print("test shape:", test_data.shape)

print("test 药物个数:", len(list(test_data["condition"].unique())))
print("test 细胞系个数:", len(list(test_data["cell_type"].unique())))
# print(list(test_data["cov_drug"].unique()))

pert_list = list(train_data["condition"].unique())
pert_list.extend(list(valid_data["condition"].unique()))
pert_list.extend(list(test_data["condition"].unique()))
pert_list = list(set(pert_list))
print("treat 总共药物数量:", len(pert_list))

cell_list = list(train_data["cell_type"].unique())
cell_list.extend(list(valid_data["cell_type"].unique()))
cell_list.extend(list(test_data["cell_type"].unique()))
cell_list = list(set(cell_list))
print("treat 总共细胞系数量:", len(cell_list))

print("treat 总共样本数量:", train_data.shape[0] + valid_data.shape[0] + test_data.shape[0])

# -----------------------------

control_train_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/control_train.csv")
control_valid_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/control_valid.csv")
control_test_data = pd.read_csv("./datasets/preprocess/replogle_rpe1_essential/control_test.csv")

print(
    "总样本量:",
    train_data.shape[0]
    + valid_data.shape[0]
    + test_data.shape[0]
    + control_train_data.shape[0]
    + control_valid_data.shape[0]
    + control_test_data.shape[0],
)

print("train 样本量:", train_data.shape[0] + control_train_data.shape[0])
print("valid 样本量:", valid_data.shape[0] + control_valid_data.shape[0])
print("test 样本量:", test_data.shape[0] + control_test_data.shape[0])