import pandas as pd

treat_train_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_treat_train.csv"
)
print("train treat shape:", treat_train_data.shape)

# print(train_data)
print("train treat 药物个数:", len(list(treat_train_data["SMILES"].unique())))
print("train treat 细胞系个数:", len(list(treat_train_data["cell_type"].unique())))

treat_valid_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_treat_valid.csv"
)
print("valid treat shape:", treat_valid_data.shape)

print("valid treat 药物个数:", len(list(treat_valid_data["SMILES"].unique())))
print("valid treat 细胞系个数:", len(list(treat_valid_data["cell_type"].unique())))

treat_test_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_treat_test.csv"
)
print("test treat shape:", treat_test_data.shape)

print("test treat 药物个数:", len(list(treat_test_data["SMILES"].unique())))
print("test treat 细胞系个数:", len(list(treat_test_data["cell_type"].unique())))
print(list(treat_test_data["cov_drug"].unique()))

pert_list = list(treat_train_data["SMILES"].unique())
pert_list.extend(list(treat_valid_data["SMILES"].unique()))
pert_list.extend(list(treat_test_data["SMILES"].unique()))
pert_list = list(set(pert_list))
print("treat 总共药物数量:", len(pert_list))

cell_list = list(treat_train_data["cell_type"].unique())
cell_list.extend(list(treat_valid_data["cell_type"].unique()))
cell_list.extend(list(treat_test_data["cell_type"].unique()))
cell_list = list(set(cell_list))
print("treat 总共细胞系数量:", len(cell_list))

print(
    "treat 总共样本数量:",
    treat_train_data.shape[0] + treat_valid_data.shape[0] + treat_test_data.shape[0],
)

# -------------------------

control_train_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_control_train.csv"
)
print("control train shape:", control_train_data.shape)

# print(train_data)
print("control train 细胞系个数:", len(list(control_train_data["cell_type"].unique())))

control_valid_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_control_valid.csv"
)
print("control valid shape:", control_valid_data.shape)

print("control valid 细胞系个数:", len(list(control_valid_data["cell_type"].unique())))

control_test_data = pd.read_csv(
    "./datasets/preprocess/sciplex3/chemcpa_trapnell_control_test.csv"
)
print("control test shape:", control_test_data.shape)

print("control test 细胞系个数:", len(list(control_test_data["cell_type"].unique())))

cell_list = list(control_train_data["cell_type"].unique())
cell_list.extend(list(control_valid_data["cell_type"].unique()))
cell_list.extend(list(control_test_data["cell_type"].unique()))
cell_list = list(set(cell_list))
print("control 总共细胞系数量:", len(cell_list))

print(
    "control 总共样本数量:",
    control_train_data.shape[0]
    + control_valid_data.shape[0]
    + control_test_data.shape[0],
)

print(
    "treat + control 总共样本数量:",
    treat_train_data.shape[0]
    + treat_valid_data.shape[0]
    + treat_test_data.shape[0]
    + control_train_data.shape[0]
    + control_valid_data.shape[0]
    + control_test_data.shape[0],
)

print("train 样本个数", treat_train_data.shape[0] + control_train_data.shape[0])
print("valid 样本个数", treat_valid_data.shape[0] + control_valid_data.shape[0])
print("test 样本个数", treat_test_data.shape[0] + control_test_data.shape[0])
