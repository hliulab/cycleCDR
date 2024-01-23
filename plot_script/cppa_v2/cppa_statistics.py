import pandas as pd

pert_list = []
cell_list_sum = []

train_data = pd.read_csv("./datasets/preprocess/cppa/treat_perturbed_data_train.csv")
print("train shape:", train_data.shape)

cell_list = set()
drug_list = set()
for i in range(train_data.shape[0]):
    temp = train_data.loc[i, "id"].split("_")
    cell_list.add(temp[1])
    drug_list.add(temp[-2])

pert_list.extend(list(drug_list))
cell_list_sum.extend(list(cell_list))
# print(train_data)
print("train 药物个数:", len(list(drug_list)))
print("train 细胞个数:", len(list(cell_list)))

valid_data = pd.read_csv("./datasets/preprocess/cppa/treat_perturbed_data_valid.csv")
print("valid shape:", valid_data.shape)

cell_list = set()
drug_list = set()
for i in range(train_data.shape[0]):
    temp = train_data.loc[i, "id"].split("_")
    cell_list.add(temp[1])
    drug_list.add(temp[-2])

pert_list.extend(list(drug_list))
cell_list_sum.extend(list(cell_list))

print("valid 药物个数:", len(list(drug_list)))
print("valid 细胞个数:", len(list(cell_list)))

test_data = pd.read_csv("./datasets/preprocess/cppa/treat_perturbed_data_test.csv")
print("test shape:", test_data.shape)

cell_list = set()
drug_list = set()
for i in range(train_data.shape[0]):
    temp = train_data.loc[i, "id"].split("_")
    cell_list.add(temp[1])
    drug_list.add(temp[-2])

pert_list.extend(list(drug_list))
cell_list_sum.extend(list(cell_list))

print("test 药物个数:", len(list(drug_list)))
print("test 细胞个数:", len(list(cell_list)))


print("treat 总共药物数量:", len(pert_list))
print("treat 总共细胞系数量:", len(cell_list_sum))
print("treat 总共样本数量:", train_data.shape[0] + valid_data.shape[0] + test_data.shape[0])

# ------------------------------

control_train_data = pd.read_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_train.csv"
)
control_valid_data = pd.read_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_valid.csv"
)
control_test_data = pd.read_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_test.csv"
)

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

