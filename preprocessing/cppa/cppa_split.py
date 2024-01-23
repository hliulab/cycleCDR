# 读取 tsv 文件
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# 读取文件夹下的所有文件
def read_files(path):
    data = None
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            try:
                temp = pd.read_csv(path + file, sep="\t")
            except Exception:
                temp = pd.read_csv(path + file, sep="\t", encoding="ISO-8859-1")

            if data is not None:
                data = pd.concat([data, temp], axis=0)
            else:
                data = temp

    data = data.reset_index(drop=True)
    return data


perturbed_data = read_files(
    "./datasets/row/cppa/CPPA_v1.0_Cell_line_perturbed_responses_p0_p1_RPPA/"
)  # noqa: E501

# 删除 perturbed_delta 中的全为 nan 的列
perturbed_data = perturbed_data.dropna(axis=1, how="all")

with open("./datasets/preprocess/cppa/drug_smiles.pkl", "rb") as f:
    drug_smiles = pickle.load(f)


perturbed_data_treat_summ = read_files(
    "./datasets/row/cppa/CPPA_v1.0_Cell_line_perturbed_responses_p0_p1_TreatmentSummary/"
)

perturbed_data_treat_summ = perturbed_data_treat_summ[
    perturbed_data_treat_summ["compound_name_2"].isna()
]
perturbed_data_treat_summ = perturbed_data_treat_summ[
    perturbed_data_treat_summ["compound_name_3"].isna()
]


# 删除 perturbed_data_treat_summ 中 time 列字符长度大于 8 的行
perturbed_data_treat_summ = perturbed_data_treat_summ[
    perturbed_data_treat_summ["time"].apply(lambda x: len(str(x)) <= 8)
]
# 删除 perturbed_data_treat_summ 中 time 列值为 0-1 和 0-2 的行
# perturbed_data_treat_summ = perturbed_data_treat_summ[~perturbed_data_treat_summ["time"].isin(["0-1", "0-2"])]  # noqa: E501
# perturbed_data_treat_summ 中 time 列转为 str 类型
perturbed_data_treat_summ["cell_line_name"] = perturbed_data_treat_summ[
    "cell_line_name"
].apply(lambda x: "-".join(str(x).split(" ")))
perturbed_data_treat_summ["time"] = perturbed_data_treat_summ["time"].apply(
    lambda x: str(x)
)
perturbed_data_treat_summ["2D_3D"] = perturbed_data_treat_summ["2D_3D"].apply(
    lambda x: str(x)
)
perturbed_data_treat_summ["stimuli"] = perturbed_data_treat_summ["stimuli"].apply(
    lambda x: "-".join(str(x).split(" "))
)
perturbed_data_treat_summ["compound_name_1"] = perturbed_data_treat_summ[
    "compound_name_1"
].apply(lambda x: "-".join(str(x).split(" ")))

perturbed_data_treat_summ["dosage_1"] = perturbed_data_treat_summ["dosage_1"].apply(
    lambda x: str(x)
)

control_drug = ["DMSO", "CONTROL", "nan"]

control = perturbed_data_treat_summ[
    perturbed_data_treat_summ["compound_name_1"].isin(control_drug)
]
control["set_cell_time_stimuli_2D_3D_control"] = (
    control.set
    + "_"
    + control.cell_line_name
    + "_"
    + control.time
    + "_"
    + control.stimuli
    + "_"
    + control["2D_3D"]
)
print(control["set_cell_time_stimuli_2D_3D_control"].value_counts())


treat = perturbed_data_treat_summ[
    ~perturbed_data_treat_summ["compound_name_1"].isin(control_drug)
]
treat = treat[treat["compound_name_1"].isin(drug_smiles.keys())]
treat = treat[
    ~treat.dosage_1.isin(
        ["nan", "0", "0.1?M", "1?M", "2?M", "10", "2ug/ml", "1.0", "62.5"]
    )
]

# cppa 的剂量统一为 mM
# treat 的 dosage_1 列以 mM 结尾的行,将 mM 删除
treat["dosage_1"] = treat["dosage_1"].apply(lambda x: x[:-2] if x.endswith("mM") else x)
# treat 的 dosage_1 列以 µM 结尾的行,将 µM 删除,并且将值除以 1000
treat["dosage_1"] = treat["dosage_1"].apply(
    lambda x: str(float(x[:-2]) / 1000)
    if x.endswith("µM") or x.endswith("uM") or x.endswith("UM")
    else x
)
# treat 的 dosage_1 列以 nM 结尾的行,将 nM 删除,并且将值除以 1000000
treat["dosage_1"] = treat["dosage_1"].apply(
    lambda x: str(float(x[:-2]) / 1000000) if x.endswith("nM") or x.endswith("nm") else x
)

treat["set_cell_time_stimuli_2D_3D_drug_dose_treat"] = (
    treat.set
    + "_"
    + treat.cell_line_name
    + "_"
    + treat.time
    + "_"
    + treat.stimuli
    + "_"
    + treat["2D_3D"]
    + "_"
    + treat.compound_name_1
    + "_"
    + treat.dosage_1
)

perturbed_data = perturbed_data.fillna(0)

std = perturbed_data.iloc[:, 1:].std(axis=0)
std = np.asarray(std)
idx = np.where(std <= 0.01)[0].tolist()

# 删除 idx 对应 perturbed_data 中的列
perturbed_data = perturbed_data.drop(perturbed_data.columns[idx], axis=1)

# treat 根据 set_cell_time_stimuli_2D_3D_drug_dose_treat 分组,并且遍历每个组
treat_perturbed_data = []
for name, group in treat.groupby("set_cell_time_stimuli_2D_3D_drug_dose_treat"):
    # 根据 group 的 UID 查询 perturbed_data 中的数据
    uid_perturbed_data = perturbed_data[perturbed_data["UID"].isin(group["UID"])]

    # uid_pertrubed_data 从 1 列开始到最后一列取平均
    uid_perturbed_data = uid_perturbed_data.iloc[:, 1:].mean(axis=0)

    # 将 uid_perturbed_data 转换为 DataFrame, 并且添加到 new_perturbed_data 中
    uid_perturbed_data = pd.DataFrame(uid_perturbed_data).T
    uid_perturbed_data.columns = perturbed_data.columns[1:]
    # 设置 uid_perturbed_data 的 index 为 name
    uid_perturbed_data["id"] = [name]
    treat_perturbed_data.append(uid_perturbed_data)

treat_perturbed_data = pd.concat(treat_perturbed_data, ignore_index=True)


# 重新设置 treat_perturbed_data 的 index 为自增长
treat_perturbed_data = treat_perturbed_data.reset_index(drop=True)
print(treat_perturbed_data.shape)

# treat 根据 set_cell_time_stimuli_2D_3D_control 分组,并且遍历每个组
control_perturbed_data = []
for name, group in control.groupby("set_cell_time_stimuli_2D_3D_control"):
    # 根据 group 的 UID 查询 perturbed_data 中的数据
    uid_perturbed_data = perturbed_data[perturbed_data["UID"].isin(group["UID"])]

    # uid_pertrubed_data 从 1 列开始到最后一列取平均
    uid_perturbed_data = uid_perturbed_data.iloc[:, 1:].mean(axis=0)

    # 将 uid_perturbed_data 转换为 DataFrame, 并且添加到 new_perturbed_data 中
    uid_perturbed_data = pd.DataFrame(uid_perturbed_data).T
    uid_perturbed_data.columns = perturbed_data.columns[1:]
    # 设置 uid_perturbed_data 的 index 为 name
    uid_perturbed_data["id"] = [name]

    control_perturbed_data.append(uid_perturbed_data)

control_perturbed_data = pd.concat(control_perturbed_data, ignore_index=True)


# 重新设置 treat_perturbed_data 的 index 为自增长
control_perturbed_data = control_perturbed_data.reset_index(drop=True)
print(control_perturbed_data.shape)

save_treat_perturbed_data = []
save_control_perturbed_data = []
for set_cell_time_stimuli_2D_3D_drug_dose_treat in treat[
    "set_cell_time_stimuli_2D_3D_drug_dose_treat"
].unique():
    set_cell_time_stimuli_2D_3D = "_".join(
        set_cell_time_stimuli_2D_3D_drug_dose_treat.split("_")[:-2]
    )

    temp_treat = treat_perturbed_data[
        treat_perturbed_data["id"] == set_cell_time_stimuli_2D_3D_drug_dose_treat
    ]
    temp_control = control_perturbed_data[
        control_perturbed_data["id"] == set_cell_time_stimuli_2D_3D
    ]
    if not temp_control.empty and not temp_treat.empty:
        save_treat_perturbed_data.append(temp_treat)
        save_control_perturbed_data.append(temp_control)

save_treat_perturbed_data = pd.concat(save_treat_perturbed_data, ignore_index=True)
save_control_perturbed_data = pd.concat(save_control_perturbed_data, ignore_index=True)


save_control_perturbed_data = save_control_perturbed_data.reset_index(drop=True)
save_treat_perturbed_data = save_treat_perturbed_data.reset_index(drop=True)
assert save_treat_perturbed_data.shape[0] == save_control_perturbed_data.shape[0]

# 判定 save_treat_perturbed_data 和 save_control_perturbed_data 的条件是否相同
for i in range(save_treat_perturbed_data.shape[0]):
    if (
        "_".join(save_treat_perturbed_data.iloc[i]["id"].split("_")[:-2])
        != save_control_perturbed_data.iloc[i]["id"]
    ):
        print(save_control_perturbed_data.iloc[i]["id"])
        print(save_treat_perturbed_data.iloc[i]["id"])
        print("error")
        break


# 判定 save_treat_perturbed_data 和 save_control_perturbed_data 的 columns 是否相同
assert (save_treat_perturbed_data.columns == save_control_perturbed_data.columns).all()

print("treat: ", save_treat_perturbed_data.shape)
print("control: ", save_control_perturbed_data.shape)

r2_dict = {}
for i in range(save_treat_perturbed_data.shape[0]):
    treat = save_treat_perturbed_data.iloc[i, :-1]
    control = save_control_perturbed_data.iloc[i, :-1]

    r2 = r2_score(treat, control)

    r2_dict[i] = r2

sorted_r2_list = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)

key_list = []
for key, value in sorted_r2_list:
    key_list.append(key)

valid_and_test = key_list[int(len(key_list) * 0.71) : int(len(key_list) * 0.91)]

save_treat_perturbed_data_valid = save_treat_perturbed_data.iloc[valid_and_test]
# 将 save_treat_perturbed_data_valid 中的样本从 save_treat_perturbed_data 中删除
save_treat_perturbed_data_train = save_treat_perturbed_data.drop(
    save_treat_perturbed_data_valid.index
)
# 从 save_treat_perturbed_data_valid 中随机抽取 50% 样本存入 save_treat_perturbed_data_test 中
save_treat_perturbed_data_test = save_treat_perturbed_data_valid.sample(
    frac=0.5, random_state=1
)
# 将 save_treat_perturbed_data_test 中的样本从 save_treat_perturbed_data_valid 中删除
save_treat_perturbed_data_valid = save_treat_perturbed_data_valid.drop(
    save_treat_perturbed_data_test.index
)
print("train: ", save_treat_perturbed_data_train.shape)
print("valid: ", save_treat_perturbed_data_valid.shape)
print("test: ", save_treat_perturbed_data_test.shape)


def find(controls: pd.DataFrame, treats: pd.DataFrame):
    res = []
    for i in range(treats.shape[0]):
        id = "_".join(treats.iloc[i]["id"].split("_")[:-2])
        t = controls[controls["id"] == id]
        if not t.empty:
            temp = pd.DataFrame(t.iloc[0]).T
            temp.columns = controls.columns
            res.append(temp)
    res = pd.concat(res, ignore_index=True)
    return res


# 匹配 control
print("匹配 control ing ...")
save_control_perturbed_data_train = find(
    save_control_perturbed_data, save_treat_perturbed_data_train
)
save_control_perturbed_data_valid = find(
    save_control_perturbed_data, save_treat_perturbed_data_valid
)
save_control_perturbed_data_test = find(
    save_control_perturbed_data, save_treat_perturbed_data_test
)
print("匹配 control end ...")

assert (
    save_treat_perturbed_data_train.shape[0] == save_control_perturbed_data_train.shape[0]
)
assert (
    save_treat_perturbed_data_valid.shape[0] == save_control_perturbed_data_valid.shape[0]
)
assert (
    save_treat_perturbed_data_test.shape[0] == save_control_perturbed_data_test.shape[0]
)

print("数据校验开始")
for i in range(save_treat_perturbed_data_train.shape[0]):
    # 获取 save_treat_perturbed_data_train 的最后一列的用 _ 分割的字符串的前面两个元素
    # 用 _ 连接这两个元素
    # 判定这个字符串是否等于 save_control_perturbed_data_train 的最后一列

    if (
        "_".join(save_treat_perturbed_data_train.iloc[i]["id"].split("_")[:-2])
        != save_control_perturbed_data_train.iloc[i]["id"]
    ):
        print(save_treat_perturbed_data_train.iloc[i]["id"])
        print(save_control_perturbed_data_train.iloc[i]["id"])
        print("train 数据校验失败")
        break

for i in range(save_treat_perturbed_data_valid.shape[0]):
    if (
        "_".join(save_treat_perturbed_data_valid.iloc[i]["id"].split("_")[:-2])
        != save_control_perturbed_data_valid.iloc[i]["id"]
    ):
        print(save_treat_perturbed_data_valid.iloc[i]["id"])
        print(save_control_perturbed_data_valid.iloc[i]["id"])
        print("valid 数据校验失败")
        break

for i in range(save_treat_perturbed_data_test.shape[0]):
    if (
        "_".join(save_treat_perturbed_data_test.iloc[i]["id"].split("_")[:-2])
        != save_control_perturbed_data_test.iloc[i]["id"]
    ):
        print(save_treat_perturbed_data_test.iloc[i]["id"])
        print(save_control_perturbed_data_test.iloc[i]["id"])
        print("test 数据校验失败")
        break


print("数据校验结束")

save_control_perturbed_data_train.to_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_train.csv", index=False
)
save_control_perturbed_data_valid.to_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_valid.csv", index=False
)
save_control_perturbed_data_test.to_csv(
    "./datasets/preprocess/cppa/control_perturbed_data_test.csv", index=False
)

save_treat_perturbed_data_train.to_csv(
    "./datasets/preprocess/cppa/treat_perturbed_data_train.csv", index=False
)
save_treat_perturbed_data_valid.to_csv(
    "./datasets/preprocess/cppa/treat_perturbed_data_valid.csv", index=False
)
save_treat_perturbed_data_test.to_csv(
    "./datasets/preprocess/cppa/treat_perturbed_data_test.csv", index=False
)

print("train: ", save_treat_perturbed_data_train.shape)
print("valid: ", save_treat_perturbed_data_valid.shape)
print("test: ", save_treat_perturbed_data_test.shape)

print("finish")
