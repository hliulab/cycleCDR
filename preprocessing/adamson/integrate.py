import pandas as pd


treat_train_K562_df = pd.read_csv("./datasets/preprocess/adamson/treat_train_K562(?).csv")
# treat_train_rpe1_df = pd.read_csv("./datasets/preprocess/adamson/treat_train_rpe1.csv")

treat_train_df = pd.concat([treat_train_K562_df], ignore_index=True)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_train_df.to_csv(
    "./datasets/preprocess/adamson/treat_train.csv",
    index=False,
)
print(treat_train_df)

control_train_K562_df = pd.read_csv("./datasets/preprocess/adamson/control_train_K562(?).csv")
# control_train_rpe1_df = pd.read_csv(
#     "./datasets/preprocess/adamson/control_train_rpe1.csv"
# )

control_train_df = pd.concat(
    [control_train_K562_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_train_df.to_csv(
    "./datasets/preprocess/adamson/control_train.csv",
    index=False,
)
print(control_train_df)

# if treat_train_df.shape[0] != control_train_df.shape[0]:
#     print("nani1")
#     exit()

treat_valid_K562_df = pd.read_csv("./datasets/preprocess/adamson/treat_valid_K562(?).csv")
# treat_valid_rpe1_df = pd.read_csv("./datasets/preprocess/adamson/treat_valid_rpe1.csv")

treat_valid_df = pd.concat([treat_valid_K562_df], ignore_index=True)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_valid_df.to_csv(
    "./datasets/preprocess/adamson/treat_valid.csv",
    index=False,
)
print(treat_valid_df)


control_valid_K562_df = pd.read_csv("./datasets/preprocess/adamson/control_valid_K562(?).csv")
# control_valid_rpe1_df = pd.read_csv("./datasets/preprocess/adamson/control_valid_rpe1.csv")

control_valid_df = pd.concat(
    [control_valid_K562_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_valid_df.to_csv(
    "./datasets/preprocess/adamson/control_valid.csv",
    index=False,
)
print(control_valid_df)

# if treat_valid_df.shape[0] != control_valid_df.shape[0]:
#     print("nani2")
#     exit()

treat_test_K562_df = pd.read_csv("./datasets/preprocess/adamson/treat_test_K562(?).csv")
# treat_test_rpe1_df = pd.read_csv("./datasets/preprocess/adamson/treat_test_rpe1.csv")

treat_test_df = pd.concat([treat_test_K562_df], ignore_index=True)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_test_df.to_csv(
    "./datasets/preprocess/adamson/treat_test.csv",
    index=False,
)
print(treat_test_df)

control_test_K562_df = pd.read_csv("./datasets/preprocess/adamson/control_test_K562(?).csv")
# control_test_rpe1_df = pd.read_csv("./datasets/preprocess/adamson/control_test_rpe1.csv")

control_test_df = pd.concat(
    [control_test_K562_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_test_df.to_csv(
    "./datasets/preprocess/adamson/control_test.csv",
    index=False,
)
print(control_test_df)

# if treat_test_df.shape[0] != control_test_df.shape[0]:
#     print("nani3")
#     exit()
