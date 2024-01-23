import pandas as pd


treat_train_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_train_MCF7.csv"
)
treat_train_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_train_K562.csv"
)
treat_train_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_train_A549.csv"
)

treat_train_df = pd.concat(
    [treat_train_MCF7_df, treat_train_K562_df, treat_train_A549_df], ignore_index=True
)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_train_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_train.csv", index=False
)
print(treat_train_df.shape)

control_train_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_train_MCF7.csv"
)
control_train_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_train_K562.csv"
)
control_train_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_train_A549.csv"
)

control_train_df = pd.concat(
    [control_train_MCF7_df, control_train_K562_df, control_train_A549_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_train_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_train.csv", index=False
)
print(control_train_df.shape)

if treat_train_df.shape[0] != control_train_df.shape[0]:
    print("nani1")
    exit()


treat_valid_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_valid_MCF7.csv"
)
treat_valid_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_valid_K562.csv"
)
treat_valid_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_valid_A549.csv"
)

treat_valid_df = pd.concat(
    [treat_valid_MCF7_df, treat_valid_K562_df, treat_valid_A549_df], ignore_index=True
)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_valid_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_valid.csv", index=False
)
print(treat_valid_df.shape)

control_valid_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_valid_MCF7.csv"
)
control_valid_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_valid_K562.csv"
)
control_valid_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_valid_A549.csv"
)

control_valid_df = pd.concat(
    [control_valid_MCF7_df, control_valid_K562_df, control_valid_A549_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_valid_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_valid.csv", index=False
)
print(control_valid_df.shape)

if treat_valid_df.shape[0] != control_valid_df.shape[0]:
    print("nani2")
    exit()


treat_test_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test_MCF7.csv"
)
treat_test_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test_K562.csv"
)
treat_test_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test_A549.csv"
)

treat_test_df = pd.concat(
    [treat_test_MCF7_df, treat_test_K562_df, treat_test_A549_df], ignore_index=True
)
# 修改 treat_df 的标签列名
# treat_train_df.rename(columns={'978': 'cell_type', "979": "SMILES", "980": "cov_drug"}, inplace=True)
treat_test_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_treat_test.csv", index=False
)
print(treat_test_df.shape)

control_test_MCF7_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test_MCF7.csv"
)
control_test_K562_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test_K562.csv"
)
control_test_A549_df = pd.read_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test_A549.csv"
)

control_test_df = pd.concat(
    [control_test_MCF7_df, control_test_K562_df, control_test_A549_df],
    ignore_index=True,
)
# 修改 control_train_df 的标签列名
# control_train_df.rename(columns={'978': 'cell_type'}, inplace=True)
control_test_df.to_csv(
    "./datasets/preprocess/sciplex4/chemcpa_trapnell_control_test.csv", index=False
)
print(control_test_df.shape)

if treat_test_df.shape[0] != control_test_df.shape[0]:
    print("nani3")
    exit()

print("finish!")
