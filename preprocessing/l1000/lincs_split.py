import pandas as pd
from sklearn.metrics import r2_score


treat_data = pd.read_csv("./datasets/preprocess/l1000/l1000_treat_24h_10.csv")
treat_data = treat_data.drop(columns=["Unnamed: 0"])

control_data = pd.read_csv("./datasets/preprocess/l1000/l1000_control_24h.csv")
control_data = control_data.set_index("Unnamed: 0")

r2_dict = {}
for i in range(treat_data.shape[0]):
    treat = treat_data.iloc[i, :-3]
    cell_id = treat_data.iloc[i, -3]
    control = control_data.loc[cell_id, :]

    r2 = r2_score(treat, control)

    r2_dict[i] = r2

sorted_r2_list = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)

key_list = []
for key, value in sorted_r2_list:
    key_list.append(key)

valid_and_test = key_list[int(len(key_list) * 0.75) : int(len(key_list) * 0.95)]

valid = treat_data.iloc[valid_and_test]
train = treat_data.drop(valid.index)
test = valid.sample(frac=0.5, replace=False, random_state=100)
valid = valid.drop(test.index)

print(train.shape)
print(valid.shape)
print(test.shape)

train.to_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_train.csv")
valid.to_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_valid.csv")
test.to_csv("./datasets/preprocess/l1000/l1000_treat_24h_10_test.csv")
