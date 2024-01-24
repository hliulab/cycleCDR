import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset


class CPADataset(Dataset):
    def __init__(
        self, drug_path, control_path, treat_path, de_gene_path, dtype=torch.float32
    ):
        drugs = pd.read_parquet(drug_path)

        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")

        # de_gene_idx 的 index 去重
        self.de_gene_idx = self.de_gene_idx[
            ~self.de_gene_idx.index.duplicated(keep='first')
        ]

        control_cell = pd.read_csv(control_path)
        control_cell = control_cell.set_index("Unnamed: 0")

        treat_train_cell = pd.read_csv(treat_path)
        treat_train_cell = treat_train_cell.drop(columns=["Unnamed: 0"])

        self.drugs = drugs
        self.control_cell = control_cell
        self.treat_train_cell = treat_train_cell
        self.dtype = dtype

    def __len__(self):
        return len(self.treat_train_cell)

    def __getitem__(self, i):
        temp = self.treat_train_cell.iloc[i, :]
        cell_id = temp["cell_id"]
        pert_id = temp["pert_id"]
        temp = temp.drop(["cell_id", "pert_id", "pert_dose"])

        control = (
            torch.tensor(self.control_cell.loc[cell_id].values, dtype=self.dtype) / 10
        )
        treat = torch.tensor(temp.to_list(), dtype=self.dtype) / 10
        smiles = torch.tensor(self.drugs.loc[pert_id].values, dtype=self.dtype)

        # 判定 cell_id + pert_id 是否在 de_gene_idx 的 index 中
        if pert_id in self.de_gene_idx.index:
            de_idx = self.de_gene_idx.loc[pert_id].to_numpy()
            if de_idx.shape[0] != 50:
                de_idx = de_idx[0]
        else:
            de_idx = [-1 for _ in range(50)]

        de_idx = torch.tensor(de_idx, dtype=torch.long)

        return (control, treat, smiles, cell_id + pert_id, de_idx)


class Sciplex3Dataset(Dataset):
    def __init__(
        self,
        sciplex_control_path,
        sciplex_treat_path,
        drug_path,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        self.drugs = pd.read_parquet(drug_path)
        self.split_size = split_size

        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")

        self.control = pd.read_csv(sciplex_control_path)
        self.treat = pd.read_csv(sciplex_treat_path)

        self.len = self.treat.shape[0]
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]
        cell_id = treat.loc["cell_type"]

        if cell_id != self.control.iloc[i].loc["cell_type"]:
            raise ValueError("cell_id not match")

        smile = treat.loc["SMILES"]
        cov_drug = treat.loc["cov_drug"]
        de_idx = self.de_gene_idx.loc[cov_drug].to_numpy()

        control = self.control.iloc[i].to_numpy()[:-1]
        treat = treat.to_numpy()[:-3]

        drug = list(self.drugs.loc[smile])

        control = torch.tensor(list(control), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)
        smiles = torch.tensor(drug, dtype=self.dtype)

        de_idx = torch.tensor(list(de_idx), dtype=torch.long)

        return (control, treat, smiles, cov_drug, de_idx)


class CPPADataset(Dataset):
    def __init__(self, control_path, treat_path, drug_path, dtype=torch.float32):
        self.drugs = pd.read_parquet(drug_path)

        self.drugs = self.drugs.set_index(self.drugs.index.str.split(" ").str.join("-"))

        self.control = pd.read_csv(control_path)

        self.treat = pd.read_csv(treat_path)

        self.dtype = dtype

    def __len__(self):
        return self.treat.shape[0]

    def __getitem__(self, i):
        treat = self.treat.iloc[i]
        id = treat["id"]
        pert_name = treat["id"]
        id = id.split("_")
        drug_name = id[-2]

        dose = id[-1]

        drug = list(self.drugs.loc[drug_name])

        treat = treat.to_numpy()[:-1]
        control = self.control.iloc[i].to_numpy()[:-1]

        control = torch.tensor(list(control), dtype=self.dtype) / 10
        treat = torch.tensor(list(treat), dtype=self.dtype) / 10
        smiles = torch.tensor(drug, dtype=self.dtype)
        dose = torch.tensor([float(dose)], dtype=self.dtype)

        # 判定 dose 是否为 torch.tensor([0])

        if dose == torch.tensor([0.0], dtype=self.dtype):
            print("dose is 0")
            exit()

        return (control, treat, smiles, dose, pert_name)


class DixitDataset(Dataset):
    def __init__(
        self,
        control_path,
        treat_path,
        pert_path,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        with open(pert_path, "rb") as f:
            self.perts = pickle.load(f)

        self.split_size = split_size

        self.control = pd.read_csv(control_path)
        self.treat = pd.read_csv(treat_path)
        self.treat_len = self.treat.shape[0]

        self.de_gene = pd.read_csv(de_gene_path, index_col=0)

        self.dtype = dtype

    def __len__(self):
        return self.treat_len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]
        condition = treat.loc["condition"]
        cov_pert = treat.loc["cov_pert"]
        treat = treat.to_numpy()[:-3]
        pert = self.perts[condition.split("+")[0]]
        de_idx = self.de_gene.loc[cov_pert].to_numpy()

        control = self.control.iloc[i, :-1]

        control = torch.tensor(list(control.to_numpy()), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)
        pert = torch.tensor([pert], dtype=self.dtype) / 1000
        de_index = torch.tensor(de_idx, dtype=torch.long)

        return (control, treat, pert, cov_pert, de_index)
    
class Rep1Dataset(Dataset):
    def __init__(
        self,
        control_path,
        treat_path,
        pert_path,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        with open(pert_path, "rb") as f:
            self.perts = pickle.load(f)

        self.split_size = split_size

        self.control = pd.read_csv(control_path)
        self.treat = pd.read_csv(treat_path)
        self.treat_len = self.treat.shape[0]

        self.de_gene = pd.read_csv(de_gene_path, index_col=0)

        self.dtype = dtype

    def __len__(self):
        return self.treat_len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]
        condition = treat.loc["condition"]
        cov_pert = treat.loc["cov_pert"]
        treat = treat.to_numpy()[:-3]
        pert = self.perts[condition.split("+")[0]]
        de_idx = self.de_gene.loc[cov_pert].to_numpy()

        control = self.control.iloc[i, :-1]

        control = torch.tensor(list(control.to_numpy()), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)
        pert = torch.tensor([pert], dtype=self.dtype) / 10000
        de_index = torch.tensor(de_idx, dtype=torch.long)

        return (control, treat, pert, cov_pert, de_index)
    

class Rep1K562Dataset(Dataset):
    def __init__(
        self,
        control_path,
        treat_path,
        pert_path,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        with open(pert_path, "rb") as f:
            self.perts = pickle.load(f)

        self.split_size = split_size

        self.control = pd.read_csv(control_path)
        self.treat = pd.read_csv(treat_path)
        self.treat_len = self.treat.shape[0]

        self.de_gene = pd.read_csv(de_gene_path, index_col=0)

        self.dtype = dtype

    def __len__(self):
        return self.treat_len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]
        condition = treat.loc["condition"]
        cov_pert = treat.loc["cov_pert"]
        treat = treat.to_numpy()[:-3]
        pert = self.perts[condition.split("+")[0]]
        de_idx = self.de_gene.loc[cov_pert].to_numpy()

        control = self.control.iloc[i, :-1]

        control = torch.tensor(list(control.to_numpy()), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)
        pert = torch.tensor([pert], dtype=self.dtype) / 1000
        de_index = torch.tensor(de_idx, dtype=torch.long)

        return (control, treat, pert, cov_pert, de_index)


def load_dataset_splits(config):
    if config["dataset"] == "l1000":
        train_dataset = CPADataset(
            config["l1000_drug"],
            config["l1000_control_gene"],
            config["l1000_treat_train_gene"],
            config["l1000_de_gene"],
        )
        valid_dataset = CPADataset(
            config["l1000_drug"],
            config["l1000_control_gene"],
            config["l1000_treat_valid_gene"],
            config["l1000_de_gene"],
        )
        test_dataset = CPADataset(
            config["l1000_drug"],
            config["l1000_control_gene"],
            config["l1000_treat_test_gene"],
            config["l1000_de_gene"],
        )
    elif config["dataset"] == "sciplex3":
        train_dataset = Sciplex3Dataset(
            config["sciplex3_control_train"],
            config["sciplex3_treat_train"],
            config["sciplex3_drug"],
            config["sciplex3_de_gene"],
        )
        valid_dataset = Sciplex3Dataset(
            config["sciplex3_control_valid"],
            config["sciplex3_treat_valid"],
            config["sciplex3_drug"],
            config["sciplex3_de_gene"],
        )
        test_dataset = Sciplex3Dataset(
            config["sciplex3_control_test"],
            config["sciplex3_treat_test"],
            config["sciplex3_drug"],
            config["sciplex3_de_gene"],
        )
    elif config["dataset"] == "cppa":
        train_dataset = CPPADataset(
            config["cppa_control_train"],
            config["cppa_treat_train"],
            config["cppa_drug"],
        )
        valid_dataset = CPPADataset(
            config["cppa_control_valid"],
            config["cppa_treat_valid"],
            config["cppa_drug"],
        )
        test_dataset = CPPADataset(
            config["cppa_control_test"], config["cppa_treat_test"], config["cppa_drug"]
        )
    elif config["dataset"] == "dixit":
        train_dataset = DixitDataset(
            config["dixit_control_train"],
            config["dixit_treat_train"],
            config["dixit_pert"],
            config["dixit_de_gene"],
        )
        valid_dataset = DixitDataset(
            config["dixit_control_valid"],
            config["dixit_treat_valid"],
            config["dixit_pert"],
            config["dixit_de_gene"],
        )
        test_dataset = DixitDataset(
            config["dixit_control_test"],
            config["dixit_treat_test"],
            config["dixit_pert"],
            config["dixit_de_gene"],
        )
    elif config["dataset"] == "rep1":
        train_dataset = Rep1Dataset(
            config["rep1_control_train"],
            config["rep1_treat_train"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )
        valid_dataset = Rep1Dataset(
            config["rep1_control_valid"],
            config["rep1_treat_valid"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )
        test_dataset = Rep1Dataset(
            config["rep1_control_test"],
            config["rep1_treat_test"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )
    elif config["dataset"] == "rep1k562":
        train_dataset = Rep1K562Dataset(
            config["rep1_control_train"],
            config["rep1_treat_train"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )
        valid_dataset = Rep1K562Dataset(
            config["rep1_control_valid"],
            config["rep1_treat_valid"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )
        test_dataset = Rep1K562Dataset(
            config["rep1_control_test"],
            config["rep1_treat_test"],
            config["rep1_pert"],
            config["rep1_de_gene"],
        )

    splits = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    return splits
