import os
import pickle
import torch
import pandas as pd
from torch_geometric import data as DATA

from .pre_smile_data import smile_to_graph


class CPAGATDataset:
    def __init__(
        self,
        row_drug_data_path,
        processed_drug_paths,
        control_path,
        treat_path,
        de_gene_path,
        dtype=torch.float32,
    ):
        if os.path.isfile(processed_drug_paths):
            print(f'Pre-processed data found: {processed_drug_paths}, loading ...')
            self.drugs = torch.load(processed_drug_paths)
        else:
            print(
                f'Pre-processed data {processed_drug_paths} not found, doing pre-processing...'
            )
            self.process(row_drug_data_path, processed_drug_paths)
            self.drugs = torch.load(processed_drug_paths)

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
        drug_graph = self.drugs[pert_id]

        # 判定 cell_id + pert_id 是否在 de_gene_idx 的 index 中
        if pert_id in self.de_gene_idx.index:
            de_idx = self.de_gene_idx.loc[pert_id].to_numpy()
            if de_idx.shape[0] != 50:
                de_idx = de_idx[0]
        else:
            de_idx = [-1 for _ in range(50)]

        de_idx = torch.tensor(de_idx, dtype=torch.long)

        return (control, treat, drug_graph, cell_id + pert_id, de_idx)

    def process(self, row_data_path, processed_paths):
        data_dict = {}
        compound_iso_smiles = []
        row_data_df = pd.read_csv(row_data_path)
        compound_iso_smiles += list(row_data_df['canonical_smiles'])
        compound_iso_smiles = set(compound_iso_smiles)
        count = 0
        for smile in compound_iso_smiles:
            if len(smile) < 2:
                continue
            count = count + 1
            print('smiles ', count, smile)
            c_size, features, edge_index, atoms = smile_to_graph(smile)

            if len(edge_index) == 0:
                continue

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            )

            pert_ids = row_data_df[row_data_df['canonical_smiles'] == smile][
                'pert_id'
            ].to_numpy()
            for pert_id in pert_ids:
                data_dict[pert_id] = GCNData

        print('Graph construction done. Saving to file.')

        torch.save(data_dict, processed_paths)


class CPPADataset:
    def __init__(
        self,
        control_path,
        treat_path,
        row_drug_data_path,
        processed_drug_paths,
        dtype=torch.float32,
    ):
        if os.path.isfile(processed_drug_paths):
            print(f'Pre-processed data found: {processed_drug_paths}, loading ...')
            self.drugs = torch.load(processed_drug_paths)
        else:
            print(
                f'Pre-processed data {processed_drug_paths} not found, doing pre-processing...'
            )
            self.process(row_drug_data_path, processed_drug_paths)
            self.drugs = torch.load(processed_drug_paths)

        # self.drugs.set_index(self.drugs.index.str.split(
        #     " ").str.join("-"), inplace=True)

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

        drug = self.drugs[drug_name]

        treat = treat.to_numpy()[:-1]
        control = self.control.iloc[i].to_numpy()[:-1]

        control = torch.tensor(list(control), dtype=self.dtype) / 10
        treat = torch.tensor(list(treat), dtype=self.dtype) / 10
        # smiles = torch.tensor(drug, dtype=self.dtype)
        dose = torch.tensor([float(dose)], dtype=self.dtype)

        # 判定 dose 是否为 torch.tensor([0])

        if dose == torch.tensor([0.0], dtype=self.dtype):
            print("dose is 0")
            exit()

        return (control, treat, drug, dose, pert_name)

    def process(self, row_data_path, processed_paths):
        data_dict = {}
        with open(row_data_path, 'rb') as f:
            row_smiles = pickle.load(f)

        count = 0
        for drug_name in row_smiles.keys():
            count = count + 1
            print('smiles ', count, drug_name)
            c_size, features, edge_index, atoms = smile_to_graph(row_smiles[drug_name])

            if len(edge_index) == 0:
                continue

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            )

            data_dict[drug_name] = GCNData

        print('Graph construction done. Saving to file.')

        torch.save(data_dict, processed_paths)


class Sciplex3Dataset:
    def __init__(
        self,
        sciplex_control_path,
        sciplex_treat_path,
        row_drug_data_path,
        processed_drug_paths,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        if os.path.isfile(processed_drug_paths):
            print(f'Pre-processed data found: {processed_drug_paths}, loading ...')
            self.drugs = torch.load(processed_drug_paths)
        else:
            print(
                f'Pre-processed data {processed_drug_paths} not found, doing pre-processing...'
            )
            self.process(row_drug_data_path, processed_drug_paths)
            self.drugs = torch.load(processed_drug_paths)

        print("drug load done")
        self.split_size = split_size

        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")

        print("gene loading...")
        self.control = pd.read_csv(sciplex_control_path)
        self.treat = pd.read_csv(sciplex_treat_path)
        print("gene load done")

        self.len = self.treat.shape[0]
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]

        cov_drug = treat.loc["cov_drug"]
        smile = treat.loc["SMILES"]

        de_idx = self.de_gene_idx.loc[cov_drug].to_numpy()

        control = self.control.iloc[i].to_numpy()[:-1]
        treat = treat.to_numpy()[:-3]

        drug = self.drugs[smile]

        control = torch.tensor(list(control), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)

        de_idx = torch.tensor(list(de_idx), dtype=torch.long)

        return (control, treat, drug, cov_drug, de_idx)

    def process(self, row_data_path, processed_paths):
        data_dict = {}

        row_smiles = pd.read_parquet(row_data_path)

        row_smiles = row_smiles.index.to_list()

        count = 0
        for smile in row_smiles:
            count = count + 1
            print('smiles ', count, smile)
            c_size, features, edge_index, atoms = smile_to_graph(smile)

            if len(edge_index) == 0:
                continue

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            )

            data_dict[smile] = GCNData

        print('Graph construction done. Saving to file.')

        torch.save(data_dict, processed_paths)


class Sciplex4Dataset:
    def __init__(
        self,
        sciplex_control_path,
        sciplex_treat_path,
        row_drug_data_path,
        processed_drug_paths,
        de_gene_path,
        split_size=1000,
        dtype=torch.float32,
    ):
        if os.path.isfile(processed_drug_paths):
            print(f'Pre-processed data found: {processed_drug_paths}, loading ...')
            self.drugs = torch.load(processed_drug_paths)
        else:
            print(
                f'Pre-processed data {processed_drug_paths} not found, doing pre-processing...'
            )
            self.process(row_drug_data_path, processed_drug_paths)
            self.drugs = torch.load(processed_drug_paths)

        print("drug load done")
        self.split_size = split_size

        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")

        print("gene loading...")
        self.control = pd.read_csv(sciplex_control_path)
        self.treat = pd.read_csv(sciplex_treat_path)
        print("gene load done")

        self.len = self.treat.shape[0]
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        treat = self.treat.iloc[i]

        cov_drug = treat.loc["cov_drug"]
        smile = treat.loc["SMILES"]

        de_idx = self.de_gene_idx.loc[cov_drug].to_numpy()

        control = self.control.iloc[i].to_numpy()[:-1]
        treat = treat.to_numpy()[:-3]

        drug = self.drugs[smile]

        control = torch.tensor(list(control), dtype=self.dtype)
        treat = torch.tensor(list(treat), dtype=self.dtype)

        de_idx = torch.tensor(list(de_idx), dtype=torch.long)

        return (control, treat, drug, cov_drug, de_idx)

    def process(self, row_data_path, processed_paths):
        data_dict = {}

        row_smiles = pd.read_parquet(row_data_path)

        row_smiles = row_smiles.index.to_list()

        count = 0
        for smile in row_smiles:
            count = count + 1
            print('smiles ', count, smile)
            c_size, features, edge_index, atoms = smile_to_graph(smile)

            if len(edge_index) == 0:
                continue

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            )

            data_dict[smile] = GCNData

        print('Graph construction done. Saving to file.')

        torch.save(data_dict, processed_paths)


def load_dataset_splits_for_gat(config):
    if config["dataset"] == "l1000":
        train_dataset = CPAGATDataset(
            config["l1000_row_drug"],
            config["l1000_drug_graph"],
            config["l1000_control_gene"],
            config["l1000_treat_train_gene"],
            config["l1000_de_gene"],
        )
        valid_dataset = CPAGATDataset(
            config["l1000_row_drug"],
            config["l1000_drug_graph"],
            config["l1000_control_gene"],
            config["l1000_treat_valid_gene"],
            config["l1000_de_gene"],
        )
        test_dataset = CPAGATDataset(
            config["l1000_row_drug"],
            config["l1000_drug_graph"],
            config["l1000_control_gene"],
            config["l1000_treat_test_gene"],
            config["l1000_de_gene"],
        )
    elif config["dataset"] == "sciplex3":
        train_dataset = Sciplex3Dataset(
            config["sciplex3_control_train"],
            config["sciplex3_treat_train"],
            config["sciplex3_row_drug"],
            config["sciplex3_processed_drug"],
            config["sciplex3_de_gene"],
        )
        valid_dataset = Sciplex3Dataset(
            config["sciplex3_control_valid"],
            config["sciplex3_treat_valid"],
            config["sciplex3_row_drug"],
            config["sciplex3_processed_drug"],
            config["sciplex3_de_gene"],
        )
        test_dataset = Sciplex3Dataset(
            config["sciplex3_control_test"],
            config["sciplex3_treat_test"],
            config["sciplex3_row_drug"],
            config["sciplex3_processed_drug"],
            config["sciplex3_de_gene"],
        )
    elif config["dataset"] == "sciplex4":
        train_dataset = Sciplex4Dataset(
            config["sciplex4_control_train"],
            config["sciplex4_treat_train"],
            config["sciplex4_row_drug"],
            config["sciplex4_processed_drug"],
            config["sciplex4_de_gene"],
        )
        valid_dataset = Sciplex4Dataset(
            config["sciplex4_control_valid"],
            config["sciplex4_treat_valid"],
            config["sciplex4_row_drug"],
            config["sciplex4_processed_drug"],
            config["sciplex4_de_gene"],
        )
        test_dataset = Sciplex4Dataset(
            config["sciplex4_control_test"],
            config["sciplex4_treat_test"],
            config["sciplex4_row_drug"],
            config["sciplex4_processed_drug"],
            config["sciplex4_de_gene"],
        )
    elif config["dataset"] == "cppa":
        train_dataset = CPPADataset(
            config["cppa_control_train"],
            config["cppa_treat_train"],
            config["cppa_row_drug"],
            config["cppa_processed_drug"],
        )
        valid_dataset = CPPADataset(
            config["cppa_control_valid"],
            config["cppa_treat_valid"],
            config["cppa_row_drug"],
            config["cppa_processed_drug"],
        )
        test_dataset = CPPADataset(
            config["cppa_control_test"],
            config["cppa_treat_test"],
            config["cppa_row_drug"],
            config["cppa_processed_drug"],
        )

    splits = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    return splits
