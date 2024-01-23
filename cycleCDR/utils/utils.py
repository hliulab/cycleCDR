import yaml
import torch
import random
from optuna import Trial


def read_config():
    config_path = './configs/train_sciplex_gat_row_optuna_fixed.yaml'

    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        data = list(data)[0]

    return data


def parse_config(trial: Trial):
    config = read_config()

    res = {}

    for key in config.keys():
        if isinstance(config[key], dict):
            if config[key]["type"] == "int":
                res[key] = trial.suggest_int(key, config[key]["min"], config[key]["max"])
            elif config[key]["type"] == "float":
                res[key] = trial.suggest_float(
                    key, config[key]["min"], config[key]["max"]
                )
            elif config[key]["type"] == "choices":
                res[key] = trial.suggest_categorical(key, config[key]["options"])
        else:
            res[key] = config[key]

    return res


def get_batch_data(batch, dataset_name="l1000"):
    controls = batch[0].cuda()
    treats = batch[1].cuda()
    drugs = batch[2].cuda()

    if dataset_name == "l1000":
        pert_names = batch[3]
        de_index = batch[4]
        # de_index = None
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "cppa":
        dose = batch[3].cuda()
        pert_names = batch[4]
        return controls, treats, drugs, dose, None, pert_names
    elif dataset_name == "adamson":
        de_index = batch[3].cuda()
        pert_names = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "sciplex3":
        pert_names = batch[3]
        de_index = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "sciplex4":
        pert_names = batch[3]
        de_index = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "dixit":
        pert_names = batch[3]
        de_index = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "rep1":
        pert_names = batch[3]
        de_index = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    elif dataset_name == "rep1k562":
        pert_names = batch[3]
        de_index = batch[4]
        return controls, treats, drugs, None, de_index, pert_names
    else:
        raise ValueError(f"{dataset_name} not support get_batch_data")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
