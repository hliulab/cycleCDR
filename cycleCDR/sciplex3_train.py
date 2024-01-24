import os
import sys
import torch
# import pickle
import optuna
import itertools
from optuna import Trial
import torch.optim as optim
import torch.distributed as dist
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.getcwd())
from cycleCDR.utils import Trainer
from cycleCDR.model import cycleCDR
from cycleCDR.utils import set_seed
from cycleCDR.utils import parse_config
from cycleCDR.dataset import load_dataset_splits
from cycleCDR.dataset import load_dataset_splits_for_gat


def main(single_trial: Trial):
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    config = parse_config(trial)

    set_seed(config["seed"])

    if config["is_drug_gat"]:
        datasets = load_dataset_splits_for_gat(config)
    else:
        datasets = load_dataset_splits(config)

    model = cycleCDR(config).cuda()

    if config["is_drug_gat"] and config["gat_pretrained"]:
        print("drug gat pretrained")
        pretained = torch.load(config["gat_pretrained_path"])
        temp = OrderedDict()
        for k, v in pretained.items():
            if "att_l" in k:
                k = k.replace("att_l", "att_src")
            if "att_r" in k:
                k = k.replace("att_r", "att_dst")
            if "lin_l" in k:
                k = k.replace("lin_l", "lin_src")
            if "lin_r" in k:
                k = k.replace("lin_r", "lin_dst")
            temp[k] = v

        model.drug_encoder.load_state_dict(temp)

    optimizer_D = optim.Adam(
        itertools.chain(
            model.discriminator_A.parameters(), model.discriminator_B.parameters()
        ),
        lr=config["d_lr"],
        weight_decay=float(config["d_weight_decay"]),
    )
    # lr 由 scheduler 调整, 初始 lr 也是在 scheduler 中设置, 这里的 lr 是一个无用参数
    optimizer_G = optim.Adam(
        itertools.chain(
            model.encoderG_A.parameters(),
            model.encoderG_B.parameters(),
            model.decoderG_A.parameters(),
            model.decoderG_B.parameters(),
        ),
        lr=config["g_lr"],
        weight_decay=float(config["g_weight_decay"]),
    )

    optimizer_DRUG = optim.Adam(
        model.drug_encoder.parameters(),
        lr=config["drug_lr"],
        weight_decay=float(config["drug_weight_decay"]),
    )

    scheduler_G = lr_scheduler.StepLR(
        optimizer_G, step_size=config["g_step_size"], gamma=config["g_gamma"]
    )

    scheduler_D = lr_scheduler.StepLR(
        optimizer_D, step_size=config["d_step_size"], gamma=config["d_gamma"]
    )

    scheduler_DRUG = lr_scheduler.StepLR(
        optimizer_DRUG, step_size=config["drug_step_size"], gamma=config["drug_gamma"]
    )

    train_sampler = DistributedSampler(datasets["train"], shuffle=True)
    valid_sampler = DistributedSampler(datasets["valid"], shuffle=True)
    test_sampler = DistributedSampler(datasets["test"], shuffle=False)
    train_dataloader = DataLoader(
        dataset=datasets["train"],
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
    )
    valid_dataloader = DataLoader(
        dataset=datasets["valid"],
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset=datasets["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=4,
    )

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=True,
    )

    trainer = Trainer(
        model,
        config["num_epoch"],
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer_DRUG=optimizer_DRUG,
        scheduler_DRUG=scheduler_DRUG,
        dataset_name=config["dataset"],
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        is_mse=config["is_mse"],
        is_gan=config["is_gan"],
        config=config,
    )

    trainer.fit()
    trainer.plot(str(model.device))

    return trainer.treats_r2_cpa_list[-1]


if __name__ == "__main__":
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE")
    if world_size is None:
        world_size = os.environ.get("PMI_SIZE")
    if world_size is None:
        raise RuntimeError("Neither MPICH nor OpenMPI is avaliable.")
    os.environ["WORLD_SIZE"] = str(world_size)

    rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    if rank is None:
        rank = os.environ.get("PMI_RANK")
    os.environ["RANK"] = str(rank)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "21000"

    dist.init_process_group("gloo")
    local_rank = dist.get_rank()

    # if torch.cuda.device_count() >= 1:
    #     torch.distributed.init_process_group(backend="nccl")
    #     local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    # else:
    #     device = torch.device("cuda:0")
    study = None
    n_trials = 1

    if local_rank == 0:
        study = optuna.create_study(direction='maximize')
        study.optimize(main, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                main(None)
            except optuna.TrialPruned:
                pass

    # if local_rank == 0:
    #     with open("./results/studies/study_sciplex.pkl", "wb") as f:
    #         pickle.dump(study, f)
