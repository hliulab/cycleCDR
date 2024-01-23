import os
import sys
import yaml
import torch
import random
import itertools
import torch.optim as optim
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.getcwd())
from cycleCDR.model import cycleCDR
from cycleCDR.utils import Trainer
from cycleCDR.dataset import load_dataset_splits
from cycleCDR.dataset import load_dataset_splits_for_gat


def parse_config():
    config_path = './configs/train_l1000_gat.yaml'

    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        # salf_load_all方法得到的是一个迭代器，需要使用list()方法转换为列表
        data = list(data)[0]

    return data


def get_lr_lambda(config):
    num_epoch = config["num_epoch"]
    lr = config["lr"]
    lambda_lr_A = config["lambda_lr_A"]
    lambda_lr_B = config["lambda_lr_B"]

    return lambda epoch: lr * (
        (num_epoch * lambda_lr_A - epoch) / (num_epoch * lambda_lr_B)
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = parse_config()

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0")

    set_seed(config["seed"])

    if config["is_drug_gat"]:
        datasets = load_dataset_splits_for_gat(config)
    else:
        datasets = load_dataset_splits(config)

    # print(datasets["train"][2])
    # exit()

    model = cycleCDR(config).cuda()
    if config["pretrained"]:
        pretrained_dict = torch.load(config["pretrained_path"])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # print(pretrained_dict.keys())
        # exit()

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
        lr=1,
        weight_decay=float(config["weight_decay"]),
    )
    if config["dataset"] == "sciplex3":
        optimizer_D = optim.SGD(
            itertools.chain(
                model.discriminator_A.parameters(), model.discriminator_B.parameters()
            ),
            lr=1,
            weight_decay=float(config["weight_decay"]),
        )
        # lr 由 scheduler 调整, 初始 lr 也是在 scheduler 中设置, 这里的 lr 是一个无用参数
        optimizer_G = optim.Adam(
            itertools.chain(
                model.encoderG_A.parameters(),
                model.encoderG_B.parameters(),
                model.decoderG_A.parameters(),
                model.decoderG_B.parameters(),
                model.drug_encoder.parameters(),
            ),
            lr=config["lr"],
            weight_decay=float(config["weight_decay"]),
        )
        # scheduler_G = optim.lr_scheduler.LambdaLR(
        #     optimizer_G, lr_lambda=get_lr_lambda(config)
        # )
        scheduler_G = lr_scheduler.StepLR(
            optimizer_G, step_size=config["step_size"], gamma=config["gamma"]
        )
        # scheduler_D = optim.lr_scheduler.LambdaLR(
        #     optimizer_D, lr_lambda=get_lr_lambda(config)
        # )
        scheduler_D = lr_scheduler.StepLR(
            optimizer_D, step_size=config["step_size"], gamma=config["gamma"]
        )
    elif config["dataset"] == "l1000":
        # lr 由 scheduler 调整, 初始 lr 也是在 scheduler 中设置, 这里的 lr 是一个无用参数
        optimizer_G = optim.Adam(
            itertools.chain(
                model.encoderG_A.parameters(),
                model.encoderG_B.parameters(),
                model.decoderG_A.parameters(),
                model.decoderG_B.parameters(),
                model.drug_encoder.parameters(),
            ),
            lr=1,
            weight_decay=float(config["weight_decay"]),
        )
        scheduler_G = optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=get_lr_lambda(config)
        )
        scheduler_D = optim.lr_scheduler.LambdaLR(
            optimizer_D, lr_lambda=get_lr_lambda(config)
        )
    elif config["dataset"] == "cppa":
        optimizer_G = optim.Adam(
            itertools.chain(
                model.encoderG_A.parameters(),
                model.encoderG_B.parameters(),
                model.decoderG_A.parameters(),
                model.decoderG_B.parameters(),
                model.drug_encoder.parameters(),
                model.dose_encoder.parameters(),
            ),
            lr=1,
            weight_decay=float(config["weight_decay"]),
        )
        scheduler_G = optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=get_lr_lambda(config)
        )
        scheduler_D = optim.lr_scheduler.LambdaLR(
            optimizer_D, lr_lambda=get_lr_lambda(config)
        )
    elif config["dataset"] == "adamson":
        optimizer_G = optim.Adam(
            itertools.chain(
                model.encoderG_A.parameters(),
                model.encoderG_B.parameters(),
                model.decoderG_A.parameters(),
                model.decoderG_B.parameters(),
                model.drug_encoder.parameters(),
            ),
            lr=1,
            weight_decay=float(config["weight_decay"]),
        )
        scheduler_G = optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=get_lr_lambda(config)
        )
        scheduler_D = optim.lr_scheduler.LambdaLR(
            optimizer_D, lr_lambda=get_lr_lambda(config)
        )

    print("Let's use", torch.cuda.device_count(), "GPUs! device: ", local_rank)

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
        dataset_name=config["dataset"],
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        is_mse=config["is_mse"],
        is_gan=config["is_gan"],
    )

    trainer.fit()
    trainer.plot(str(model.device))
    print(config)
