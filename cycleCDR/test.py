import time
import torch
from torch import nn


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    net = nn.Sequential(
        nn.Linear(128, 256, dtype=torch.float64),
        nn.BatchNorm1d(256),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.BatchNorm1d(512),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.BatchNorm1d(512),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.BatchNorm1d(512),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(256, 512, dtype=torch.float64),
        nn.Linear(512, 1024, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.Linear(1024, 2000, dtype=torch.float64),
        nn.ReLU(),
    ).cuda()

    print("yes")
    for i in range(60 * 24 * 30):
        for i in range(60 * 60 * 60):
            x = torch.randn(2, 3).cuda()
            y = torch.randn(2, 3).cuda()
            z = x * y
        time.sleep(60)
