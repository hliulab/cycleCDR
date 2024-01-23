import numpy as np
import math
import matplotlib.pyplot as plt

def get_lr_lambda():
    warmup_start_lr = 0.0001
    warmup_factor = 1.12
    warmup_epoch = 15

    num_epoch = 60
    lr = 0.0007

    return lambda epoch: warmup_start_lr * (warmup_factor ** epoch) if  epoch < warmup_epoch else \
        lr * 0.5 * ( math.cos((epoch - warmup_epoch) /(num_epoch - warmup_epoch) * math.pi) + 1)


def get_lr_lambda_sc():

    num_epoch = 50
    lr =  0.0001

    return lambda epoch: lr * ((num_epoch * 1.9 - epoch) / (num_epoch * 1.9))



def Warmup_cosine():

    lrs = []
    lr_lambda= get_lr_lambda_sc()
    for i in range(50):
        lr = lr_lambda(i)
        lrs.append(lr)

    
    lrs = np.array(lrs)

    # plot
    plt.plot(lrs)
    plt.xlabel("Iters")
    plt.ylabel("lr")
    plt.show()


if __name__ == "__main__":
    Warmup_cosine()
