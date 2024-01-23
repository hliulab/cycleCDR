import torch
import pickle
import torch.nn as nn
from .gat import GATNet
from .utils import MLP
from .cppa_model import cppa_model
from .l1000_model import l1000_model
from .sciplex3_model import sciplex3_model
from .sciplex4_model import sciplex4_model
from .dixit_model import dixit_model
from .rep1_model import rep1_model
from .rep1k562_model import rep1k562_model


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()

        self.loss = nn.BCELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = torch.Tensor(prediction.size(0), 1).fill_(1.0).cuda()
        else:
            target_tensor = torch.Tensor(prediction.size(0), 1).fill_(0.0).cuda()

        return target_tensor

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss


class FixedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, prediction, target, weight=None):
        if weight is not None:
            if weight.shape[0] == 1:
                weight = weight.repeat(target.shape[0], 1).cuda()

            loss = torch.mean(weight * torch.square(prediction - target))

        else:
            loss = torch.mean(torch.square(prediction - target))

        return loss


class cycleCDR(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()

        self.loss_autoencoder = FixedMSELoss()
        self.loss_gan = GANLoss("lsgan")

        self.lambda_idt = config["lambda_idt"]
        self.lambda_A = config["lambda_A"]
        self.lambda_B = config["lambda_B"]

        self.lambda_gen_A = config["lambda_gen_A"]
        self.lambda_gen_B = config["lambda_gen_B"]
        self.lambda_cycle_a_rec = config["lambda_cycle_a_rec"]

        self.is_mse = config["is_mse"]
        self.is_gan = config["is_gan"]
        self.gen_a_rec = config["gen_a_rec"]

        self.is_train_gat = config["is_train_gat"]

        self.is_mse_log = config["is_mse_log"]

        self.lambda_disc_A = config["lambda_disc_A"]
        self.lambda_disc_B = config["lambda_disc_B"]

        self.num_epoch = config["num_epoch"]

        if config["is_mse_de_gene"]:
            if config["dataset"] == "l1000":
                mse_de_gene_path = config["l1000_mse_de_gene"]
            elif config["dataset"] == "sciplex3":
                mse_de_gene_path = config["sciplex3_mse_de_gene"]
            elif config["dataset"] == "sciplex4":
                mse_de_gene_path = config["sciplex4_mse_de_gene"]
            else:
                raise ValueError(config["dataset"] + "not support mse de gene")

            with open(mse_de_gene_path, "rb") as f:
                l1000_mse_de_gene = pickle.load(f)

            l1000_other_gene = [i for i in range(2000)]

            for i in l1000_mse_de_gene:
                l1000_other_gene.remove(i)

            mse_weight = {}
            for i in l1000_mse_de_gene:
                mse_weight[i] = config["lambda_mse_de_gene"]

            for i in l1000_other_gene:
                mse_weight[i] = config["lambda_mse_other_gene"]

            self.mse_weight = []

            for i in range(2000):
                self.mse_weight.append(mse_weight[i])

            self.mse_weight = torch.tensor(self.mse_weight, dtype=dtype).unsqueeze(0)

        else:
            self.mse_weight = None

        if config["dataset"] == "l1000":
            if config["is_drug_gat"]:
                self.drug_encoder = GATNet(dtype=dtype)
            else:
                self.drug_encoder = MLP(
                    [193, 256, 128],
                    batch_norm=config["batch_norm"],
                    last_activate=None,
                    dtype=dtype,
                )
        elif config["dataset"] == "sciplex3":
            if config["is_drug_gat"]:
                self.drug_encoder = GATNet(dtype=dtype)
            else:
                self.drug_encoder = MLP(
                    [178, 256, 128], batch_norm=True, dropout=0.0, dtype=dtype
                )
        elif config["dataset"] == "sciplex4":
            if config["is_drug_gat"]:
                self.drug_encoder = GATNet(dtype=dtype)
            else:
                self.drug_encoder = MLP(
                    [178, 256, 128], batch_norm=True, dropout=0.0, dtype=dtype
                )
        elif config["dataset"] == "cppa":
            if config["is_drug_gat"]:
                self.drug_encoder = GATNet(dtype=dtype)
            else:
                self.drug_encoder = MLP(
                    [178, 256, 128],
                    batch_norm=config["batch_norm"],
                    last_activate=None,
                    dtype=dtype,
                )
            self.dose_encoder = MLP(
                [1, 128],
                batch_norm=config["batch_norm"],
                last_activate=None,
                dtype=dtype,
            )
        elif config["dataset"] == "dixit":
            self.drug_encoder = MLP(
                [1, 128], batch_norm=True, dtype=dtype
            )
        elif config["dataset"] == "rep1":
            self.drug_encoder = MLP(
                [1, 128], batch_norm=True, dtype=dtype
            )
        elif config["dataset"] == "rep1k562":
            self.drug_encoder = MLP(
                [1, 128], batch_norm=True, dtype=dtype
            )

        model_dict = {
            "l1000": l1000_model,
            "sciplex3": sciplex3_model,
            "sciplex4": sciplex4_model,
            "dixit": dixit_model,
            "rep1": rep1_model,
            "rep1k562": rep1k562_model,
            "cppa": cppa_model,
        }
        (
            self.encoderG_A,
            self.decoderG_A,
            self.encoderG_B,
            self.decoderG_B,
            self.discriminator_A,
            self.discriminator_B,
        ) = model_dict[config["dataset"]](config, dtype)

    def pert_encoder(self, x, dose=None):
        if dose is None:
            return self.drug_encoder(x)

        # 矩阵对位相乘
        return torch.mul(self.drug_encoder(x), self.dose_encoder(dose))

    def netG_A(self, controls, drug_embedding):
        base_state = self.encoderG_A(controls)
        treat = self.decoderG_A(torch.add(base_state, drug_embedding))

        return treat

    def idtG_A(self, x):
        base_state = self.encoderG_A(x)
        out = self.decoderG_A(base_state)

        return out

    def netG_B(self, treat, drug_embedding):
        base_state = self.encoderG_B(treat)
        control = self.decoderG_B(torch.sub(base_state, drug_embedding))

        return control

    def idtG_B(self, x):
        base_state = self.encoderG_B(x)
        out = self.decoderG_B(base_state)

        return out

    def update_G(self, controls, treats, drugs, dose=None, epoch=None):
        self.set_requires_grad([self.discriminator_A, self.discriminator_B], False)

        if self.is_mse_log:
            self.mse_weight = (
                torch.abs(torch.log2(torch.div(treats + 1e-8, controls + 1e-8))) * 10
            )

        self.drug_embedding = self.pert_encoder(drugs, dose)

        self.fake_treat = self.netG_A(controls, self.drug_embedding)
        self.rec_control = self.netG_B(self.fake_treat, self.drug_embedding)
        self.fake_control = self.netG_B(treats, self.drug_embedding)
        self.rec_treat = self.netG_A(self.fake_control, self.drug_embedding)

        loss = {}

        if self.is_mse:
            self.loss_gen_A = self.loss_autoencoder(
                self.fake_treat, treats, self.mse_weight
            )
            self.loss_gen_B = self.loss_autoencoder(
                self.fake_control, controls, self.mse_weight
            )
            self.loss_G = (
                self.loss_gen_A * self.lambda_gen_A + self.loss_gen_B * self.lambda_gen_B
            )
            loss["loss_gen_A"] = self.loss_gen_A
            loss["loss_gen_B"] = self.loss_gen_B
            # print("mse")
        else:
            self.loss_G = 0

        self.idt_A = self.idtG_A(treats)
        self.loss_idt_A = self.loss_autoencoder(self.idt_A, treats, self.mse_weight)
        self.idt_B = self.idtG_B(controls)
        self.loss_idt_B = self.loss_autoencoder(self.idt_B, controls, self.mse_weight)

        self.loss_G += (
            self.loss_idt_A * self.lambda_B * self.lambda_idt
            + self.loss_idt_B * self.lambda_A * self.lambda_idt
        )

        loss["loss_idt_A"] = self.loss_idt_A
        loss["loss_idt_B"] = self.loss_idt_B

        if self.gen_a_rec:
            cycle_a_rec_control = self.idtG_A(controls)
            self.loss_cycle_a_rec = self.loss_autoencoder(
                cycle_a_rec_control, controls, self.mse_weight
            )
            self.loss_G += self.loss_cycle_a_rec * self.lambda_cycle_a_rec
            loss["loss_cycle_a_rec"] = self.loss_cycle_a_rec

        if self.is_gan:
            self.loss_D_A = self.loss_gan(self.discriminator_A(self.fake_treat), True)
            self.loss_D_B = self.loss_gan(self.discriminator_B(self.fake_control), True)
            self.loss_G += (
                self.loss_D_A * self.lambda_disc_A + self.loss_D_B * self.lambda_disc_B
            )
            loss["loss_D_A"] = self.loss_D_A
            loss["loss_D_B"] = self.loss_D_B
            # print("gan")

        self.loss_cycle_A = self.loss_autoencoder(self.rec_control, controls)
        self.loss_cycle_B = self.loss_autoencoder(self.rec_treat, treats)

        self.loss_G += (
            self.loss_cycle_A * self.lambda_A + self.loss_cycle_B * self.lambda_B
        )

        loss["loss_cycle_A"] = self.loss_cycle_A
        loss["loss_cycle_B"] = self.loss_cycle_B

        return self.loss_G, loss

    def updata_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.loss_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.loss_gan(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_D

    def update_D(self, controls, treats, epoch=None):
        self.set_requires_grad([self.discriminator_A, self.discriminator_B], True)

        self.loss_D_A = self.updata_D_basic(
            self.discriminator_A, treats, self.fake_treat.detach()
        )
        self.loss_D_B = self.updata_D_basic(
            self.discriminator_B, controls, self.fake_control.detach()
        )

        return self.loss_D_A, self.loss_D_B

    def forward(self, controls, treats, drugs, dose=None, update_G=True, epoch=None):
        if not self.is_train_gat:
            self.set_requires_grad([self.drug_encoder], False)

        if update_G:
            return self.update_G(controls, treats, drugs, dose, epoch)
        else:
            return self.update_D(controls, treats, epoch)

    def predict(self, controls, treats, drugs, dose=None):
        drug_embedding = self.pert_encoder(drugs, dose)
        fake_treats = self.netG_A(controls, drug_embedding)
        fake_controls = self.netG_B(treats, drug_embedding)

        return fake_controls, fake_treats

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
