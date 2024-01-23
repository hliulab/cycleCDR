import os
import time
import torch
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
from numpy import mean
from matplotlib import pyplot as plt

from .utils import get_batch_data
from .evaluate import evaluate_r2_CPA, evaluate_GEARS, compute_metrics


class Trainer:
    def __init__(
        self,
        model,
        num_epoch,
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer_DRUG=None,
        scheduler_DRUG=None,
        dataset_name="l1000",
        train_sampler=None,
        valid_sampler=None,
        is_mse=True,
        is_gan=False,
        config=None,
    ):
        self.model = model
        self.num_epoch = num_epoch
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_DRUG = optimizer_DRUG
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        self.scheduler_DRUG = scheduler_DRUG
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.dataset_name = dataset_name
        self.is_mse = is_mse
        self.is_gan = is_gan
        self.config_hash = hashlib.md5(str(config).encode()).hexdigest()

    def fit(self):
        self.loss_gen_A_list = []
        self.loss_gen_B_list = []
        self.loss_idt_A_list = []
        self.loss_idt_B_list = []
        self.loss_cycle_a_rec_list = []
        self.loss_cycle_A_list = []
        self.loss_cycle_B_list = []
        self.loss_D_A_list = []
        self.loss_D_B_list = []

        self.controls_r2_list = []
        self.treats_r2_list = []
        self.treats_r2_de_list = []
        self.controls_r2_cpa_list = []
        self.treats_r2_cpa_list = []
        self.treats_r2_cpa_de_list = []

        self.controls_explained_variance_list = []
        self.treats_explained_variance_list = []
        self.controls_explained_variance_cpa_list = []
        self.treats_explained_variance_cpa_list = []
        self.treats_explained_variance_de_list = []
        self.treats_explained_variance_cpa_de_list = []

        self.G_lr = []
        for epoch in range(self.num_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            loss_gen_A = []
            loss_gen_B = []
            loss_idt_A = []
            loss_idt_B = []
            loss_cycle_a_rec = []
            loss_cycle_A = []
            loss_cycle_B = []
            loss_D_A_sum = []
            loss_D_B_sum = []

            self.model.train()
            bar = tqdm(self.train_dataloader)
            for batch in bar:
                controls, treats, drugs, dose, _, _ = get_batch_data(
                    batch, self.dataset_name
                )

                loss_G, loss = self.model(
                    controls, treats, drugs, dose, update_G=True, epoch=epoch
                )
                self.optimizer_G.zero_grad()
                if self.optimizer_DRUG is not None:
                    self.optimizer_DRUG.zero_grad()
                loss_G.backward(retain_graph=True)
                self.optimizer_G.step()
                if self.optimizer_DRUG is not None:
                    self.optimizer_DRUG.step()

                if "loss_gen_A" in loss.keys():
                    loss_gen_A.append(loss["loss_gen_A"].item())
                    loss_gen_B.append(loss["loss_gen_B"].item())

                loss_idt_A.append(loss["loss_idt_A"].item())
                loss_idt_B.append(loss["loss_idt_B"].item())

                if "loss_cycle_a_rec" in loss.keys():
                    loss_cycle_a_rec.append(loss["loss_cycle_a_rec"].item())

                loss_cycle_A.append(loss["loss_cycle_A"].item())
                loss_cycle_B.append(loss["loss_cycle_B"].item())

                if self.is_gan:
                    loss_D_A, loss_D_B = self.model(
                        controls, treats, drugs, dose, update_G=False, epoch=epoch
                    )
                    self.optimizer_D.zero_grad()
                    loss_D_A.backward(retain_graph=True)
                    loss_D_B.backward()
                    self.optimizer_D.step()

                    loss_D_A_sum.append(loss_D_A.item())
                    loss_D_B_sum.append(loss_D_B.item())

                desc = f"Epoch: {epoch+1}/{self.num_epoch} loss_cycle_A: {mean(loss_cycle_A):.6f}"
                if self.is_gan:
                    desc += f" loss_D_A: {mean(loss_D_A_sum):.6f} loss_D_B: {mean(loss_D_B_sum):.6f}"
                if self.optimizer_DRUG is not None:
                    desc += f" drug_lr: {self.optimizer_DRUG.param_groups[0]['lr']:.6f}"
                bar.set_description(desc)

            if self.optimizer_DRUG is not None:
                self.G_lr.append(self.optimizer_DRUG.param_groups[0]['lr'])
            self.scheduler_G.step()
            if self.optimizer_DRUG is not None:
                self.scheduler_DRUG.step()

            if len(loss_gen_A) > 0:
                self.loss_gen_A_list.append(mean(loss_gen_A))
                self.loss_gen_B_list.append(mean(loss_gen_B))
            self.loss_idt_A_list.append(mean(loss_idt_A))
            self.loss_idt_B_list.append(mean(loss_idt_B))
            if len(loss_cycle_a_rec) > 0:
                self.loss_cycle_a_rec_list.append(mean(loss_cycle_a_rec))
            self.loss_cycle_A_list.append(mean(loss_cycle_A))
            self.loss_cycle_B_list.append(mean(loss_cycle_B))

            if self.is_gan:
                self.scheduler_D.step()
                self.loss_D_A_list.append(mean(loss_D_A_sum))
                self.loss_D_B_list.append(mean(loss_D_B_sum))

            self.model.eval()
            with torch.no_grad():
                if self.valid_sampler is not None:
                    self.valid_sampler.set_epoch(epoch)
                valid_res = evaluate_r2_CPA(
                    self.model, self.valid_dataloader, "Valid", self.dataset_name
                )

            self.valid_res = valid_res

            self.controls_r2_list.append(mean(valid_res["controls_r2_list"]))
            self.treats_r2_list.append(mean(valid_res["treats_r2_list"]))
            self.controls_r2_cpa_list.append(
                mean(list(valid_res["controls_r2_cpa_dict"].values()))
            )
            self.treats_r2_cpa_list.append(
                mean(list(valid_res["treats_r2_cpa_dict"].values()))
            )

            self.controls_explained_variance_list.append(
                mean(valid_res["controls_explained_variance_list"])
            )
            self.treats_explained_variance_list.append(
                mean(valid_res["treats_explained_variance_list"])
            )
            self.controls_explained_variance_cpa_list.append(
                mean(list(valid_res["controls_explained_variance_cpa_dict"].values()))
            )
            self.treats_explained_variance_cpa_list.append(
                mean(list(valid_res["treats_explained_variance_cpa_dict"].values()))
            )

            if len(valid_res["treats_r2_de_list"]) > 0:
                self.treats_r2_de_list.append(mean(valid_res["treats_r2_de_list"]))
                self.treats_r2_cpa_de_list.append(
                    mean(list(valid_res["treats_r2_cpa_de_dict"].values()))
                )
                self.treats_explained_variance_de_list.append(
                    mean(valid_res["treats_explained_variance_de_list"])
                )
                self.treats_explained_variance_cpa_de_list.append(
                    mean(
                        list(valid_res["treats_explained_variance_cpa_de_dict"].values())
                    )
                )

            print(
                f"Valid control_r2_cpa:{self.controls_r2_cpa_list[-1]:.6f}, treat_r2_cpa:{self.treats_r2_cpa_list[-1]:.6f}"  # noqa: E501
            )
            print(
                f"Valid control_explained_variance_cpa:{self.controls_explained_variance_cpa_list[-1]:.6f}, treat_explained_variance_cpa:{self.treats_explained_variance_cpa_list[-1]:.6f}"  # noqa: E501
            )
            if len(self.treats_r2_de_list) > 0:
                print(
                    f"Valid treat_r2_cpa_de:{self.treats_r2_cpa_de_list[-1]:.6f}"  # noqa: E501
                )
                print(
                    f"Valid treat_explained_variance_cpa_de:{self.treats_explained_variance_cpa_de_list[-1]:.6f}"  # noqa: E501
                )

        self.model.eval()
        with torch.no_grad():
            test_res = evaluate_r2_CPA(
                self.model, self.test_dataloader, "Test", self.dataset_name
            )

            train_res = evaluate_r2_CPA(
                self.model, self.train_dataloader, "Train", self.dataset_name
            )

            if self.dataset_name == "adamson":
                test_res = evaluate_GEARS(
                    self.model, self.test_dataloader, self.dataset_name
                )
                self.test_metrics, _ = compute_metrics(test_res)

        self.test_res = test_res
        self.train_res = train_res

        print(
            f"Test control_r2_cpa:{mean(list(test_res['controls_r2_cpa_dict'].values())):.6f}, treat_r2_cpa:{mean(list(test_res['treats_r2_cpa_dict'].values())):.6f}"  # noqa: E501
        )
        print(
            f"Test control_explained_variance_cpa:{mean(list(test_res['controls_explained_variance_cpa_dict'].values())):.6f}, treat_explained_variance_cpa:{mean(list(test_res['treats_explained_variance_cpa_dict'].values())):.6f}"  # noqa: E501
        )

        if len(test_res['treats_r2_de_list']) > 0:
            print(
                f"Test treat_r2_cpa_de:{mean(list(test_res['treats_r2_cpa_de_dict'].values())):.6f}"
            )
            print(
                f"Test treat_explained_variance_cpa_de:{mean(list(test_res['treats_explained_variance_cpa_de_dict'].values())):.6f}"  # noqa: E501
            )

        if self.dataset_name == "adamson":
            metrics = ['mse', 'pearson']
            for m in metrics:
                print(
                    {
                        'test_' + m: self.test_metrics[m],
                        'test_de_' + m: self.test_metrics[m + '_de'],
                    }
                )

        self.save_data(str(self.model.device))

    def save_data(self, device):
        module_path = f'./results/modules/{self.dataset_name}/{self.config_hash}'
        if not os.path.exists(module_path) and device == "cuda:0":
            os.makedirs(module_path)

        while not os.path.exists(module_path):
            time.sleep(1)

        module_name = f'{module_path}/cycleCDR_{self.dataset_name}_{device}.pkl'

        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), module_name)
        else:
            torch.save(self.model.state_dict(), module_name)

        data = {
            'loss_D_A_list': self.loss_D_A_list,
            'loss_D_B_list': self.loss_D_B_list,
            'train_res': self.train_res,
            'valid_res': self.valid_res,
            'test_res': self.test_res,
            'G_lr': self.G_lr,
        }

        if self.dataset_name == "adamson":
            data['test_metrics'] = self.test_metrics

        if len(self.treats_r2_cpa_de_list) > 0:
            data['treats_r2_de_list'] = self.treats_r2_de_list
            data['treats_r2_cpa_de_list'] = self.treats_r2_cpa_de_list

        plot_data_path = f'./results/plot_data/{self.dataset_name}/{self.config_hash}'
        if os.path.exists(plot_data_path) is False and device == "cuda:0":
            os.mkdir(plot_data_path)

        while os.path.exists(plot_data_path) is False:
            time.sleep(1)

        with open(
            f'{plot_data_path}/cycleCDR_{str(self.dataset_name)}_{str(device)}.pkl',
            'wb',
        ) as f:
            pickle.dump(data, f)

    def plot(self, device):
        if device != "cuda:0":
            return
        if (
            os.path.exists(f'./results/plot/{self.dataset_name}/{self.config_hash}')
            is False
        ):
            os.mkdir(f'./results/plot/{self.dataset_name}/{self.config_hash}')

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.controls_r2_list,
            'red',
            label=f' Max R2 = {max(self.controls_r2_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('control r2')
        plt.title("control reconstruct R2")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/control_r2.png"
        )
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.controls_r2_cpa_list,
            'red',
            label=f' Max R2 = {max(self.controls_r2_cpa_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('control r2')
        plt.title("control reconstruct R2 CPA")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/control_r2_cpa.png"
        )
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.treats_r2_list,
            'red',
            label=f' Max R2 = {max(self.treats_r2_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('treat r2')
        plt.title("treat reconstruct R2")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(f"./results/plot/{self.dataset_name}/{self.config_hash}/treat_r2.png")
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.treats_r2_cpa_list,
            'red',
            label=f' Max R2 = {max(self.treats_r2_cpa_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('treat r2')
        plt.title("treat reconstruct R2 CPA")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/treat_r2_cpa.png"
        )
        plt.cla()
        plt.clf()

        if len(self.treats_r2_de_list) != 0:
            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.treats_r2_de_list,
                'red',
                label=f' Max R2 = {max(self.treats_r2_de_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('treat r2')
            plt.title("treat reconstruct R2 DE")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/treat_r2_de.png"
            )
            plt.cla()
            plt.clf()

        if len(self.treats_r2_cpa_de_list) != 0:
            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.treats_r2_cpa_de_list,
                'red',
                label=f' Max R2 = {max(self.treats_r2_cpa_de_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('treat r2')
            plt.title("treat reconstruct R2 CPA DE")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/treat_r2_cpa_de.png"
            )
            plt.cla()
            plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.loss_idt_A_list,
            'red',
            label=f' loss = {min(self.loss_idt_A_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("idt a loss")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_idt_a.png"
        )
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.loss_idt_B_list,
            'red',
            label=f' loss = {min(self.loss_idt_B_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("idt b loss")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_idt_b.png"
        )
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.loss_cycle_A_list,
            'red',
            label=f' loss = {min(self.loss_cycle_A_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("cycle a loss")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_cycle_a.png"
        )
        plt.cla()
        plt.clf()

        x = np.arange(self.num_epoch)
        plt.plot(
            x,
            self.loss_cycle_B_list,
            'red',
            label=f' loss = {min(self.loss_cycle_B_list):.4f}',
        )
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("cycle b loss")
        plt.legend(loc="upper right", fontsize=10)
        plt.savefig(
            f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_cycle_b.png"
        )
        plt.cla()
        plt.clf()

        if len(self.loss_gen_A_list) > 0:
            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.loss_gen_A_list,
                'red',
                label=f' loss = {min(self.loss_gen_A_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title("gen a loss")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_gen_a.png"
            )
            plt.cla()
            plt.clf()

            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.loss_gen_B_list,
                'red',
                label=f' loss = {min(self.loss_gen_B_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title("gen b loss")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_gen_b.png"
            )
            plt.cla()
            plt.clf()

        if len(self.loss_cycle_a_rec_list) > 0:
            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.loss_cycle_a_rec_list,
                'red',
                label=f' loss = {min(self.loss_cycle_a_rec_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title("gen a rec loss")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/loss_gen_a_rec.png"
            )
            plt.cla()
            plt.clf()

        if self.is_gan:
            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.loss_D_A_list,
                'red',
                label=f' loss = {min(self.loss_D_A_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title("discriminator A loss")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/discriminator_A_loss.png"
            )
            plt.cla()
            plt.clf()

            x = np.arange(self.num_epoch)
            plt.plot(
                x,
                self.loss_D_B_list,
                'red',
                label=f' loss = {min(self.loss_D_B_list):.4f}',
            )
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title("discriminator B loss")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(
                f"./results/plot/{self.dataset_name}/{self.config_hash}/discriminator_B_loss.png"
            )
            plt.cla()
            plt.clf()

        if self.G_lr is not None and len(self.G_lr) > 0:
            x = np.arange(self.num_epoch)
            plt.plot(x, self.G_lr, 'red', label=f' lr = {max(self.G_lr):.4f}')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.title("lr")
            plt.legend(loc="upper right", fontsize=10)
            plt.savefig(f"./results/plot/{self.dataset_name}/{self.config_hash}/lr.png")
            plt.cla()
            plt.clf()
