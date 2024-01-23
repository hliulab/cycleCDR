import math
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, explained_variance_score

from .utils import get_batch_data

sys.path.append(os.getcwd())
from cycleCDR.model import cycleCDR


def compute_r2(y_true, y_pred):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    t = y_true.cpu().numpy().tolist()
    p = y_pred.cpu().numpy().tolist()
    # metric = R2Score().to(y_true.device)  # cppa
    # metric.update(y_pred, y_true)  # same as sklearn.r2_score(y_true, y_pred)
    # res = metric.compute().item()
    res = r2_score(t, p)
    return max(res, 0)


def compute_explained_variance_score(y_true, y_pred):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    t = y_true.cpu().numpy().tolist()
    p = y_pred.cpu().numpy().tolist()
    res = explained_variance_score(t, p)
    return res


def compute_pearsonr_score(y_true: Tensor, y_pred: Tensor):
    '''
    计算 pearsonr
    '''

    p = np.corrcoef(y_true.cpu().numpy(), y_pred.cpu().numpy())

    if math.isnan(p[0, 1]):
        return 0.0
    return p[0, 1]


def evaluate_r2_CPA(model: cycleCDR, dataloader, method="Valid", dataset_name="l1000"):
    true_controls_dict = {}
    pred_controls_dict = {}
    true_treats_dict = {}
    pred_treats_dict = {}

    de_gene_idxs_dict = {}

    controls_r2_list = []
    controls_r2_cpa_dict = {}
    treats_r2_list = []
    treats_r2_de_list = []
    treats_r2_cpa_dict = {}
    treats_r2_cpa_de_dict = {}

    controls_pearsonr_list = []
    controls_pearsonr_cpa_dict = {}
    treats_pearsonr_list = []
    treats_pearsonr_de_list = []
    treats_pearsonr_cpa_dict = {}
    treats_pearsonr_cpa_de_dict = {}

    controls_explained_variance_list = []
    controls_explained_variance_cpa_dict = {}
    treats_explained_variance_list = []
    treats_explained_variance_de_list = []
    treats_explained_variance_cpa_dict = {}
    treats_explained_variance_cpa_de_dict = {}

    bar = tqdm(dataloader)
    for batch in bar:
        controls, treats, drugs, dose, de_gene_idxs, pert_names = get_batch_data(
            batch, dataset_name
        )

        if "predict" in dir(model):
            pred_controls, pred_treats = model.predict(controls, treats, drugs, dose)
        else:
            pred_controls, pred_treats = model.module.predict(
                controls, treats, drugs, dose
            )

        controls_r2 = [
            compute_r2(controls[i], pred_controls[i]) for i in range(controls.shape[0])
        ]
        treats_r2 = [
            compute_r2(treats[i], pred_treats[i]) for i in range(treats.shape[0])
        ]
        controls_explained_variance = [
            compute_explained_variance_score(controls[i], pred_controls[i])
            for i in range(controls.shape[0])
        ]
        treats_explained_variance = [
            compute_explained_variance_score(treats[i], pred_treats[i])
            for i in range(treats.shape[0])
        ]
        controls_pearsonr = [
            compute_pearsonr_score(controls[i], pred_controls[i])
            for i in range(controls.shape[0])
        ]
        treats_pearsonr = [
            compute_pearsonr_score(treats[i], pred_treats[i])
            for i in range(treats.shape[0])
        ]

        controls_r2_list.extend(controls_r2)
        treats_r2_list.extend(treats_r2)
        controls_explained_variance_list.extend(controls_explained_variance)
        treats_explained_variance_list.extend(treats_explained_variance)
        controls_pearsonr_list.extend(controls_pearsonr)
        treats_pearsonr_list.extend(treats_pearsonr)

        if de_gene_idxs is not None:
            for i in range(treats.shape[0]):
                idx = de_gene_idxs[i].tolist()
                if -1 not in idx and len(idx) == 50:
                    treats_r2_de = compute_r2(treats[i, idx], pred_treats[i, idx])
                    treats_explained_variance_de = compute_explained_variance_score(
                        treats[i, idx], pred_treats[i, idx]
                    )
                    treats_pearsonr_de = compute_pearsonr_score(
                        treats[i, idx], pred_treats[i, idx]
                    )

                    treats_r2_de_list.append(treats_r2_de)
                    treats_explained_variance_de_list.append(treats_explained_variance_de)
                    treats_pearsonr_de_list.append(treats_pearsonr_de)

        for pert in set(pert_names):
            # pert_names 中等于 pert 的索引
            idx = [i for i, x in enumerate(pert_names) if x == pert]

            if pert not in true_controls_dict.keys():
                true_controls_dict[pert] = []
                pred_controls_dict[pert] = []
                true_treats_dict[pert] = []
                pred_treats_dict[pert] = []

            for id in idx:
                true_controls_dict[pert].append(controls[id].cpu())
                pred_controls_dict[pert].append(pred_controls[id].cpu())
                true_treats_dict[pert].append(treats[id].cpu())
                pred_treats_dict[pert].append(pred_treats[id].cpu())

            if de_gene_idxs is not None:
                de_gene_idxs_dict[pert] = de_gene_idxs[idx[0]]

        # desc = f"{method} controls_r2: {mean(controls_r2_list):.6f} treats_r2: {mean(treats_r2_list):.6f} "
        # desc += f"controls_explained_variance: {mean(controls_explained_variance_list):.6f} treats_explained_variance: {mean(treats_explained_variance_list):.6f}"  # noqa
        bar.set_description(f"{method}:")

    for pert in true_controls_dict.keys():
        temp_true_controls = (
            torch.stack(true_controls_dict[pert]).mean(0).detach().cuda().squeeze(0)
        )
        temp_pred_controls = (
            torch.stack(pred_controls_dict[pert]).mean(0).detach().cuda().squeeze(0)
        )

        controls_r2_cpa = compute_r2(temp_true_controls, temp_pred_controls)
        controls_r2_cpa_dict[pert] = controls_r2_cpa
        controls_explained_variance_cpa = compute_explained_variance_score(
            temp_true_controls, temp_pred_controls
        )
        controls_explained_variance_cpa_dict[pert] = controls_explained_variance_cpa
        controls_pearsonr_cpa = compute_pearsonr_score(
            temp_true_controls, temp_pred_controls
        )
        controls_pearsonr_cpa_dict[pert] = controls_pearsonr_cpa

        temp_true_treats = (
            torch.stack(true_treats_dict[pert]).mean(0).detach().cuda().squeeze(0)
        )
        temp_pred_treats = (
            torch.stack(pred_treats_dict[pert]).mean(0).detach().cuda().squeeze(0)
        )
        treats_r2_cpa = compute_r2(temp_true_treats, temp_pred_treats)
        treats_r2_cpa_dict[pert] = treats_r2_cpa
        treats_explained_variance_cpa = compute_explained_variance_score(
            temp_true_treats, temp_pred_treats
        )
        treats_explained_variance_cpa_dict[pert] = treats_explained_variance_cpa
        treats_pearsonr_cpa = compute_pearsonr_score(temp_true_treats, temp_pred_treats)
        treats_pearsonr_cpa_dict[pert] = treats_pearsonr_cpa

        if len(de_gene_idxs_dict.keys()) > 0 and -1 not in de_gene_idxs_dict[pert]:
            treats_r2_cpa_de = compute_r2(
                temp_true_treats[de_gene_idxs_dict[pert]],
                temp_pred_treats[de_gene_idxs_dict[pert]],
            )
            treats_r2_cpa_de_dict[pert] = treats_r2_cpa_de
            treats_explained_variance_cpa_de = compute_explained_variance_score(
                temp_true_treats[de_gene_idxs_dict[pert]],
                temp_pred_treats[de_gene_idxs_dict[pert]],
            )
            treats_explained_variance_cpa_de_dict[pert] = treats_explained_variance_cpa_de
            treats_pearsonr_cpa_de = compute_pearsonr_score(
                temp_true_treats[de_gene_idxs_dict[pert]],
                temp_pred_treats[de_gene_idxs_dict[pert]],
            )
            treats_pearsonr_cpa_de_dict[pert] = treats_pearsonr_cpa_de

    res = {
        "controls_r2_list": controls_r2_list,
        "treats_r2_list": treats_r2_list,
        "treats_r2_de_list": treats_r2_de_list,
        "controls_r2_cpa_dict": controls_r2_cpa_dict,
        "treats_r2_cpa_dict": treats_r2_cpa_dict,
        "treats_r2_cpa_de_dict": treats_r2_cpa_de_dict,
        "controls_pearsonr_list": controls_pearsonr_list,
        "treats_pearsonr_list": treats_pearsonr_list,
        "treats_pearsonr_de_list": treats_pearsonr_de_list,
        "controls_pearsonr_cpa_dict": controls_pearsonr_cpa_dict,
        "treats_pearsonr_cpa_dict": treats_pearsonr_cpa_dict,
        "treats_pearsonr_cpa_de_dict": treats_pearsonr_cpa_de_dict,
        "controls_explained_variance_list": controls_explained_variance_list,
        "treats_explained_variance_list": treats_explained_variance_list,
        "treats_explained_variance_de_list": treats_explained_variance_de_list,
        "controls_explained_variance_cpa_dict": controls_explained_variance_cpa_dict,
        "treats_explained_variance_cpa_dict": treats_explained_variance_cpa_dict,
        "treats_explained_variance_cpa_de_dict": treats_explained_variance_cpa_de_dict,
        "true_treats_dict": true_treats_dict,
        "pred_treats_dict": pred_treats_dict,
        "true_controls_dict": true_controls_dict,
    }

    return res


def compute_metrics(results):
    metrics = {}
    metrics_pert = {}

    metric2fct = {'mse': mse, 'pearson': pearsonr}

    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in results['pert_names']:
        metrics_pert[pert] = {}

        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(
                    results['pred_treats'][pert].mean(0),
                    results['truth_treats'][pert].mean(0),
                )[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(
                    results['pred_treats'][pert].mean(0),
                    results['truth_treats'][pert].mean(0),
                )

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

        if pert != 'ctrl':
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(
                        results['pred_treats_de'][pert].mean(0),
                        results['truth_treats_de'][pert].mean(0),
                    )[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(
                        results['pred_treats_de'][pert].mean(0),
                        results['truth_treats_de'][pert].mean(0),
                    )

                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0

    for m in metric2fct.keys():
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])

    return metrics, metrics_pert


def evaluate_GEARS(model: cycleCDR, dataloader, dataset_name):
    pred_treats_dict = {}
    truth_treats_dict = {}
    pred_treats_de = {}
    truth_treats_de = {}
    results = {}
    pert_names_set = set()

    bar = tqdm(dataloader)
    for batch in bar:
        controls, treats, perts, dose, de_index, pert_names = get_batch_data(
            batch, dataset_name
        )

        pert_names_set.update(pert_names)

        if "predict" in dir(model):
            _, pred_treats = model.predict(controls, treats, perts, None)
        else:
            _, pred_treats = model.module.predict(controls, treats, perts, None)

        for i in range(len(pert_names)):
            if pert_names[i] not in pred_treats_dict.keys():
                pred_treats_dict[pert_names[i]] = []
                truth_treats_dict[pert_names[i]] = []
                pred_treats_de[pert_names[i]] = []
                truth_treats_de[pert_names[i]] = []

            pred_treats_dict[pert_names[i]].append(pred_treats[i].cpu())
            truth_treats_dict[pert_names[i]].append(treats[i].cpu())
            pred_treats_de[pert_names[i]].append(pred_treats[i, de_index[i]].cpu())
            truth_treats_de[pert_names[i]].append(treats[i, de_index[i]].cpu())

    for pert_name in pred_treats_dict.keys():
        pred_treats_dict[pert_name] = (
            torch.stack(pred_treats_dict[pert_name]).detach().cpu().numpy()
        )
        truth_treats_dict[pert_name] = (
            torch.stack(truth_treats_dict[pert_name]).detach().cpu().numpy()
        )
        pred_treats_de[pert_name] = (
            torch.stack(pred_treats_de[pert_name]).detach().cpu().numpy()
        )
        truth_treats_de[pert_name] = (
            torch.stack(truth_treats_de[pert_name]).detach().cpu().numpy()
        )

    results['pred_treats'] = pred_treats_dict
    results['truth_treats'] = truth_treats_dict
    results['pred_treats_de'] = pred_treats_de
    results['truth_treats_de'] = truth_treats_de
    results['pert_names'] = list(pert_names_set)

    return results
