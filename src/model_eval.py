import torch

from typing import Tuple

import numpy as np
from sklearn import linear_model, metrics


def link_sign_prediction_logistic_function(embeddings: np.ndarray, train_X: np.ndarray, train_y: np.ndarray,
                                           test_X: np.ndarray, test_y: np.ndarray,
                                           sample_weight=None) -> Tuple[float, float, float, float, float]:
    """
    link_sign_prediction_logistic_function [summary]
    Link sign prediction is a binary classification machine learning task.
    It will return the metrics for link sign prediction (i.e., Accuracy, Binary-F1, Macro-F1, Micro-F1 and AUC).

    Args:
        embeddings (np.ndarray): The embeddings for signed graph.
        train_X (np.ndarray): The indices for training data (e.g., [[0, 1], [0, 2]])
        train_y (np.ndarray): The sign for training data (e.g., [[1, -1]])
        test_X (np.ndarray): The indices for test data (e.g., [[1, 2]])
        test_y (np.ndarray): The sign for test data (e.g., [[1]])
        class_weight (Union[dict, str], optional): Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. Defaults to None.

    Returns:
        [type]: The metrics for link sign prediction task.
        Tuple[float,float,float,float,float]: Accuracy, Binary-F1, Macro-F1, Micro-F1 and AUC.
        :param test_y:
        :param test_X:
        :param train_y:
        :param train_X:
        :param embeddings:
        :param sample_weight:

    """
    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    logistic_function = linear_model.LogisticRegression(
        solver='lbfgs', max_iter=1000)
    logistic_function.fit(train_X1, train_y, sample_weight=sample_weight)
    pred = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)
    accuracy = metrics.accuracy_score(test_y, pred)
    f1 = metrics.f1_score(test_y, pred)
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    auc_score = metrics.roc_auc_score(test_y, pred_p[:, 1])
    return accuracy, f1, f1_macro, f1_micro, auc_score


def eval_model(model, train_edge_index, test_edge_index, train_ground_truth, test_ground_truth, sample_weight=None,
               using_lr=False):
    # predict on input data
    model.eval()
    if using_lr:
        with torch.no_grad():
            zeta = model.forward()
            zeta = zeta.detach().cpu()
        metrics_list = link_sign_prediction_logistic_function(embeddings=zeta, train_X=train_edge_index,
                                                              train_y=train_ground_truth, test_X=test_edge_index,
                                                              test_y=test_ground_truth, sample_weight=sample_weight)
    else:
        with torch.no_grad():
            zeta = model.forward()
            zeta_src = zeta[test_edge_index[:, 0], :]
            zeta_tgt = zeta[test_edge_index[:, 1], :]
            zeta = torch.concat([zeta_src, zeta_tgt], dim=1)
            pred_prob = model.sign_classification_net(zeta).detach().cpu().numpy().flatten()
            pred_label = pred_prob > 0.5
            accuracy = metrics.accuracy_score(test_ground_truth, pred_label)
            f1 = metrics.f1_score(test_ground_truth, pred_label)
            f1_macro = metrics.f1_score(test_ground_truth, pred_label, average='macro')
            f1_micro = metrics.f1_score(test_ground_truth, pred_label, average='micro')
            auc_score = metrics.roc_auc_score(test_ground_truth, pred_prob)
            ap_score = metrics.average_precision_score(test_ground_truth, pred_prob)
            precision, recall, _ = metrics.precision_recall_curve(test_ground_truth, pred_prob)
            pr_auc = metrics.auc(recall, precision)
            metrics_list = [accuracy, f1, f1_macro, f1_micro, auc_score, ap_score, pr_auc]

    return {'accuracy': metrics_list[0], 'f1': metrics_list[1], 'f1_macro': metrics_list[2],
            'f1_micro': metrics_list[3], 'auc_score': metrics_list[4], 'ap': metrics_list[5], 'pr_auc': metrics_list[6]}

def eval_robust_model(model, x, adj, test_edge_index, test_ground_truth):
    with torch.no_grad():
        zeta = model.forward(x, adj)
        pred_prob = model.discriminate(zeta, test_edge_index.T).detach().cpu().numpy().flatten()
        pred_label = pred_prob > 0.5
        accuracy = metrics.accuracy_score(test_ground_truth, pred_label)
        f1 = metrics.f1_score(test_ground_truth, pred_label)
        f1_macro = metrics.f1_score(test_ground_truth, pred_label, average='macro')
        f1_micro = metrics.f1_score(test_ground_truth, pred_label, average='micro')
        auc_score = metrics.roc_auc_score(test_ground_truth, pred_prob)
        ap_score = metrics.average_precision_score(test_ground_truth, pred_prob)
        precision, recall, _ = metrics.precision_recall_curve(test_ground_truth, pred_prob)
        pr_auc = metrics.auc(recall, precision)
        metrics_list = [accuracy, f1, f1_macro, f1_micro, auc_score, ap_score, pr_auc]

    return {'accuracy': metrics_list[0], 'f1': metrics_list[1], 'f1_macro': metrics_list[2],
            'f1_micro': metrics_list[3], 'auc_score': metrics_list[4], 'ap': metrics_list[5], 'pr_auc': metrics_list[6]}


class TrainLogMetrics:
    def __init__(self, num_splits, loss_types_list):
        self.train_loss_dict = {}
        self.val_metrics_dict = {}
        for run_id in range(num_splits):
            self.train_loss_dict[run_id] = {loss_type: [] for loss_type in loss_types_list}
            self.val_metrics_dict[run_id] = []

    def train_update(self, run_id, loss_type, value):
        self.train_loss_dict[run_id][loss_type].append(value)

    def val_update(self, run_id, value):
        self.val_metrics_dict[run_id].append(value)


class TestLogMetrics:
    def __init__(self, num_splits, metric_names_list):
        self.test_metrics_dict = {}
        for metric in metric_names_list:
            self.test_metrics_dict[metric] = [None] * num_splits

    def update(self, metric_name, run_id, value):
        self.test_metrics_dict[metric_name][run_id] = value

    def get_metric_stats(self, metric_name, float_precision=4):
        avg_val, std_val = np.mean(self.test_metrics_dict[metric_name]), np.std(self.test_metrics_dict[metric_name])
        return round(avg_val, float_precision), round(std_val, float_precision)

