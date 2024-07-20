import torch
import numpy as np

from sklearn.linear_model import LogisticRegression


def pseudo_label_unlabeled_set(model, labeled_x, labeled_y, unlabeled_x, device):
    with torch.no_grad():
        model.eval()
        zeta, _ = model.forward()
        zeta = zeta.detach().cpu().numpy()
        x_train = []
        for i, j in labeled_x:
            x_train.append(np.concatenate([zeta[i], zeta[j]]))
        # train the logistic regression model to predict the edge sign
        logistic_function = LogisticRegression(solver='lbfgs', max_iter=1000)
        logistic_function.fit(x_train, labeled_y.detach().cpu().numpy())
        # Transform the edges to label in numpy format
        x_pl = []
        for i, j in unlabeled_x:
            x_pl.append(np.concatenate([zeta[i], zeta[j]]))
        pred = logistic_function.predict(x_pl)
        return torch.Tensor(pred).to(device).float()
