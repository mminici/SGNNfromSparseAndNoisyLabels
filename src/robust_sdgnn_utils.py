import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch_geometric.utils import (structured_negative_sampling)


class SignedConv(nn.Module):
    def __init__(self, in_channels, out_channels, first_aggr: bool, cuda=False):
        super(SignedConv, self).__init__()
        self.adj = None
        self.pos_mask = None
        self.neg_mask = None
        self.pos_adj = None
        self.neg_adj = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos = Linear(2 * self.in_channels, self.out_channels // 2)
            self.lin_neg = Linear(2 * self.in_channels, self.out_channels // 2)
        else:
            self.lin_pos = Linear(3 * self.in_channels // 2, self.out_channels // 2)
            self.lin_neg = Linear(3 * self.in_channels // 2, self.out_channels // 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_neg.reset_parameters()

    def forward(self, x, adj):
        self.adj = adj
        self.pos_mask = (self.adj >= 0).int()
        self.neg_mask = (self.adj <= 0).int()
        self.pos_adj = self.adj * self.pos_mask
        self.neg_adj = self.adj * self.neg_mask * -1

        self.pos_adj = F.normalize(self.pos_adj, p=1)
        self.neg_adj = F.normalize(self.neg_adj, p=1)

        if self.first_aggr:
            out_pos = torch.cat([torch.mm(self.pos_adj, x), x], dim=1)
            out_pos = self.lin_pos(out_pos)

            out_neg = torch.cat([torch.mm(self.neg_adj, x), x], dim=1)
            out_neg = self.lin_neg(out_neg)

            return torch.cat([out_pos, out_neg], dim=1)
        else:
            x_dim = x.shape[1]
            x_pos = x[:, 0:x_dim // 2]
            x_neg = x[:, x_dim // 2:]

            out_pos = torch.cat([torch.mm(self.pos_adj, x_pos),
                                 torch.mm(self.neg_adj, x_neg),
                                 x_pos], dim=1)
            out_pos = self.lin_pos(out_pos)

            out_neg = torch.cat([torch.mm(self.pos_adj, x_neg),
                                 torch.mm(self.neg_adj, x_pos),
                                 x_neg], dim=1)
            out_neg = self.lin_neg(out_neg)

            return torch.cat([out_pos, out_neg], dim=1)

    def _normalized(self, adj):
        deg = adj.sum(dim=1, keepdim=True)
        adj = adj / deg
        adj.nan_to_num_(nan=0.0)
        return adj


class SignedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 lamb=5, bias=True, op='cat'):  # 64 64
        """
        The signed graph convolutional network model
        Args:
            in_channels:
            hidden_channels:
            num_layers:
            lamb:
            bias:
            op: edge representation operation ["cat", "mean", "add"]
        """
        super(SignedGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb
        self.op = "cat"

        self.conv1 = SignedConv(in_channels, hidden_channels, first_aggr=True)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SignedConv(hidden_channels, hidden_channels,
                                         first_aggr=False))
        if op == "cat":
            self.lin = nn.Sequential(nn.Linear(2 * hidden_channels, 1), nn.Sigmoid())
        else:
            self.lin = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())
        self.sign_classification_loss = torch.nn.BCELoss()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin._modules['0'].reset_parameters()

    def forward(self, x, adj):
        z = torch.tanh(self.conv1(x, adj))
        for conv in self.convs:
            z = torch.tanh(conv(z, adj))
        return z

    def discriminate(self, z, edge_index):
        if self.op == "cat":
            edge_feature = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        elif self.op == "mean":
            edge_feature = (z[edge_index[0]] + z[edge_index[1]]) / 2
        elif self.op == "add":
            edge_feature = z[edge_index[0]] + z[edge_index[1]]
        else:
            raise Exception(f'{self.op} op is not allowed. Select one between [cat, mean, add]')
        logits = self.lin(edge_feature)
        return logits

    def nll_loss(self, z, pos_edge_index, neg_edge_index, w_s=[1, 1]):
        pos_pred = self.discriminate(z, pos_edge_index)
        neg_pred = self.discriminate(z, neg_edge_index)
        nll_loss = self.sign_classification_loss(pos_pred,
                                                 torch.ones_like(pos_pred)) * w_s[0]
        nll_loss += self.sign_classification_loss(neg_pred,
                                                  torch.zeros_like(neg_pred)) * w_s[1]
        return nll_loss

    def pos_embedding_loss(self, z, pos_edge_index):
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_pos = self.pos_embedding_loss(z, pos_edge_index)
        loss_neg = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_pos + loss_neg)
        # return nll_loss


def feature_loss(x, estimate_adj, gamma_1=1., gamma_2=2.):
    """
    return the loss of feature smoothness between
    Args:
        gamma_1:
        gamma_2:
    Returns:
    """
    zeros = estimate_adj.new_zeros(estimate_adj.shape)
    adj_pos = torch.where(estimate_adj >= 0,
                          estimate_adj, zeros)
    adj_neg = torch.where(estimate_adj <= 0,
                          estimate_adj, zeros)
    D_pos = torch.diag(adj_pos.sum(dim=1))
    D_neg = torch.diag(adj_neg.sum(dim=1))
    ret = gamma_1 * torch.trace(x.T @ (D_pos - adj_pos) @ x) - \
          gamma_2 * torch.trace(x.T @ (D_neg - adj_neg) @ x)
    return ret
