from typing import List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric_signed_directed.nn.general import MSGNN_link_prediction

from losses import Sign_Product_Entropy_Loss, Sign_Direction_Loss, Sign_Triangle_Loss


class SDRLayer(nn.Module):
    r"""The signed directed relationship layer from
    `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

        Args:
            in_dim (int): Dimension of input features. Defaults to 20.
            out_dim (int): Dimension of output features. Defaults to 20.
            edge_lists (list): Edge list for current motifs.
    """

    def __init__(
            self,
            in_dim: int = 20,
            out_dim: int = 20,
            edge_lists: list = [],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.edge_lists = edge_lists
        self.aggs = []

        agg = GATConv

        for i in range(len(edge_lists)):
            self.aggs.append(
                agg(in_dim, out_dim)
            )
            self.add_module('agg_{}'.format(i), self.aggs[-1])

        self.mlp_layer = nn.Sequential(
            nn.Linear(in_dim * (len(edge_lists) + 1), out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim)
        )

    def reset_parameters(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)

        self.mlp_layer.apply(init_weights)
        for agg in self.aggs:
            agg.reset_parameters()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        neigh_feats = []
        for edges, agg in zip(self.edge_lists, self.aggs):
            x2 = agg(x, edges)
            neigh_feats.append(x2)
        combined = torch.cat([x] + neigh_feats, 1)
        combined = self.mlp_layer(combined)
        return combined


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.nn.Parameter(x, requires_grad=requires_grad)


class MySDGNN(nn.Module):
    r"""The SDGNN model from  `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

    Args:
        node_num (int, optional): The number of nodes.
        edge_index_s (LongTensor): The edgelist with sign. (e.g., :obj:`torch.LongTensor([[0, 1, -1], [0, 2, 1]])` )
        in_dim (int, optional): Size of each input sample features. Defaults to 20.
        out_dim (int): Size of each hidden embeddings. Defaults to 20.
        layer_num (int, optional): Number of layers. Defaults to 2.
        init_emb: (FloatTensor, optional): The initial embeddings. Defaults to :obj:`None`, which will use TSVD as initial embeddings.
        init_emb_grad (bool optional): Whether to set the initial embeddings to be trainable. (default: :obj:`False`)
        lamb_d (float, optional): Balances the direction loss contributions of the overall objective. (default: :obj:`1.0`)
        lamb_t (float, optional): Balances the triangle loss contributions of the overall objective. (default: :obj:`1.0`)
    """

    def __init__(
            self,
            node_num: int,
            edge_index_s,
            features: torch.Tensor,
            in_dim: int = 20,
            out_dim: int = 20,
            layer_num: int = 2,
            lamb_d: float = 5.0,
            lamb_t: float = 1.0,
            learnable_features: bool = False,
            reduction: str = 'mean',
            weights: torch.Tensor = None,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.tri_weight = None
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.device = edge_index_s.device
        self.lamb_d = lamb_d
        self.lamb_t = lamb_t
        # module to predict the edge sign
        self.sign_classification_net = nn.Sequential(nn.Linear(self.out_dim * 2, 1),
                                                     nn.Sigmoid())
        self.sign_classification_loss = torch.nn.BCELoss(reduction=reduction, weight=weights)
        self.sign_classification_loss_joint = torch.nn.BCELoss(reduction='none')

        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()

        self.x = features
        if learnable_features:
            self.x = nn.Parameter(self.x, requires_grad=True)

        self.adj_lists = self.build_adj_lists(edge_index_s)
        self.edge_lists = [self.map_adj_to_edges(i) for i in self.adj_lists]

        self.layers = []
        for i in range(layer_num):
            if i == 0:
                layer = SDRLayer(in_dim, out_dim,
                                 edge_lists=self.edge_lists)
            else:
                layer = SDRLayer(out_dim, out_dim,
                                 edge_lists=self.edge_lists)
            self.add_module(f'SDRLayer_{i}', layer)
            self.layers.append(layer)

        self.loss_sign = Sign_Product_Entropy_Loss()
        self.loss_direction = Sign_Direction_Loss(emb_dim=out_dim)
        self.loss_tri = Sign_Triangle_Loss(emb_dim=out_dim, edge_weight=self.tri_weight)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def map_adj_to_edges(self, adj_list: List) -> torch.LongTensor:
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = torch.LongTensor(edges).to(self.device)
        return edges.t()

    def get_features(self, u: int, v: int, r_edgelists: List) -> Tuple[int, int, int, int, int,
                                                                       int, int, int, int, int, int, int, int, int, int, int]:
        pos_in_edgelist, pos_out_edgelist, neg_in_edgelist, neg_out_edgelist = r_edgelists

        d1_1 = len(set(pos_out_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d1_2 = len(set(pos_out_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))
        d1_3 = len(set(neg_out_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d1_4 = len(set(neg_out_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))

        d2_1 = len(set(pos_out_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d2_2 = len(set(pos_out_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))
        d2_3 = len(set(neg_out_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d2_4 = len(set(neg_out_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))

        d3_1 = len(set(pos_in_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d3_2 = len(set(pos_in_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))
        d3_3 = len(set(neg_in_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d3_4 = len(set(neg_in_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))

        d4_1 = len(set(pos_in_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d4_2 = len(set(pos_in_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))
        d4_3 = len(set(neg_in_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d4_4 = len(set(neg_in_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))

        return (d1_1, d1_2, d1_3, d1_4, \
                d2_1, d2_2, d2_3, d2_4, \
                d3_1, d3_2, d3_3, d3_4, \
                d4_1, d4_2, d4_3, d4_4)

    def build_adj_lists(self, edge_index_s: torch.LongTensor) -> List:
        edge_index_s_list = edge_index_s.cpu().numpy().tolist()
        self.weight_dict = defaultdict(dict)

        pos_edgelist = defaultdict(set)
        pos_out_edgelist = defaultdict(set)
        pos_in_edgelist = defaultdict(set)
        neg_edgelist = defaultdict(set)
        neg_out_edgelist = defaultdict(set)
        neg_in_edgelist = defaultdict(set)

        for node_i, node_j, s in edge_index_s_list:

            if s > 0:
                pos_edgelist[node_i].add(node_j)
                pos_edgelist[node_j].add(node_i)

                pos_out_edgelist[node_i].add(node_j)
                pos_in_edgelist[node_j].add(node_i)
            if s < 0:
                neg_edgelist[node_i].add(node_j)
                neg_edgelist[node_j].add(node_i)

                neg_out_edgelist[node_i].add(node_j)
                neg_in_edgelist[node_j].add(node_i)

        r_edgelists = (pos_in_edgelist, pos_out_edgelist,
                       neg_in_edgelist, neg_out_edgelist)

        adj1 = pos_out_edgelist.copy()
        adj2 = neg_out_edgelist.copy()
        for i in adj1:
            for j in adj1[i]:
                v_list = self.get_features(i, j, r_edgelists)
                mask = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
                counts = np.dot(v_list, mask)
                self.weight_dict[i][j] = counts

        for i in adj2:
            for j in adj2[i]:
                v_list = self.get_features(i, j, r_edgelists)
                mask = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
                counts = np.dot(v_list, mask)
                self.weight_dict[i][j] = counts

        row = []
        col = []
        value = []
        for i in self.weight_dict:
            for j in self.weight_dict[i]:
                row.append(i)
                col.append(j)
                value.append(self.weight_dict[i][j])
        self.tri_weight = sp.csc_matrix((value, (row, col)),
                                        shape=(self.node_num, self.node_num))
        return [pos_out_edgelist, pos_in_edgelist, neg_out_edgelist, neg_in_edgelist]

    def forward(self) -> torch.FloatTensor:
        x = self.x
        for layer_m in self.layers:
            x = layer_m(x)
        return x

    def loss(self):
        z = self.forward()
        loss_sign = self.link_sign_loss(z, self.pos_edge_index, self.neg_edge_index)
        loss_direction = self.loss_direction(z, self.pos_edge_index, self.neg_edge_index)
        loss_triangle = self.loss_tri(z, self.pos_edge_index, self.neg_edge_index)
        return z, loss_sign + self.lamb_d * loss_direction + self.lamb_t * loss_triangle

    def link_sign_loss(self, z, pos_edge_index, neg_edge_index):
        z_pos_src = z[pos_edge_index[0], :]
        z_pos_tgt = z[pos_edge_index[1], :]
        z_pos = torch.concat([z_pos_src, z_pos_tgt], dim=1)
        z_neg_src = z[neg_edge_index[0], :]
        z_neg_tgt = z[neg_edge_index[1], :]
        z_neg = torch.concat([z_neg_src, z_neg_tgt], dim=1)
        pos_pred = torch.flatten(self.sign_classification_net(z_pos))
        neg_pred = torch.flatten(self.sign_classification_net(z_neg))
        pos_loss = self.sign_classification_loss(pos_pred, torch.ones_like(pos_pred))
        neg_loss = self.sign_classification_loss(neg_pred, torch.zeros_like(neg_pred))
        return pos_loss + neg_loss

    def link_sign_loss_joint(self,
                             z: torch.Tensor,
                             edge_index: torch.LongTensor,
                             edge_labels: torch.FloatTensor):
        z_src = z[edge_index[0], :]
        z_tgt = z[edge_index[1], :]
        z = torch.concat([z_src, z_tgt], dim=1)
        pred = torch.flatten(self.sign_classification_net(z))
        return self.sign_classification_loss_joint(pred, edge_labels)

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                if src is not None:
                    name_t, param_t = tgt
                    grad = src
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, to_var(param))


class MyMSGNN(MSGNN_link_prediction):
    r"""The MSGNN model for link prediction from the
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://proceedings.mlr.press/v198/he22c.html>`_ paper.

    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        conv_bias (bool, optional): Whether to use bias in the convolutional layers, default :obj:`True`.
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """

    def __init__(self, num_features: int, hidden: int = 2, q: float = 0.25, K: int = 2, label_dim: int = 2, \
                 activation: bool = True, trainable_q: bool = False, layer: int = 2, dropout: float = 0.5,
                 normalization: str = 'sym',
                 cached: bool = False, conv_bias: bool = True, absolute_degree: bool = True):
        super(MyMSGNN, self).__init__(num_features=num_features, hidden=num_features,q=q, K=K, label_dim=label_dim,
                                      activation=activation, trainable_q=trainable_q, layer=layer, dropout=dropout,
                                      normalization=normalization, cached=cached, conv_bias=conv_bias,
                                      absolute_degree=absolute_degree)
        self.linear = nn.Linear(hidden * 4, 1)
        self.dropout = dropout
        self.output_fn = nn.Sigmoid()

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
                query_edges: torch.LongTensor, edge_weight: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat(
            (real[query_edges[:, 0]], real[query_edges[:, 1]], imag[query_edges[:, 0]], imag[query_edges[:, 1]]),
            dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        self.z = x.clone()
        x = self.linear(x)
        return self.output_fn(x)


class MLP(nn.Module):
    def __init__(self, node_num, features, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.node_num = node_num
        self.out_dim = 1

        # Features
        self.features = features
        self.in_dim = self.features.shape[1] * 2

        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        ])

        # Output layer
        self.output_fn = nn.Sigmoid()

        # Custom loss component
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x) -> torch.FloatTensor:
        x = torch.concat([self.features[x[:, 0]], self.features[x[:, 1]]], dim=1)  # Bx2d
        for layer in self.layers:
            x = layer(x)
        return self.output_fn(x)

    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def params(self):
        for name, param in self.named_parameters():
            yield param


class EstimateAdj(nn.Module):
    # from https://github.com/Alex-Zeyu/RSGNN/commit/2e672737b5e8f0fbaea50f4ce505ee35de0f8d42
    def __init__(self, adj, symmetric=True):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric

    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            self.estimated_adj.data = (self.estimated_adj.data + self.estimated_adj.t()) / 2

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
