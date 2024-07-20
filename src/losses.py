import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)


class Sign_Triangle_Loss(nn.Module):
    r"""An implementation of the Signed Triangle Loss used in
     `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

    Args:
        emb_dim (int): The embedding size.
    """

    def __init__(self,
                 emb_dim: int,
                 edge_weight: sp.csc_matrix
                 ) -> None:
        super().__init__()
        self.lin = nn.Linear(emb_dim * 2, 1)
        self.edge_weight = edge_weight

    def forward(
            self,
            z: torch.Tensor,
            pos_edge_index: torch.LongTensor,
            neg_edge_index: torch.LongTensor
    ) -> torch.Tensor:
        device = z.device
        z_11 = z[pos_edge_index[0], :]
        z_12 = z[pos_edge_index[1], :]
        ind1 = pos_edge_index[0].cpu().numpy().tolist()
        ind2 = pos_edge_index[1].cpu().numpy().tolist()
        edge_w1 = torch.from_numpy(self.edge_weight[ind1, ind2]).reshape(-1, 1).to(device)

        z_21 = z[neg_edge_index[0], :]
        z_22 = z[neg_edge_index[1], :]
        ind1 = neg_edge_index[0].cpu().numpy().tolist()
        ind2 = neg_edge_index[1].cpu().numpy().tolist()
        edge_w2 = torch.from_numpy(self.edge_weight[ind1, ind2]).reshape(-1, 1).to(device)

        rs1 = self.lin(torch.cat([z_11, z_12], dim=1))
        rs2 = self.lin(torch.cat([z_21, z_22], dim=1))

        pos_loss = F.binary_cross_entropy_with_logits(rs1, torch.ones_like(rs1), weight=edge_w1, reduction='mean')

        neg_loss = F.binary_cross_entropy_with_logits(rs2, torch.zeros_like(rs2), weight=edge_w2, reduction='mean')

        return pos_loss + neg_loss


class Sign_Direction_Loss(nn.Module):
    r"""An implementation of the Signed Direction Loss used in
     `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

    Args:
        emb_dim (int): The embedding size.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.score_function1 = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        self.score_function2 = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            z: torch.Tensor,
            pos_edge_index: torch.LongTensor,
            neg_edge_index: torch.LongTensor
    ) -> torch.Tensor:
        z_11 = z[pos_edge_index[0], :]
        z_12 = z[pos_edge_index[1], :]

        z_21 = z[neg_edge_index[0], :]
        z_22 = z[neg_edge_index[1], :]

        s1 = self.score_function1(z_11)
        s2 = self.score_function2(z_12)
        q = torch.where((s1 - s2) > -0.5,
                        torch.ones_like(s1) * -0.5, s1 - s2)
        tmp = (q - (s1 - s2))
        pos_loss = torch.einsum("ij,ij->i", [tmp, tmp]).mean()

        s1 = self.score_function1(z_21)
        s2 = self.score_function2(z_22)
        q = torch.where((s1 - s2) > 0.5,
                        s1 - s2, torch.ones_like(s1) * 0.5)
        tmp = (q - (s1 - s2))
        neg_loss = torch.einsum("ij,ij->i", [tmp, tmp]).mean()
        return pos_loss + neg_loss


class Sign_Product_Entropy_Loss(nn.Module):
    r"""An implementation of the Signed Entropy Loss used in
     `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigma = torch.nn.Sigmoid()

    def build_weights(self, z, edge_index, labels):
        z_source_node_edge1 = z[edge_index[:, 0, 0]]
        z_target_node_edge1 = z[edge_index[:, 0, 1]]
        z_source_node_edge2 = z[edge_index[:, 1, 0]]
        z_target_node_edge2 = z[edge_index[:, 1, 1]]
        edge1_product = torch.einsum("ij, ij->i", [z_source_node_edge1, z_target_node_edge1])
        edge2_product = torch.einsum("ij, ij->i", [z_source_node_edge2, z_target_node_edge2])
        edge1_mask = torch.Tensor(labels[:, 0] == 1).bool()
        edge2_mask = torch.Tensor(labels[:, 1] == 1).bool()
        sigma_edge1 = self.sigma(edge1_product)
        sigma_edge2 = self.sigma(edge2_product)
        weight_edge1 = sigma_edge1 * edge1_mask + (1 - sigma_edge1) * ~edge1_mask
        weight_edge2 = sigma_edge2 * edge2_mask + (1 - sigma_edge2) * ~edge2_mask
        return weight_edge1 * weight_edge2

    def forward(
            self,
            z: torch.Tensor,
            pos_edge_index: torch.LongTensor,
            neg_edge_index: torch.LongTensor,
            known_pos_edge_index: torch.LongTensor = None,
            known_pos_labels: torch.FloatTensor = None,
            known_neg_edge_index: torch.LongTensor = None,
            known_neg_labels: torch.FloatTensor = None,
            reduction='mean'
    ) -> torch.Tensor:
        z_11 = z[pos_edge_index[0], :]
        z_12 = z[pos_edge_index[1], :]

        z_21 = z[neg_edge_index[0], :]
        z_22 = z[neg_edge_index[1], :]

        if known_pos_edge_index is not None and known_pos_labels is not None:
            pos_weights = self.build_weights(z, known_pos_edge_index, known_pos_labels).detach()
        else:
            pos_weights = None
        if known_neg_edge_index is not None and known_neg_labels is not None:
            neg_weights = self.build_weights(z, known_neg_edge_index, known_neg_labels).detach()
        else:
            neg_weights = None

        product1 = torch.einsum("ij, ij->i", [z_11, z_12])
        product2 = torch.einsum("ij, ij->i", [z_21, z_22])
        loss_pos = F.binary_cross_entropy_with_logits(product1, torch.ones_like(product1), reduction=reduction,
                                                      weight=pos_weights)
        loss_neg = F.binary_cross_entropy_with_logits(product2, torch.zeros_like(product2), reduction=reduction,
                                                      weight=neg_weights)
        return loss_pos + loss_neg

    # noinspection PyMethodMayBeStatic
    def join_forward(self,
                     z: torch.Tensor,
                     edge_index: torch.LongTensor,
                     edge_labels: torch.FloatTensor,
                     weights: torch.FloatTensor = None,
                     reduction='mean'):
        z_source = z[edge_index[0], :]
        z_target = z[edge_index[1], :]
        product = torch.einsum("ij, ij->i", [z_source, z_target])
        return F.binary_cross_entropy_with_logits(product, edge_labels, reduction=reduction, weight=weights)


class MyProb_Balanced_Normalized_Loss(torch.nn.Module):
    r"""An implementation of the probabilistic balanced normalized cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        node2comm_mat (torch.Tensor): matrix where <i,j> element is 1 if node i belongs to community j, 0 otherwise.
        device: torch device.
    """

    def __init__(self, node2comm_mat, device):
        super(MyProb_Balanced_Normalized_Loss, self).__init__()
        self.node2comm_mat = node2comm_mat.to(device)
        self.device = device

    def forward(self, A_p: torch.FloatTensor, A_n: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the probabilistic balanced normalized cut loss function.

        Arg types:
            * A_p (PyTorch FloatTensor) - binary adjacency matrix considering only positive edges
            * A_n (PyTorch FloatTensor) - binary adjacency matrix considering only negative edges

        Return types:
            * loss value (torch.Tensor).
        """

        # build matrix
        D_p = torch.diag(A_p.T.sum(dim=0), diagonal=0)
        D_n = torch.diag(A_n.T.sum(dim=0), diagonal=0)
        D_bar = torch.clone(D_p + D_n)
        L = D_p - (A_p - A_n)  # laplacian of a signed graph

        epsilon = torch.FloatTensor([1e-6]).to(self.device)
        L = L.to(self.device)
        D_bar = D_bar.to(self.device)
        result = torch.zeros(1).to(self.device)
        for k in range(self.node2comm_mat.shape[1]):
            prob_vector_mat = self.node2comm_mat[:, k, None]
            denominator = torch.matmul(torch.transpose(prob_vector_mat, 0, 1), torch.matmul(
                D_bar, prob_vector_mat))[0, 0] + epsilon  # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(
                prob_vector_mat, 0, 1), torch.matmul(L, prob_vector_mat)))[0, 0]

            result += numerator / denominator
        return result[0]


class Prob_Balanced_Normalized_Loss_Prob_Embeddings(torch.nn.Module):
    r"""An implementation of the probabilistic balanced normalized cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        A_p (scipy sparse matrices): Positive part of adjacency matrix A.
        A_n (scipy sparse matrices): Negative part of adjacency matrix A.
    """

    def __init__(self, A_p: torch.FloatTensor, A_n: torch.FloatTensor, device):
        super(Prob_Balanced_Normalized_Loss_Prob_Embeddings, self).__init__()
        D_p = torch.diag(A_p.T.sum(dim=0), diagonal=0)
        D_n = torch.diag(A_n.T.sum(dim=0), diagonal=0)
        D_p_o = torch.diag(A_p.T.sum(dim=1), diagonal=0)
        D_n_o = torch.diag(A_n.T.sum(dim=1), diagonal=0)
        self.D_bar = torch.clone(D_p + D_n + D_p_o + D_n_o).to(device)
        self.L = (D_p + D_p_o) - (A_p - A_n)  # laplacian of a signed graph
        self.L = self.L.to(device)
        self.device = device

    def forward(self, prob: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the probabilistic balanced normalized cut loss function.

        Arg types:
            * prob (PyTorch FloatTensor) - Prediction probability matrix made by the model

        Return types:
            * loss value (torch.Tensor).
        """
        epsilon = torch.FloatTensor([1e-6]).to(self.device)
        result = torch.zeros(1).to(self.device)
        for k in range(prob.shape[1]):
            prob_vector_mat = prob[:, k, None]
            denominator = torch.matmul(torch.transpose(prob_vector_mat, 0, 1), torch.matmul(
                self.D_bar, prob_vector_mat))[0, 0] + epsilon  # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(
                prob_vector_mat, 0, 1), torch.matmul(self.L, prob_vector_mat)))[0, 0]
            ratio = numerator / denominator
            if ratio.item() < 0:
                print('hey there is a problem')
                print('numerator > 0', numerator.item() > 0)
                print('denominator > 0', denominator.item() > 0)
            result += ratio
        return result[0]
