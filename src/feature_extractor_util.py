import torch
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from torch_sparse import coalesce


def my_create_spectral_features(
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor,
        node_num: int,
        dim: int
) -> torch.FloatTensor:
    if neg_edge_index.shape[0] > 0:
        edge_index = torch.cat(
            [pos_edge_index, neg_edge_index], dim=1)
    else:
        edge_index = pos_edge_index
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))

    pos_val = torch.full(
        (pos_edge_index.size(1),), 2, dtype=torch.float)
    if neg_edge_index.shape[0] > 0:
        neg_val = torch.full(
            (neg_edge_index.size(1),), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)
    else:
        val = pos_val

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    # Borrowed from:
    # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)


def create_unsigned_spectral_features(edge_index,
                                      latent_dim,
                                      num_nodes,
                                      device):
    return my_create_spectral_features(pos_edge_index=edge_index,
                                       neg_edge_index=torch.Tensor([]),
                                       node_num=num_nodes,
                                       dim=latent_dim
                                       ).to(device)


def create_signed_spectral_features(edge_index,
                                    edge_label,
                                    latent_dim,
                                    num_nodes,
                                    device):
    return my_create_spectral_features(pos_edge_index=edge_index[:, edge_label == 1.],
                                       neg_edge_index=edge_index[:, edge_label != 1.],
                                       node_num=num_nodes,
                                       dim=latent_dim
                                       ).to(device)
