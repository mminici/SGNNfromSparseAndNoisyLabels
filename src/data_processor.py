import concurrent.futures
import numpy as np
import torch
import networkx as nx

from torch_geometric_signed_directed.data.signed.SignedData import SignedData
from social_balance_data_utils import get_data_for_triad_social_balance
from feature_extractor_util import create_signed_spectral_features
from community import community_louvain


def convert_labels_to_weights(labels):
    edge_weight = torch.clone(labels)  # assuming label are generated according to sign task
    # convert zero to minus one, then convert to float tensor
    edge_weight[edge_weight == 0] = -1.
    return edge_weight.float()


def get_edge_index_with_sign(edges, labels):
    edge_index = torch.clone(edges)
    edge_labels = torch.clone(labels)
    return torch.cat([edge_index, convert_labels_to_weights(edge_labels).unsqueeze(dim=1)], dim=1)


def parallel_get_data_for_triad_social_balance(signed_datasets, filter_transitive, device):
    get_data_for_triad_social_balance_fn = lambda idx: get_data_for_triad_social_balance(signed_datasets[idx]['train'],
                                                                                         filter_transitive,
                                                                                         device)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function to the data list using multiple threads
        results = list(executor.map(get_data_for_triad_social_balance_fn, range(len(signed_datasets))))
        for i in range(len(results)):
            signed_datasets[i]['unknowns'] = results[i][0]
            signed_datasets[i]['knowns'] = results[i][1]
    return signed_datasets


def mask_data(signed_datasets, mask_perc, seed, is_transductive=False, random_masking=True):
    # Masking the signs of edges in the training set
    for run_id in signed_datasets:
        edge_index = signed_datasets[run_id]['train']['edges'].T
        # Create surrogate torch geometric signed data object to re-use link_split function
        tr_data = SignedData(edge_index=edge_index,
                             edge_weight=signed_datasets[run_id]['train']['label'])
        tr_and_masked = tr_data.link_split(prob_val=0.0,
                                           prob_test=mask_perc,
                                           task='sign',
                                           maintain_connect=not random_masking,
                                           seed=seed,
                                           splits=1)
        signed_datasets[run_id]['train']['edges'] = torch.clone(tr_and_masked[0]['train']['edges'])
        signed_datasets[run_id]['train']['label'] = torch.clone(tr_and_masked[0]['train']['label'])
        signed_datasets[run_id]['train']['masked_edges'] = torch.clone(tr_and_masked[0]['test']['edges'])
        signed_datasets[run_id]['train']['masked_label'] = torch.clone(tr_and_masked[0]['test']['label'])
        if is_transductive:
            signed_datasets[run_id]['train']['masked_edges'] = torch.concat(
                [signed_datasets[run_id]['train']['masked_edges'],
                 torch.clone(tr_and_masked[0]['val']['edges']),
                 torch.clone(tr_and_masked[0]['test']['edges'])],
                dim=0)
            signed_datasets[run_id]['train']['masked_label'] = torch.concat(
                [signed_datasets[run_id]['train']['masked_label'],
                 torch.clone(tr_and_masked[0]['val']['label']),
                 torch.clone(tr_and_masked[0]['test']['label'])])
        del tr_data
        del tr_and_masked
    return signed_datasets


def convert_neg_labels(signed_datasets, neg_label_val, neg_new_label_val):
    """
        Convert negative labels in the specified datasets to zero.

        Parameters:
        - signed_datasets (dict): A dictionary containing datasets for different run_ids.
                                 The structure should be like:
                                 {
                                     run_id_1: {
                                         'train': {'edges': torch.Tensor, 'label': torch.Tensor},
                                         'val': {'edges': torch.Tensor, 'label': torch.Tensor},
                                         'test': {'edges': torch.Tensor, 'label': torch.Tensor}
                                     },
                                     run_id_2: {
                                         ...
                                     },
                                     ...
                                 }
        - neg_label_val (int): The value representing negative labels to be converted.
        - neg_new_label_val (int): The value representing the new negative labels.

        Returns:
        - dict: The modified signed_datasets dictionary with negative labels converted to zero.
    """
    for run_id in signed_datasets:
        for split in ['train', 'val', 'test']:
            signed_datasets[run_id][split]['label'][
                signed_datasets[run_id][split]['label'] == neg_label_val] = neg_new_label_val
        if 'masked_label' in signed_datasets[run_id]['train']:
            signed_datasets[run_id]['train']['masked_label'][
                signed_datasets[run_id]['train']['masked_label'] == neg_label_val] = neg_new_label_val
    return signed_datasets


def check_data_processing(generated_datasets):
    # Assert various conditions
    for split_num in generated_datasets:
        # Training contains more nodes than validation and test sets
        tr_nodes = torch.unique(torch.flatten(generated_datasets[split_num]['train']['edges'])).detach().cpu().numpy()
        val_nodes = torch.unique(torch.flatten(generated_datasets[split_num]['val']['edges'])).detach().cpu().numpy()
        test_nodes = torch.unique(torch.flatten(generated_datasets[split_num]['test']['edges'])).detach().cpu().numpy()
        diff_val_to_train = len(np.setdiff1d(val_nodes, tr_nodes))
        assert diff_val_to_train == 0, f'val:{len(val_nodes)}-train:{len(tr_nodes)}- diff:{diff_val_to_train}'
        diff_test_to_train = len(np.setdiff1d(test_nodes, tr_nodes))
        assert diff_test_to_train == 0, f'test:{len(test_nodes)}-train:{len(tr_nodes)}- diff:{diff_test_to_train}'


def inject_noise(signed_datasets, noise_perc):
    for run_id in signed_datasets:
        for idx in range(signed_datasets[run_id]['train']['edges'].shape[0]):
            if np.random.random() <= noise_perc:
                if signed_datasets[run_id]['train']['label'][idx] == 1:
                    signed_datasets[run_id]['train']['label'][idx] = 0
                else:
                    signed_datasets[run_id]['train']['label'][idx] = 1
    return signed_datasets


def generate_train_val_test_masked_sets(data,
                                        val_perc,
                                        test_perc,
                                        mask_perc,
                                        seed,
                                        num_splits,
                                        latent_dim,
                                        device,
                                        is_transductive=False,
                                        random_masking=True,
                                        unlabeled_perc=None,
                                        filter_transitive=False,
                                        noise_perc=0.0,
                                        feature_extractor_fn=create_signed_spectral_features
                                        ):
    """
        Generate train, validation, and test datasets with masked edges for the training.
        It also computes the node features.

        Parameters:
        - data (torch_geometric_signed_directed.data.signed.SignedData.SignedData): The input data for generating datasets.
        - val_perc (float): The percentage of data to allocate for validation.
        - test_perc (float): The percentage of data to allocate for testing.
        - mask_perc (float): The percentage of training edges to mask.
        - seed (int): Seed for reproducibility.
        - num_splits (int): Number of train, validation, test splits to create.
        - latent_dim (int): Dimensionality of latent features.
        - device (torch.device): The device to place the generated datasets and features.
        - feature_extractor_fn (Callable): The function to extract features from edges.
                                          Default is create_unsigned_spectral_features.

        Returns:
        - dict: A dictionary containing train, validation, and test datasets with masked edges and computed features.
    """
    # Create several train, val, test splits
    signed_datasets = data.link_split(prob_val=val_perc,
                                      prob_test=test_perc,
                                      task='sign',
                                      maintain_connect=True,
                                      seed=seed,
                                      splits=num_splits)
    # Compute features for each split
    for split_num in signed_datasets:
        signed_datasets[split_num]['features'] = feature_extractor_fn(
            edge_index=signed_datasets[split_num]['train']['edges'].T,
            edge_label=signed_datasets[split_num]['train']['label'],
            latent_dim=latent_dim,
            num_nodes=signed_datasets[split_num]['train']['edges'].max().item() + 1,
            device=device)
    # convert zero to minus 1 for negative edges
    signed_datasets = convert_neg_labels(signed_datasets, 0, -1)
    # Masking part of the training edges
    signed_datasets = mask_data(signed_datasets,
                                mask_perc=mask_perc,
                                seed=seed,
                                is_transductive=is_transductive,
                                random_masking=random_masking)
    # moving back part of the masked edges into the supervised set in case unlabeled_perc is not None
    if unlabeled_perc is not None:
        signed_datasets = _move_back_masked_to_supervised(signed_datasets, unlabeled_perc)
    # convert minus 1 to zero for negative edges
    signed_datasets = convert_neg_labels(signed_datasets, -1, 0)
    if noise_perc > 0:
        signed_datasets = inject_noise(signed_datasets, noise_perc)
    # Extract incomplete triads
    signed_datasets = parallel_get_data_for_triad_social_balance(signed_datasets=signed_datasets,
                                                                 filter_transitive=filter_transitive,
                                                                 device=device)
    # check_data_processing(signed_datasets)
    return signed_datasets


def _move_back_masked_to_supervised(signed_datasets, unlabeled_perc):
    for run_id in signed_datasets:
        num_masked_edges = signed_datasets[run_id]['train']['masked_edges'].shape[0]
        unlabeled_mask = torch.full((num_masked_edges,), fill_value=False)
        unlabeled_mask[torch.randperm(num_masked_edges)[:int(num_masked_edges * unlabeled_perc)]] = True
        signed_datasets[run_id]['train']['masked_edges'] = signed_datasets[run_id]['train']['masked_edges'][
            unlabeled_mask]
        signed_datasets[run_id]['train']['masked_label'] = signed_datasets[run_id]['train']['masked_label'][
            unlabeled_mask]
    return signed_datasets


def setup_data_for_lrw(signed_data, device, proxy_perc=0.5):
    """
        Process data for the LRW procedure, partitioning training edges into [training, proxy] sets.

        Parameters:
        - signed_data (dict): The input data dictionary containing edges and labels.
        - device (torch.device): The device to place the processed data.
        - proxy_perc (float, optional): The percentage of edges to allocate for the proxy set.
                                        Default is 0.5.

        Returns:
        - tuple: A tuple containing the processed signed_data, proxy dataset, and proxy labels.
    """
    num_train_edges = signed_data['edges'].shape[0]
    train_edges_idx = np.random.permutation(range(num_train_edges))[:int(num_train_edges * (1 - proxy_perc))]
    train_edges_mask = torch.full((num_train_edges,), fill_value=False)
    train_edges_mask[train_edges_idx] = True
    signed_data['proxy_edges'] = signed_data['edges'][~train_edges_mask]
    signed_data['proxy_label'] = signed_data['label'][~train_edges_mask]
    # Concat positive and negative edges of the proxy edges
    positive_proxy_idxs = signed_data['proxy_label'] == 1.
    negative_proxy_idxs = signed_data['proxy_label'] == 0.
    # g is the clean dataset as in the original paper of LRW
    g_dataset = torch.cat([signed_data['proxy_edges'][positive_proxy_idxs],
                           signed_data['proxy_edges'][negative_proxy_idxs]],
                          dim=0).to(device).T
    g_labels = torch.ones(positive_proxy_idxs.sum())
    g_labels = torch.cat([g_labels, torch.zeros(negative_proxy_idxs.sum())]).to(device)
    # Update data used to directly train the model
    signed_data['edges'] = signed_data['edges'][train_edges_mask].long()
    signed_data['label'] = signed_data['label'][train_edges_mask].float()
    return signed_data, g_dataset.long(), g_labels.float()


def get_mesoscale_labels(data, device):
    # compute communities
    graph_edges = torch.cat([data['train']['edges'],
                             data['val']['edges'],
                             data['test']['edges'],
                             data['train']['masked_edges']]).detach().cpu().numpy()
    g = nx.from_edgelist(graph_edges, create_using=nx.Graph())
    partitions = community_louvain.best_partition(g)
    num_communities = max(partitions.values()) + 1
    node2comm = torch.zeros((max(g.nodes()) + 1, num_communities)).to(device).float()
    for node_id in partitions:
        node2comm[node_id, partitions[node_id]] = 1.
    # Label masked edges according to mesoscale social balance
    masked_edges_label_mesoscale_sb = torch.zeros(data['train']['masked_edges'].shape[0]).to(
        device)
    for i in range(data['train']['masked_edges'].shape[0]):
        node_u_membership_vec = node2comm[data['train']['masked_edges'][i][0]]
        node_v_membership_vec = node2comm[data['train']['masked_edges'][i][1]]
        # intra-comm edge
        if (node_u_membership_vec * node_v_membership_vec).sum() == 1:
            masked_edges_label_mesoscale_sb[i] = 1.
    return masked_edges_label_mesoscale_sb


def create_noisy_labels_meso(data, device, mesoscale_labels):
    # Augment the noisy dataset with mesoscale knowledge
    f_dataset = torch.clone(data['train']['masked_edges']).T.to(device)
    f_labels = torch.clone(mesoscale_labels).to(device)
    return f_dataset, f_labels


def create_noisy_labels_micro_and_meso(data, device, mesoscale_labels):
    # Concat positive and negative edges of the incomplete triads
    # f is the noisy dataset (it contains our incomplete triads) as in the original paper
    f_dataset = torch.cat([data['unknowns'][0],
                           data['unknowns'][1]],
                          dim=1).to(device)
    f_labels = torch.ones(data['unknowns'][0].shape[1])
    f_labels = torch.cat([f_labels, torch.zeros(data['unknowns'][1].shape[1])]).to(device)
    # Augment the noisy dataset with mesoscale knowledge
    f_dataset = torch.cat([f_dataset,
                           torch.clone(data['train']['masked_edges']).T], dim=1).to(device)
    f_labels = torch.cat([f_labels,
                          torch.clone(mesoscale_labels)]).to(device)
    return f_dataset, f_labels


def create_noisy_labels_micro(data, device):
    # Concat positive and negative edges of the incomplete triads
    # f is the noisy dataset (it contains our incomplete triads) as in the original paper
    f_dataset = torch.cat([data['unknowns'][0],
                           data['unknowns'][1]],
                          dim=1).to(device)
    f_labels = torch.ones(data['unknowns'][0].shape[1])
    f_labels = torch.cat([f_labels, torch.zeros(data['unknowns'][1].shape[1])]).to(device)
    return f_dataset, f_labels



