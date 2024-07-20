import numpy as np
import torch
import pathlib
import pickle
import sys
sys.path.append('/mnt/nas/minici/S-GAE/src')

from data_loader import check_data_exists, save_data, load_data
from data_processor import generate_train_val_test_masked_sets, convert_neg_labels
from data_processor import parallel_get_data_for_triad_social_balance
from torch_geometric_signed_directed.data import load_signed_real_data

device_id = "3"
dataset_name = "bitcoin_alpha"
seed = 0
num_splits = 20
latent_dim = 64
test_perc, val_perc = 0.05, 0.05
mask_perc = 0.75
overwrite_data = False
is_transductive = False
random_masking = True
filter_transitive = False
noise_perc = 0.2

# Preliminary operations
device = torch.device("cuda" if torch.cuda.is_available() and device_id != "-1" else "cpu")
# Creating folder to host run-specific files
base_dir = pathlib.Path.cwd().parent
processed_data_dir = base_dir / 'data' / 'processed'
data_dir = processed_data_dir / dataset_name
if is_transductive:
    data_dir = data_dir / 'transductive'
if random_masking:
    data_dir = data_dir / 'random_masking'
if noise_perc > 0.0:
    data_dir = data_dir / f'noise_{noise_perc}'
if filter_transitive:
    data_dir = data_dir / 'filter_transitive'
data_dir = data_dir / f'seed_{seed}_num_splits_{num_splits}'
data_dir = data_dir / f'test_{round(test_perc, 2)}_val_{round(val_perc, 2)}'
data_dir = data_dir / f'mask_{round(mask_perc, 3)}'
print(data_dir)
print('Start creating basic dataset.')
# Check whether the basic dataset exists
if not check_data_exists(data_dir) or overwrite_data:
    print('producing dataset...')
    if dataset_name in ['birdwatch_USP']:
        with open(base_dir / 'data' / 'raw' / dataset_name / 'signed_data.pkl', 'rb') as file:
            data = pickle.load(file)
    else:
        # Load data using torch geometric signed directed data loader
        data = load_signed_real_data(dataset=dataset_name)
    # binarize edge weights
    data.edge_weight[data.edge_weight < 0] = -1.
    data.edge_weight[data.edge_weight > 0] = 1.
    # Pre-processing dataset
    signed_datasets = generate_train_val_test_masked_sets(data=data,
                                                          val_perc=val_perc,
                                                          test_perc=test_perc,
                                                          mask_perc=mask_perc,
                                                          noise_perc=noise_perc,
                                                          unlabeled_perc=None,
                                                          seed=seed,
                                                          num_splits=num_splits,
                                                          latent_dim=latent_dim,
                                                          is_transductive=is_transductive,
                                                          random_masking=random_masking,
                                                          filter_transitive=filter_transitive,
                                                          device=device)
    data_dir.mkdir(exist_ok=True, parents=True)
    # Save dataset to be reusable
    save_data(data_dir, signed_datasets)
else:
    signed_datasets = load_data(data_dir, device)

print('Start creating sparser datasets.')
# Now we can make the dataset "sparser" by excluding part of the unlabeled
# we will create 10 datasets with different amount of unlabeled
included_unlabeled_idxs = {run_id: [] for run_id in range(len(signed_datasets))}
unlabeled_idxs_pool = {run_id: np.array(range(signed_datasets[run_id]['train']['masked_edges'].shape[0])) for run_id in
                       range(len(signed_datasets))}
num_unlabeled_to_add_each_round = {run_id: signed_datasets[run_id]['train']['masked_edges'].shape[0] // 10 for run_id in
                                   range(len(signed_datasets))}

for unlabeled_perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    snapshot_signed_datasets = load_data(data_dir, device)
    for run_id in snapshot_signed_datasets:
        # gather how many edges I have to select this round
        num_unlabeled_to_add = num_unlabeled_to_add_each_round[run_id]
        # gather the pool from which to sample
        unlabeled_pool = unlabeled_idxs_pool[run_id]
        # select the unlabeled to add
        selected_unlabeled_idxs = np.random.permutation(range(unlabeled_pool.shape[0]))[:num_unlabeled_to_add]
        # update data structures
        if unlabeled_perc == 0.1:
            included_unlabeled_idxs[run_id] = unlabeled_pool[selected_unlabeled_idxs]
        else:
            included_unlabeled_idxs[run_id] = np.concatenate([included_unlabeled_idxs[run_id],
                                                              unlabeled_pool[selected_unlabeled_idxs]])
        unlabeled_idxs_pool[run_id] = np.delete(unlabeled_pool, selected_unlabeled_idxs)
        # Create a mask to include only selected unlabeled
        num_masked_edges = snapshot_signed_datasets[run_id]['train']['masked_edges'].shape[0]
        unlabeled_mask = torch.full((num_masked_edges,), fill_value=False)
        unlabeled_mask[torch.Tensor(included_unlabeled_idxs[run_id]).long()] = True
        snapshot_signed_datasets[run_id]['train']['masked_edges'] = snapshot_signed_datasets[run_id]['train']['masked_edges'][unlabeled_mask]
        snapshot_signed_datasets[run_id]['train']['masked_label'] = snapshot_signed_datasets[run_id]['train']['masked_label'][unlabeled_mask]
    # convert minus 1 to zero for negative edges
    if snapshot_signed_datasets[0]['train']['label'].min().item() == -1.:
        snapshot_signed_datasets = convert_neg_labels(snapshot_signed_datasets, -1, 0)
    # Extract incomplete triads
    snapshot_signed_datasets = parallel_get_data_for_triad_social_balance(signed_datasets=snapshot_signed_datasets,
                                                                          filter_transitive=filter_transitive,
                                                                          device=device)
    # Save dataset to be reusable
    (data_dir / f'unlabeled_{round(unlabeled_perc, 2)}').mkdir(parents=True, exist_ok=True)
    save_data(data_dir / f'unlabeled_{round(unlabeled_perc, 2)}', snapshot_signed_datasets)
