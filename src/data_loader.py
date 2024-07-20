from torch_geometric_signed_directed.data import load_signed_real_data
from data_processor import generate_train_val_test_masked_sets

import pickle

DATASET_FILENAME = 'signed_datasets.pkl'


def check_data_exists(data_dir):
    return (data_dir / DATASET_FILENAME).exists()


def save_data(data_dir, data):
    with open(data_dir / DATASET_FILENAME, 'wb') as file:
        pickle.dump(data, file)


def load_data(data_dir, device):
    with open(data_dir / DATASET_FILENAME, 'rb') as file:
        signed_datasets = pickle.load(file)
    # move every object to the given device
    for split_num in signed_datasets:
        for key in ['graph', 'weights', 'masked_edges', 'masked_label']:
            if key in signed_datasets[split_num]:
                signed_datasets[split_num][key] = signed_datasets[split_num][key].to(device)
        for key in ['train', 'val', 'test']:
            for inner_key in signed_datasets[split_num][key]:
                signed_datasets[split_num][key][inner_key] = signed_datasets[split_num][key][inner_key].to(device)
        for key in ['unknowns', 'knowns']:
            if key in signed_datasets[split_num]:
                signed_datasets[split_num][key] = [elem.to(device) for elem in signed_datasets[split_num][key]]
    return signed_datasets


def create_data_loader(dataset_name, base_dir, data_dir, hyper_params, device):
    if hyper_params["unlabeled_perc"] is not None:
        assert check_data_exists(data_dir), f'{data_dir} does not exist.'
    if not check_data_exists(data_dir) or hyper_params["overwrite_data"]:
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
        datasets = generate_train_val_test_masked_sets(data=data,
                                                       val_perc=hyper_params["val_perc"],
                                                       test_perc=hyper_params["test_perc"],
                                                       mask_perc=hyper_params["mask_perc"],
                                                       noise_perc=hyper_params["noise_perc"],
                                                       unlabeled_perc=hyper_params["unlabeled_perc"],
                                                       seed=hyper_params["seed"],
                                                       num_splits=hyper_params["num_splits"],
                                                       latent_dim=hyper_params["latent_dim"],
                                                       is_transductive=hyper_params["is_transductive"],
                                                       random_masking=hyper_params["random_masking"],
                                                       filter_transitive=hyper_params['filter_transitive'],
                                                       device=device)
        # Save dataset to be reusable
        save_data(data_dir, datasets)
    else:
        datasets = load_data(data_dir, device)
    return datasets
