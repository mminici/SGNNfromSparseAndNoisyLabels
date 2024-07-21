import uuid
import os
import pathlib
import mlflow
import torch
import shutil
import numpy as np
import networkx as nx

from data_loader import check_data_exists, save_data, load_data
from data_processor import get_edge_index_with_sign, generate_train_val_test_masked_sets
from model_eval import eval_model
from my_utils import set_seed
from models import MySDGNN
from plot_utils import plot_losses

from torch_geometric_signed_directed.data import load_signed_real_data
from community import community_louvain

# noinspection PyShadowingNames
def run_experiment(dataset_name='wiki',
                   overwrite_data=False,
                   learnable_features=False,
                   val_perc=0.1,
                   test_perc=0.1,
                   mask_perc=0.05,
                   seed=0,
                   num_splits=10,
                   device_id='',
                   learning_rate=0.001,
                   weight_decay=0.001,
                   num_epochs=1000,
                   check_loss_freq=25,
                   latent_dim=4,
                   num_layers=2,
                   metric_to_optimize='auc_score',
                   alpha=1.0,
                   beta=1.0,
                   early_stopping_limit=10):
    # Start experiment
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('overwrite_data', overwrite_data)
    mlflow.log_param('learnable_features', learnable_features)
    mlflow.log_param('val_perc', val_perc)
    mlflow.log_param('test_perc', test_perc)
    mlflow.log_param('mask_perc', mask_perc)
    mlflow.log_param('seed', seed)
    mlflow.log_param('num_splits', num_splits)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('weight_decay', weight_decay)
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('latent_dim', latent_dim)
    mlflow.log_param('num_layers', num_layers)
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('beta', beta)
    mlflow.log_param('metric_to_optimize', metric_to_optimize)
    mlflow.log_param('early_stopping_limit', early_stopping_limit)

    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device = torch.device("cuda" if torch.cuda.is_available() and device_id != "-1" else "cpu")
    # Creating folder to host run-specific files
    base_dir = pathlib.Path.cwd().parent
    my_run_id = uuid.uuid4()
    interim_data_dir = base_dir / 'data' / 'interim' / f"{my_run_id}"
    interim_data_dir.mkdir(exist_ok=True, parents=True)

    # Import dataset
    processed_data_dir = base_dir / 'data' / 'processed'
    data_dir = processed_data_dir / dataset_name / f'seed_{seed}_num_splits_{num_splits}'
    data_dir = data_dir / f'test_{round(test_perc, 2)}_val_{round(val_perc, 2)}'
    data_dir = data_dir / f'mask_{round(mask_perc, 3)}'
    data_dir.mkdir(exist_ok=True, parents=True)

    if not check_data_exists(data_dir) or overwrite_data:
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
                                                              seed=seed,
                                                              num_splits=num_splits,
                                                              latent_dim=latent_dim,
                                                              device=device)
        # Save dataset to be reusable
        save_data(data_dir, signed_datasets)
    else:
        signed_datasets = load_data(data_dir, device)

    test_metrics_dict = {}
    for metric in ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score']:
        test_metrics_dict[metric] = [None] * num_splits

    train_loss_dict = {}
    val_metrics_dict = {}
    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')
        train_loss_dict[run_id] = {'supervised': [], 'mesoscale_sb': []}
        val_metrics_dict[run_id] = []
        # compute communities
        graph_edges = torch.cat([signed_datasets[run_id]['train']['edges'],
                                 signed_datasets[run_id]['val']['edges'],
                                 signed_datasets[run_id]['test']['edges'],
                                 signed_datasets[run_id]['train']['masked_edges']]).detach().cpu().numpy()
        g = nx.from_edgelist(graph_edges, create_using=nx.Graph())
        partitions = community_louvain.best_partition(g)
        num_communities = max(partitions.values()) + 1
        node2comm = torch.zeros((max(g.nodes()) + 1, num_communities)).to(device).float()
        for node_id in partitions:
            node2comm[node_id, partitions[node_id]] = 1.
        # Label masked edges according to mesoscale social balance
        masked_edges_label_mesoscale_sb = torch.zeros(signed_datasets[run_id]['train']['masked_edges'].shape[0]).to(
            device)
        for i in range(signed_datasets[run_id]['train']['masked_edges'].shape[0]):
            node_u_membership_vec = node2comm[signed_datasets[run_id]['train']['masked_edges'][i][0]]
            node_v_membership_vec = node2comm[signed_datasets[run_id]['train']['masked_edges'][i][1]]
            # intra-comm edge
            if (node_u_membership_vec * node_v_membership_vec).sum() == 1:
                masked_edges_label_mesoscale_sb[i] = 1.

        # reformat edge index to have for each edge an additional entry equal to the sign
        edge_index_with_sign = get_edge_index_with_sign(signed_datasets[run_id]['train']['edges'],
                                                        signed_datasets[run_id]['train']['label']).long().to(device)
        # Create the model
        model = MySDGNN(node_num=signed_datasets[run_id]['features'].shape[0],
                        learnable_features=learnable_features,
                        edge_index_s=edge_index_with_sign,
                        in_dim=signed_datasets[run_id]['features'].shape[1],
                        out_dim=latent_dim,
                        layer_num=num_layers,
                        lamb_d=1.0,
                        lamb_t=0.0,
                        features=signed_datasets[run_id]['features'].to(device))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'

        # move edges and labels to cpu to be used by the Logistic Regression binary classifier
        edges_dict, label_dict = {}, {}
        for data_split in ['train', 'val', 'test']:
            edges_dict[data_split] = signed_datasets[run_id][data_split]['edges'].detach().cpu()
            label_dict[data_split] = signed_datasets[run_id][data_split]['label'].detach().cpu()

        early_stopping_cnt = 0
        for epoch in range(num_epochs):
            if early_stopping_cnt > early_stopping_limit:
                break
            model.train()
            optimizer.zero_grad()
            zeta, loss = model.loss()
            # meso-scale loss
            trust_all_loss = model.loss_sign.join_forward(zeta,
                                                          signed_datasets[run_id]['train']['masked_edges'].T,
                                                          masked_edges_label_mesoscale_sb,
                                                          reduction='none')
            mesoscale_sb_loss = trust_all_loss.mean()
            global_loss = loss + mesoscale_sb_loss
            # Perform backpropagation
            global_loss.backward()
            optimizer.step()
            train_loss_dict[run_id]['supervised'].append(loss.item())
            train_loss_dict[run_id]['mesoscale_sb'].append(mesoscale_sb_loss.item())
            if epoch % check_loss_freq == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_metrics = eval_model(model=model,
                                             train_edge_index=edges_dict['train'],
                                             test_edge_index=edges_dict['val'],
                                             train_ground_truth=label_dict['train'],
                                             test_ground_truth=label_dict['val'])
                    val_metrics_dict[run_id].append(val_metrics[metric_to_optimize])
                    if val_metrics[metric_to_optimize] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[metric_to_optimize]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                    print(f'Mesoscale sb loss: {mesoscale_sb_loss.item()}')
                    print(
                        f'Epoch {epoch}/{num_epochs} train_loss: {global_loss.item()} -- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]}')
            else:
                val_metrics_dict[run_id].append(0.)

        # Test performance
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_metrics = eval_model(model=model,
                                      train_edge_index=edges_dict['train'],
                                      test_edge_index=edges_dict['test'],
                                      train_ground_truth=label_dict['train'],
                                      test_ground_truth=label_dict['test'])
        for metric_name in test_metrics:
            test_metrics_dict[metric_name][run_id] = test_metrics[metric_name]

    for split_num in range(num_splits):
        mlflow.log_artifact(interim_data_dir / f'model{split_num}.pth')  # store best model
        fig = plot_losses(
            train_values=[train_loss_dict[split_num]['supervised'], train_loss_dict[split_num]['mesoscale_sb']],
            val_values=[val_metrics_dict[split_num]],
            train_labels=['supervised loss', 'mesoscale sb loss'],
            val_labels=[f'val {metric_to_optimize}'])
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.png', dpi=800)
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.png')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')

    # Simulation ended, report metrics on test set for the best model
    for metric_name in test_metrics_dict:
        avg_val, std_val = np.mean(test_metrics_dict[metric_name]), np.std(test_metrics_dict[metric_name])
        avg_val, std_val = round(avg_val, 4), round(std_val, 4)
        print(f'Test {metric_name}: {avg_val}+-{std_val}')
        mlflow.log_metric(metric_name + '_avg', avg_val)
        mlflow.log_metric(metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'test_{metric_name}', arr=np.array(test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'test_{metric_name}.npy')
    return interim_data_dir


if __name__ == '__main__':
    # Run input parameters
    dataset_name = 'bitcoin_alpha'
    val_perc = 0.05
    test_perc = 0.05
    overwrite_data = False
    learnable_features = False
    mask_perc = [0.75, ]
    seed = [0, ]
    num_splits = [20, ]
    device_id = '0'
    # optimization hyperparameters
    learning_rate = 0.005
    weight_decay = 0.001
    num_epochs = 1000
    check_loss_freq = 25
    early_stopping_limit = 10
    # model hyperparameters
    latent_dim = 64
    num_layers = 2
    metric_to_optimize = 'f1_macro'
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-SDGNN-{seed_val}')
        for mask_perc_val in mask_perc:
            for num_splits_val in num_splits:
                with mlflow.start_run():
                    exp_dir = run_experiment(dataset_name=dataset_name,
                                             overwrite_data=overwrite_data,
                                             learnable_features=learnable_features,
                                             val_perc=val_perc,
                                             test_perc=test_perc,
                                             mask_perc=mask_perc_val,
                                             seed=seed_val,
                                             num_splits=num_splits_val,
                                             device_id=device_id,
                                             learning_rate=learning_rate,
                                             weight_decay=weight_decay,
                                             num_epochs=num_epochs,
                                             check_loss_freq=check_loss_freq,
                                             latent_dim=latent_dim,
                                             num_layers=num_layers,
                                             metric_to_optimize=metric_to_optimize,
                                             early_stopping_limit=early_stopping_limit
                                             )
                    try:
                        shutil.rmtree(exp_dir, ignore_errors=True)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))
