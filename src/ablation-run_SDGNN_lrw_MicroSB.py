import os
import mlflow
import torch
import shutil
import numpy as np

from data_loader import create_data_loader
from data_processor import get_edge_index_with_sign, create_noisy_labels_micro
from model_eval import eval_model, TrainLogMetrics, TestLogMetrics
from my_utils import set_seed, setup_env, move_data_to_device
from models import MySDGNN
from plot_utils import plot_losses


# noinspection PyShadowingNames
def run_experiment(dataset_name='wiki',
                   overwrite_data=False,
                   learnable_features=False,
                   init_eps_one=False,
                   is_transductive=False,
                   random_masking=True,
                   alpha=0.0,
                   beta=1.0,
                   val_perc=0.1,
                   test_perc=0.1,
                   mask_perc=0.05,
                   noise_perc=0.0,
                   unlabeled_perc=None,
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
                   early_stopping_limit=10):
    # Start experiment
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('overwrite_data', overwrite_data)
    mlflow.log_param('learnable_features', learnable_features)
    mlflow.log_param('init_eps_one', init_eps_one)
    mlflow.log_param('random_masking', random_masking)
    mlflow.log_param('val_perc', val_perc)
    mlflow.log_param('test_perc', test_perc)
    mlflow.log_param('noise_perc', noise_perc)
    mlflow.log_param('mask_perc', mask_perc)
    mlflow.log_param('unlabeled_perc', unlabeled_perc)
    mlflow.log_param('seed', seed)
    mlflow.log_param('num_splits', num_splits)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('weight_decay', weight_decay)
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('latent_dim', latent_dim)
    mlflow.log_param('num_layers', num_layers)
    mlflow.log_param('metric_to_optimize', metric_to_optimize)
    mlflow.log_param('early_stopping_limit', early_stopping_limit)

    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, seed, num_splits, val_perc,
                                                             test_perc, mask_perc, noise_perc,
                                                             is_transductive=is_transductive,
                                                             filter_transitive=False,
                                                             random_masking=random_masking)
    print(data_dir)
    # Create data loader for signed datasets
    signed_datasets = create_data_loader(dataset_name, base_dir, data_dir,
                                         hyper_params={'overwrite_data': overwrite_data,
                                                       'is_transductive': is_transductive,
                                                       'random_masking': random_masking,
                                                       'val_perc': val_perc,
                                                       'test_perc': test_perc,
                                                       'mask_perc': mask_perc,
                                                       'noise_perc': noise_perc,
                                                       'unlabeled_perc': unlabeled_perc,
                                                       'seed': seed,
                                                       'num_splits': num_splits,
                                                       'latent_dim': latent_dim,
                                                       'filter_transitive': False,
                                                       },
                                         device=device)
    # Transfer data to device
    signed_datasets = move_data_to_device(signed_datasets, device)
    # Create loggers
    train_logger = TrainLogMetrics(num_splits, ['supervised', 'social_balance'])
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])

    init_eps_fn = lambda x: torch.ones(x) if init_eps_one else torch.rand(x)
    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')
        # Move validation and test labels to numpy for evaluation
        for split_type in ['val', 'test']:
            signed_datasets[run_id][split_type]['label'] = signed_datasets[run_id][split_type][
                'label'].detach().cpu().numpy()

        # move data to device
        all_train_edges = torch.clone(signed_datasets[run_id]['train']['edges']).to(device)
        all_train_label = torch.clone(signed_datasets[run_id]['train']['label']).to(device)
        # reformat edge index to have for each edge an additional entry equal to the sign
        edge_index_with_sign = get_edge_index_with_sign(all_train_edges, all_train_label).long().to(device)
        # Create the model
        model = MySDGNN(node_num=signed_datasets[run_id]['features'].shape[0],
                        edge_index_s=edge_index_with_sign,
                        learnable_features=learnable_features,
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

        # create noisy dataset labeled with social balance
        f_dataset, f_labels = create_noisy_labels_micro(signed_datasets[run_id], device)
        # split training edges for Learning to Reweight procedure
        train_pos_idxs = torch.where(edge_index_with_sign[:, 2] == 1.)[0]
        train_neg_idxs = torch.where(edge_index_with_sign[:, 2] != 1.)[0]
        num_pos_edges, num_neg_edges = train_pos_idxs.shape[0], train_neg_idxs.shape[0]
        lrw_pos_mask = torch.full(fill_value=False, size=train_pos_idxs.shape)
        lrw_neg_mask = torch.full(fill_value=False, size=train_neg_idxs.shape)
        lrw_pos_mask[torch.randperm(train_pos_idxs.shape[0])[:num_pos_edges // 2]] = True
        lrw_neg_mask[torch.randperm(train_neg_idxs.shape[0])[:num_neg_edges // 2]] = True
        tr_pos_edge_index = edge_index_with_sign[train_pos_idxs, :2].T
        tr_neg_edge_index = edge_index_with_sign[train_neg_idxs, :2].T

        early_stopping_cnt = 0
        for epoch in range(num_epochs):
            if early_stopping_cnt > early_stopping_limit:
                break
            model.train()
            # Create a metamodel
            meta_model = MySDGNN(node_num=signed_datasets[run_id]['features'].shape[0],
                                 learnable_features=learnable_features,
                                 edge_index_s=edge_index_with_sign,
                                 in_dim=signed_datasets[run_id]['features'].shape[1],
                                 out_dim=latent_dim,
                                 layer_num=num_layers,
                                 lamb_d=1.0,
                                 lamb_t=0.0,
                                 features=signed_datasets[run_id]['features'].to(device),
                                 reduction='none')
            meta_model.train()
            meta_model.to(device)
            meta_model.load_state_dict(model.state_dict())  # copy the current model weights
            meta_optimizer = torch.optim.SGD(meta_model.parameters(),
                                             lr=1e-3)
            meta_model.zero_grad()
            # Compute the loss of the metamodel
            meta_model_zeta = meta_model.forward()
            meta_model_loss = meta_model.link_sign_loss_joint(meta_model_zeta,
                                                              f_dataset,
                                                              f_labels)

            eps = init_eps_fn(f_dataset.shape[1]).to(device)
            eps = torch.nn.Parameter(eps, requires_grad=True)
            eps_optimizer = torch.optim.SGD([eps], lr=1.0)
            meta_model_loss = torch.sum(meta_model_loss * eps)
            # Update the meta-model
            meta_model.zero_grad()
            meta_model_loss.backward()
            meta_optimizer.step()
            # compute the optimal epsilon
            zeta = model.forward()
            loss_sign = model.link_sign_loss(zeta, tr_pos_edge_index[:, lrw_pos_mask],
                                             tr_neg_edge_index[:, lrw_neg_mask])
            loss_direction = model.loss_direction(zeta, tr_pos_edge_index[:, lrw_pos_mask],
                                                  tr_neg_edge_index[:, lrw_neg_mask])
            l_g_meta = loss_sign + loss_direction
            l_g_meta.backward()
            eps_optimizer.step()
            # Normalize the gradients
            eps = torch.clamp(eps, min=0)
            norm_c = torch.sum(eps)
            if norm_c != 0:
                w = eps / norm_c
            else:
                w = eps
            # Compute the loss on the noisy set with the computed weights
            optimizer.zero_grad()
            zeta = model.forward()
            loss_sign = model.link_sign_loss(zeta, tr_pos_edge_index,
                                                   tr_neg_edge_index)
            loss_direction = model.loss_direction(zeta, tr_pos_edge_index,
                                                  tr_neg_edge_index)
            supervised_loss = loss_sign + loss_direction
            noisy_loss = model.link_sign_loss_joint(zeta,
                                                    f_dataset,
                                                    f_labels)  # compute the loss on the "noisy" dataset
            noisy_loss = torch.sum(noisy_loss * w)  # re-weight the loss with the learned weights
            loss = beta * supervised_loss + alpha * noisy_loss

            # Perform backpropagation
            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', beta * supervised_loss.item())
            train_logger.train_update(run_id, 'social_balance', alpha * noisy_loss.item())
            if epoch % check_loss_freq == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_metrics = eval_model(model,
                                             signed_datasets[run_id]['train']['edges'],
                                             signed_datasets[run_id]['val']['edges'],
                                             signed_datasets[run_id]['train']['label'],
                                             signed_datasets[run_id]['val']['label'])
                    train_logger.val_update(run_id, val_metrics[metric_to_optimize])
                    if val_metrics[metric_to_optimize] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[metric_to_optimize]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                    print('noisy loss: ', noisy_loss.item())
                    print(
                        f'Epoch {epoch}/{num_epochs} train_loss: {loss.item()} -- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]}')
            else:
                train_logger.val_update(run_id, 0.0)

        # Test performance
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_metrics = eval_model(model,
                                      signed_datasets[run_id]['train']['edges'],
                                      signed_datasets[run_id]['test']['edges'],
                                      signed_datasets[run_id]['train']['label'],
                                      signed_datasets[run_id]['test']['label'])
            val_metrics = eval_model(model,
                                     signed_datasets[run_id]['train']['edges'],
                                     signed_datasets[run_id]['val']['edges'],
                                     signed_datasets[run_id]['train']['label'],
                                     signed_datasets[run_id]['val']['label'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

    for split_num in range(num_splits):
        mlflow.log_artifact(interim_data_dir / f'model{split_num}.pth')  # store best model
        fig = plot_losses(
            train_values=[train_logger.train_loss_dict[split_num]['supervised'],
                          train_logger.train_loss_dict[split_num]['social_balance']],
            val_values=[train_logger.val_metrics_dict[split_num]],
            train_labels=['supervised loss', 'social balance'],
            val_labels=[f'val {metric_to_optimize}'])
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.png', dpi=800)
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.png')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')

    # Simulation ended, report metrics on test set for the best model
    for metric_name in test_logger.test_metrics_dict:
        avg_val, std_val = test_logger.get_metric_stats(metric_name)
        print(f'Test {metric_name}: {avg_val}+-{std_val}')
        mlflow.log_metric(metric_name + '_avg', avg_val)
        mlflow.log_metric(metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'test_{metric_name}', arr=np.array(test_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'test_{metric_name}.npy')
    # Simulation ended, report metrics on val set for the best model
    for metric_name in val_logger.test_metrics_dict:
        avg_val, std_val = val_logger.get_metric_stats(metric_name)
        print(f'Val {metric_name}: {avg_val}+-{std_val}')
        mlflow.log_metric('val_' + metric_name + '_avg', avg_val)
        mlflow.log_metric('val_' + metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'val_{metric_name}', arr=np.array(val_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'val_{metric_name}.npy')
    return interim_data_dir


if __name__ == '__main__':
    # Run input parameters
    dataset_name = 'bitcoin_alpha'
    val_perc = 0.05
    test_perc = 0.05
    overwrite_data = False
    learnable_features = False
    init_eps_one = True
    is_transductive = False
    random_masking = True
    mask_perc = [0.75, ]
    noise_perc = [0.0, ]
    unlabeled_perc = [None, ]
    seed = [0, ]
    num_splits = [20, ]
    device_id = '0'
    # optimization hyperparameters
    learning_rate = 0.001
    weight_decay = 0.001
    num_epochs = 1000
    check_loss_freq = 25
    early_stopping_limit = 10
    alpha_beta = [(1.0, 1.0)]
    # model hyperparameters
    latent_dim = 64
    num_layers = 2
    metric_to_optimize = 'f1_macro'
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-SDGNN-{seed_val}')
        for mask_perc_val in mask_perc:
            for num_splits_val in num_splits:
                for unlabeled_perc_val in unlabeled_perc:
                    for noise_perc_val in noise_perc:
                        for alpha_val, beta_val in alpha_beta:
                            with mlflow.start_run():
                                exp_dir = run_experiment(dataset_name=dataset_name,
                                                         overwrite_data=overwrite_data,
                                                         learnable_features=learnable_features,
                                                         init_eps_one=init_eps_one,
                                                         alpha=alpha_val,
                                                         beta=beta_val,
                                                         is_transductive=is_transductive,
                                                         random_masking=random_masking,
                                                         noise_perc=noise_perc_val,
                                                         val_perc=val_perc,
                                                         test_perc=test_perc,
                                                         mask_perc=mask_perc_val,
                                                         unlabeled_perc=unlabeled_perc_val,
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
