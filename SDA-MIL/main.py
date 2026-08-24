import argparse
import os
import time
import random

import numpy as np
import pandas as pd
import torch

from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main(args):
    folds = np.arange(args.k_start if args.k_start != -1 else 0,
                      (args.k_end if args.k_end != -1 else args.k))
    all_results = {
        'folds': [], 'test_auc': [], 'val_auc': [],
        'test_acc': [], 'val_acc': [], 'test_accuracy': [],
        'val_accuracy': [], 'test_f1': [], 'val_f1': [],
    }

    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path=os.path.join(args.split_dir, 'splits_{}.csv'.format(i)),
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        results = train(datasets, i, args)

        all_results['folds'].append(int(i))
        all_results['test_auc'].append(results['test_auc'])
        all_results['val_auc'].append(results['val_auc'])
        all_results['test_acc'].append(results['test_accuracy'])
        all_results['val_acc'].append(results['val_acc'])
        all_results['test_accuracy'].append(results['test_accuracy'])
        all_results['val_accuracy'].append(results['val_acc'])
        all_results['test_f1'].append(results['test_f1'])
        all_results['val_f1'].append(results['val_f1'])

        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.results_dir, 'summary.csv'), index=False)


parser = argparse.ArgumentParser(description='CLAM training with SCA/DMA')

# Base data/training
parser.add_argument('--data_root_dir', type=str, default=None, required=True)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--early_stop_patience', type=int, default=20)
parser.add_argument('--early_stop_min_epoch', type=int, default=25)
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], default='clam_sb')
parser.add_argument('--exp_code', type=str, required=True)
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--model_size', type=str, choices=['micro', 'small', 'big'], default='small')
parser.add_argument('--task', type=str)

# CLAM core
parser.add_argument('--no_inst_cluster', action='store_true', default=False)
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
parser.add_argument('--subtyping', action='store_true', default=False)
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--B', type=int, default=8)

parser.add_argument('--disable_graph', action='store_true', default=False,
                    help='ABLATION: disable graph refiner entirely (clean CLAM)')
parser.add_argument('--graph_layers', type=int, default=1)
parser.add_argument('--graph_heads', type=int, default=4)
parser.add_argument('--graph_k', type=int, default=8)
parser.add_argument('--graph_candidate_k', type=int, default=16)
parser.add_argument('--graph_k_geo', type=int, default=4)
parser.add_argument('--graph_dropout', type=float, default=0.1)
parser.add_argument('--graph_radius', type=float, default=0.0)
parser.add_argument('--disable_graph_ffn', action='store_true', default=False)
parser.add_argument('--graph_chunk_size', type=int, default=1024)
parser.add_argument('--graph_rerank_chunk', type=int, default=4096)

# SCA
parser.add_argument('--use_sca', action='store_true', default=False,
                    help='enable Scale-Aware Cross-patch Attention')
parser.add_argument('--sca_split_dim', type=int, default=512)
parser.add_argument('--sca_local_k', type=int, default=4)
parser.add_argument('--sca_global_k', type=int, default=8)
parser.add_argument('--sca_ortho_weight', type=float, default=0.01)

# DMA
parser.add_argument('--use_dma', action='store_true', default=False,
                    help='enable Distance-Modulated Attention Temperature')
parser.add_argument('--dma_beta', type=float, default=0.5,
                    help='DMA temperature coefficient; 0 = standard temperature')

parser.add_argument('--disable_pos_enc', action='store_true', default=False,
                    help='ABLATION: disable spatial positional encoding')
parser.add_argument('--pos_enc_scale', type=float, default=0.1)
parser.add_argument('--pos_enc_alpha_init', type=float, default=0.0)

parser.add_argument('--disable_ctx_branch', action='store_true', default=False,
                    help='ABLATION: disable auxiliary instance head over context features')
parser.add_argument('--ctx_inst_weight', type=float, default=0.1)
parser.add_argument('--inst_consistency_weight', type=float, default=0.01)

parser.add_argument('--n_max_train', type=int, default=16384,
                    help='cap on patches per slide during training')
parser.add_argument('--spatial_target_per_cell', type=int, default=64)

parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--release_pagecache', action='store_true', default=True)

parser.add_argument('--curriculum_bag_only_epochs', type=int, default=2)
parser.add_argument('--curriculum_proj_only_end', type=int, default=10)
parser.add_argument('--curriculum_full_start', type=int, default=25)

parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--verbose_pos_enc', action='store_true', default=False)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_torch(args.seed)

settings = {
    'num_splits': args.k,
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    'use_drop_out': args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt,
    'n_max_train': args.n_max_train,
    'warmup_epochs': args.warmup_epochs,
    'min_lr': args.min_lr,
    'grad_clip': args.grad_clip,
    'label_smoothing': args.label_smoothing,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_min_epoch': args.early_stop_min_epoch,
}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B,
        'use_graph': not args.disable_graph,
        'graph_layers': args.graph_layers,
        'graph_heads': args.graph_heads,
        'graph_k': args.graph_k,
        'graph_candidate_k': args.graph_candidate_k,
        'graph_k_geo': args.graph_k_geo,
        'graph_dropout': args.graph_dropout,
        'graph_chunk_size': args.graph_chunk_size,
        'graph_rerank_chunk': args.graph_rerank_chunk,
        'use_sca': args.use_sca,
        'sca_split_dim': args.sca_split_dim,
        'sca_local_k': args.sca_local_k,
        'sca_global_k': args.sca_global_k,
        'sca_ortho_weight': args.sca_ortho_weight,
        'use_dma': args.use_dma,
        'dma_beta': args.dma_beta,
        'use_pos_enc': not args.disable_pos_enc,
        'pos_enc_scale': args.pos_enc_scale,
        'pos_enc_alpha_init': args.pos_enc_alpha_init,
        'use_ctx_branch': not args.disable_ctx_branch,
        'ctx_inst_weight': args.ctx_inst_weight,
        'inst_consistency_weight': args.inst_consistency_weight,
    })

print('\nLoad Dataset')
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True,
        label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
        patient_strat=False, ignore=[])
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True,
        label_dict={'kirp_tcga':0, 'kirc_tcga':1, 'kich_tcga':2},
        patient_strat=False, patient_voting='maj', ignore=[])
    if args.model_type in ['clam_sb', 'clam_mb']:
        args.subtyping = True
else:
    raise NotImplementedError(args.task)

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)

args.results_dir = os.path.join(
    args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)
print('split_dir:', args.split_dir)
assert os.path.isdir(args.split_dir), f'split dir does not exist: {args.split_dir}'

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}: {}".format(key, val))


if __name__ == "__main__":
    start_time = time.time()
    main(args)
    end_time = time.time()
    print("finished!")
    print("end script  total {:.2f}s".format(end_time - start_time))