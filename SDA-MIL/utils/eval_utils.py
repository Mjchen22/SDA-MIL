import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {
    "dropout": args.drop_out,
    "n_classes": args.n_classes,
    "embed_dim": args.embed_dim,
    "size_arg": args.model_size,
    "k_sample": getattr(args, "B", 8),
    "subtyping": getattr(args, "subtyping", False),

    # Graph
    "use_graph": not getattr(args, "disable_graph", False),
    "graph_layers": getattr(args, "graph_layers", 1),
    "graph_heads": getattr(args, "graph_heads", 4),
    "graph_k": getattr(args, "graph_k", 8),
    "graph_candidate_k": getattr(args, "graph_candidate_k", 16),
    "graph_k_geo": getattr(args, "graph_k_geo", 4),
    "graph_dropout": getattr(args, "graph_dropout", 0.1),
    "graph_radius": getattr(args, "graph_radius", 0.0),
    "graph_use_ffn": not getattr(args, "disable_graph_ffn", False),
    "graph_chunk_size": getattr(args, "graph_chunk_size", 1024),
    "graph_rerank_chunk": getattr(args, "graph_rerank_chunk", 4096),

    # SCA
    "use_sca": getattr(args, "use_sca", False),
    "sca_split_dim": getattr(args, "sca_split_dim", 512),
    "sca_local_k": getattr(args, "sca_local_k", 4),
    "sca_global_k": getattr(args, "sca_global_k", 8),

    # DMA
    "use_dma": getattr(args, "use_dma", False),
    "dma_beta": getattr(args, "dma_beta", 0.5),

    # Positional encoding
    "use_pos_enc": not getattr(args, "disable_pos_enc", False),
    "pos_enc_scale": getattr(args, "pos_enc_scale", 0.1),
    "pos_enc_alpha_init": getattr(args, "pos_enc_alpha_init", 0.0),

    # Context branch
    "use_ctx_branch": not getattr(args, "disable_ctx_branch", False),
    "ctx_inst_weight": getattr(args, "ctx_inst_weight", 0.1),
    "inst_consistency_weight": getattr(args, "inst_consistency_weight", 0.01),
    }
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
