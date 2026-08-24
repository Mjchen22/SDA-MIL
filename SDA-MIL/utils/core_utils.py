import os
import gc
import ctypes
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from dataset_modules.dataset_generic import save_splits
from models.model_clam import CLAM_SB, CLAM_MB
from utils.utils import *

_LOCAL_TOPK = Path(__file__).resolve().parents[1] / "smooth-topk"
if _LOCAL_TOPK.is_dir() and str(_LOCAL_TOPK) not in sys.path:
    sys.path.insert(0, str(_LOCAL_TOPK))

from topk.svm import SmoothTop1SVM


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------
def _free_caches():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_pagecache_for_dataset(dataset):
    try:
        import posix
    except Exception:
        return
    POSIX_FADV_DONTNEED = 4
    files = getattr(dataset, 'pt_paths', None)
    if files is None:
        slide_data = getattr(dataset, 'slide_data', None)
        data_dir = getattr(dataset, 'data_dir', None)
        if slide_data is not None and data_dir is not None:
            try:
                files = [
                    os.path.join(data_dir, 'pt_files', f"{sid}.pt")
                    for sid in slide_data['slide_id'].values
                ]
            except Exception:
                files = None
    if not files:
        return
    for path in files:
        try:
            with open(path, 'rb') as fh:
                posix.posix_fadvise(fh.fileno(), 0, 0, POSIX_FADV_DONTNEED)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Spatial-grid subsampling
# ---------------------------------------------------------------------------
def spatial_grid_subsample_indices(coords, target_count, target_per_cell=64):
    """Spatially-balanced subsample of patches.
    Builds a 2D grid, samples up to `target_per_cell` per cell, then random
    pad/trim to exactly `target_count`. Falls back to random subsample if
    grid cells degenerate.
    """
    N = coords.shape[0]
    if N <= target_count:
        return np.arange(N)
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    n_cells = max(1, target_count // max(1, target_per_cell))
    grid_side = max(2, int(math.ceil(math.sqrt(n_cells))))
    x_min, y_min = coords_np[:, 0].min(), coords_np[:, 1].min()
    x_max, y_max = coords_np[:, 0].max() + 1, coords_np[:, 1].max() + 1
    cell_w = max(1.0, (x_max - x_min) / grid_side)
    cell_h = max(1.0, (y_max - y_min) / grid_side)
    cell_x = np.clip(((coords_np[:, 0] - x_min) // cell_w).astype(int), 0, grid_side - 1)
    cell_y = np.clip(((coords_np[:, 1] - y_min) // cell_h).astype(int), 0, grid_side - 1)
    cell_id = cell_x * grid_side + cell_y
    rng = np.random.default_rng()
    selected = []
    for cid in np.unique(cell_id):
        idx = np.where(cell_id == cid)[0]
        if len(idx) > target_per_cell:
            idx = rng.choice(idx, size=target_per_cell, replace=False)
        selected.append(idx)
    selected = np.concatenate(selected)
    if len(selected) > target_count:
        selected = rng.choice(selected, size=target_count, replace=False)
    elif len(selected) < target_count:
        remaining = np.setdiff1d(np.arange(N), selected, assume_unique=False)
        if len(remaining) > 0:
            need = target_count - len(selected)
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            selected = np.concatenate([selected, extra])
    return selected


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------
def get_curriculum_cfg(epoch, args):
    """Per-epoch loss weights.

    [0, bag_only_epochs)             : pure bag loss
    [bag_only_epochs, proj_only_end) : projected instance loss ramps in
    [proj_only_end, full_start)      : ctx + consistency ramp in
    [full_start, max_epochs)         : full schedule

    proj_w gates the SCA orthogonality loss so it doesn't dominate during
    the bag-only warmup.
    """
    bag_only = int(getattr(args, "curriculum_bag_only_epochs", 2))
    proj_end = int(getattr(args, "curriculum_proj_only_end", 10))
    full_start = int(getattr(args, "curriculum_full_start", 25))
    bw = float(getattr(args, "bag_weight", 0.7))

    if epoch < bag_only:
        return {"use_instance_eval": False, "outer_inst_w": 0.0,
                "proj_w": 0.0, "ctx_w": 0.0, "cons_w": 0.0}
    if epoch < proj_end:
        denom = max(1, proj_end - bag_only)
        proj_w = (epoch - bag_only + 1) / denom
        return {"use_instance_eval": True,
                "outer_inst_w": (1.0 - bw) * proj_w,
                "proj_w": float(proj_w), "ctx_w": 0.0, "cons_w": 0.0}
    if epoch < full_start:
        denom = max(1, full_start - proj_end)
        ramp = (epoch - proj_end + 1) / denom
        return {"use_instance_eval": True, "outer_inst_w": (1.0 - bw),
                "proj_w": 1.0,
                "ctx_w": getattr(args, "ctx_inst_weight", 0.1) * ramp,
                "cons_w": getattr(args, "inst_consistency_weight", 0.01) * ramp}
    return {"use_instance_eval": True, "outer_inst_w": (1.0 - bw),
            "proj_w": 1.0,
            "ctx_w": getattr(args, "ctx_inst_weight", 0.1),
            "cons_w": getattr(args, "inst_consistency_weight", 0.01)}

class Accuracy_Logger(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()
    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
    def log(self, Y_hat, Y):
        Y_hat, Y = int(Y_hat), int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for c in np.unique(Y):
            m = Y == c
            self.data[c]["count"] += m.sum()
            self.data[c]["correct"] += (Y_hat[m] == Y[m]).sum()
    def get_summary(self, c):
        ct, co = self.data[c]["count"], self.data[c]["correct"]
        return (float(co) / ct if ct > 0 else None), co, ct


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=25, verbose=False):
        self.patience = patience; self.stop_epoch = stop_epoch
        self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_loss_min = np.Inf
    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

# ---------------------------------------------------------------------------
# train() 鈥?top-level entry point per fold
# ---------------------------------------------------------------------------
def train(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'],
                os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print(f"\nTraining on {len(train_split)}, validating on {len(val_split)}, testing on {len(test_split)}")

    if args.bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
    else:
        ls = float(getattr(args, "label_smoothing", 0.0))
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 \
            else torch.nn.CrossEntropyLoss()

    instance_loss_fn = SmoothTop1SVM(n_classes=2)
    if torch.cuda.is_available():
        instance_loss_fn = instance_loss_fn.cuda()

    model_dict = {
        "dropout": args.drop_out,
        "n_classes": args.n_classes,
        "embed_dim": args.embed_dim,
        "size_arg": args.model_size,
        "k_sample": args.B,
        "instance_loss_fn": instance_loss_fn,
        "subtyping": args.subtyping,
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
        # Pos enc
        "use_pos_enc": not getattr(args, "disable_pos_enc", False),
        "pos_enc_scale": getattr(args, "pos_enc_scale", 0.1),
        "pos_enc_alpha_init": getattr(args, "pos_enc_alpha_init", 0.0),
        # Ctx branch
        "use_ctx_branch": not getattr(args, "disable_ctx_branch", False),
        "ctx_inst_weight": getattr(args, "ctx_inst_weight", 0.1),
        "inst_consistency_weight": getattr(args, "inst_consistency_weight", 0.01),
    }

    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict)
    else:
        raise NotImplementedError(args.model_type)

    if torch.cuda.is_available():
        model = model.cuda()
    print_network(model)

    optimizer = get_optim(model, args)

    base_lr = args.lr
    min_lr = float(getattr(args, "min_lr", 1e-6))
    warmup_epochs = int(getattr(args, "warmup_epochs", 5))

    def lr_at(epoch_f):
        if epoch_f < warmup_epochs:
            return min_lr + (base_lr - min_lr) * (epoch_f + 1) / max(1, warmup_epochs)
        progress = (epoch_f - warmup_epochs) / max(1, args.max_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    train_loader = get_split_loader(
        train_split, training=True, testing=args.testing,
        weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)

    print('\nSetup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(
        patience=int(getattr(args, "early_stop_patience", 20)),
        stop_epoch=int(getattr(args, "early_stop_min_epoch", 25)),
        verbose=True,
    ) if args.early_stopping else None

    n_classes = args.n_classes

    for epoch in range(args.max_epochs):
        new_lr = lr_at(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        curr_cfg = get_curriculum_cfg(epoch, args)
        train_loop_clam(epoch, model, train_loader, optimizer, n_classes,
                        loss_fn, writer, args, curr_cfg)
        stop = validate_clam(cur, epoch, model, val_loader, n_classes,
                             early_stopping, writer,
                             os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),
                             loss_fn)

        if getattr(args, "release_pagecache", True):
            for ds in [train_split, val_split, test_split]:
                _release_pagecache_for_dataset(ds)
        _free_caches()

        if stop:
            break

    if early_stopping is not None and early_stopping.early_stop:
        ckpt = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
        if os.path.isfile(ckpt):
            model.load_state_dict(torch.load(ckpt))
    else:
        torch.save(model.state_dict(),
                   os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if getattr(args, "verbose_pos_enc", False) and getattr(model, "use_pos_enc", False) \
            and hasattr(model, 'pos_enc'):
        try:
            print(f"[PosEnc] alpha_eff={model.pos_enc.alpha_effective:.6f}")
        except Exception:
            pass

    val_err, val_auc, val_acc, val_f1, _ = summary(model, val_loader, n_classes)
    print('Val | err: {:.4f} AUC: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
        val_err, val_auc, val_acc, val_f1))
    test_err, test_auc, test_acc, test_f1, acc_logger = summary(model, test_loader, n_classes)
    print('Test | err: {:.4f} AUC: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
        test_err, test_auc, test_acc, test_f1))
    for i in range(n_classes):
        f1_i = acc_logger.get_summary(i)[1]
        print('class {}: F1 = {:.4f}'.format(i, f1_i if f1_i is not None else float('nan')))
        if writer is not None:
            writer.add_scalar('final/test_class_{}_f1'.format(i),
                              f1_i if f1_i is not None else 0.0, 0)

    if writer is not None:
        writer.add_scalar('final/val_error', val_err, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_acc', val_acc, 0)
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/test_error', test_err, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_acc', test_acc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.close()

    return {
        'val_auc': val_auc, 'val_acc': val_acc, 'val_f1': val_f1,
        'test_auc': test_auc, 'test_accuracy': test_acc, 'test_f1': test_f1,
    }


# ---------------------------------------------------------------------------
# train_loop_clam
# ---------------------------------------------------------------------------
def train_loop_clam(epoch, model, loader, optimizer, n_classes,
                    bag_loss_fn, writer, args, curr_cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=2)

    train_loss = 0.0
    train_error = 0.0
    train_inst_loss = 0.0
    train_proj_loss = 0.0
    train_ctx_loss = 0.0
    train_cons_loss = 0.0
    train_ortho_loss = 0.0
    inst_count = 0

    n_max_train = int(getattr(args, "n_max_train", 16384))
    spatial_target_per_cell = int(getattr(args, "spatial_target_per_cell", 64))

    use_instance_eval = bool(curr_cfg.get("use_instance_eval", True))
    outer_w = float(curr_cfg.get("outer_inst_w", 0.0))
    proj_w = float(curr_cfg.get("proj_w", 1.0))

    ortho_w_base = float(getattr(args, "sca_ortho_weight", 0.01))
    ortho_w = ortho_w_base * proj_w

    grad_clip = float(getattr(args, "grad_clip", 1.0))
    print_every = int(getattr(args, "print_every", 50))

    for batch_idx, (data, label, coords) in enumerate(loader):
        # subsample patches if too large
        if data.size(0) > n_max_train and coords is not None:
            try:
                sel = spatial_grid_subsample_indices(
                    coords, target_count=n_max_train,
                    target_per_cell=spatial_target_per_cell)
                sel_t = torch.as_tensor(sel, dtype=torch.long)
                data = data.index_select(0, sel_t)
                coords = coords.index_select(0, sel_t)
            except Exception:
                perm = torch.randperm(data.size(0))[:n_max_train]
                data = data.index_select(0, perm)
                if coords is not None:
                    coords = coords.index_select(0, perm)

        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        coords_dev = coords.to(device, non_blocking=True) if coords is not None else None

        logits, Y_prob, Y_hat, _, results_dict = model(
            data, label=label, instance_eval=use_instance_eval, coords=coords_dev)

        acc_logger.log(Y_hat, label)
        loss = bag_loss_fn(logits, label)

        if use_instance_eval and 'instance_loss' in results_dict:
            instance_loss = results_dict['instance_loss']
            train_inst_loss += instance_loss.item()
            train_proj_loss += results_dict.get(
                'proj_instance_loss', torch.tensor(0.0, device=device)).item()
            train_ctx_loss += results_dict.get(
                'ctx_instance_loss', torch.tensor(0.0, device=device)).item()
            train_cons_loss += results_dict.get(
                'inst_consistency_loss', torch.tensor(0.0, device=device)).item()
            inst_count += 1
            inst_preds = results_dict.get('inst_preds', None)
            inst_labels = results_dict.get('inst_labels', None)
            if inst_preds is not None and inst_labels is not None:
                inst_logger.log_batch(inst_preds, inst_labels)
        else:
            instance_loss = torch.tensor(0.0, device=device)

        # SCA orthogonality auxiliary loss
        aux = model.get_aux_losses()
        ortho_loss = aux.get('sca_ortho', torch.tensor(0.0, device=device))
        train_ortho_loss += float(ortho_loss.item()) if torch.is_tensor(ortho_loss) else 0.0

        total_loss = ((1.0 - outer_w) * loss
                      + outer_w * instance_loss
                      + ortho_w * ortho_loss)

        train_loss += loss.item()
        error = calculate_error(Y_hat, label)
        train_error += error

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (batch_idx + 1) % print_every == 0:
            print('batch {}: bag={:.4f} inst={:.4f} ortho={:.4f} total={:.4f} bag_size={}'
                  .format(batch_idx, loss.item(),
                          instance_loss.item() if torch.is_tensor(instance_loss) else 0.0,
                          ortho_loss.item() if torch.is_tensor(ortho_loss) else 0.0,
                          total_loss.item(), int(data.size(0))))

    n_batches = max(1, len(loader))
    train_loss /= n_batches
    train_error /= n_batches
    if inst_count > 0:
        train_inst_loss /= inst_count
        train_proj_loss /= inst_count
        train_ctx_loss /= inst_count
        train_cons_loss /= inst_count
    train_ortho_loss /= n_batches

    print('Epoch {}: bag={:.4f} inst={:.4f} (proj={:.4f} ctx={:.4f} cons={:.4f}) '
          'ortho={:.4f} w_out={:.3f} err={:.4f}'.format(
              epoch, train_loss, train_inst_loss, train_proj_loss, train_ctx_loss,
              train_cons_loss, train_ortho_loss, outer_w, train_error))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_loss', train_inst_loss, epoch)
        writer.add_scalar('train/proj_loss', train_proj_loss, epoch)
        writer.add_scalar('train/ctx_loss', train_ctx_loss, epoch)
        writer.add_scalar('train/cons_loss', train_cons_loss, epoch)
        writer.add_scalar('train/ortho_loss', train_ortho_loss, epoch)


# ---------------------------------------------------------------------------
# validate_clam
# ---------------------------------------------------------------------------
def validate_clam(cur, epoch, model, loader, n_classes,
                  early_stopping, writer, ckpt_name, bag_loss_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss = 0.0
    val_error = 0.0
    val_inst_loss = 0.0
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, coords) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            coords_dev = coords.to(device, non_blocking=True) if coords is not None else None
            logits, Y_prob, Y_hat, _, results_dict = model(
                data, label=label, instance_eval=True, coords=coords_dev)
            loss = bag_loss_fn(logits, label)
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            if 'instance_loss' in results_dict:
                val_inst_loss += results_dict['instance_loss'].item()
                inst_count += 1

    n_batches = max(1, len(loader))
    val_error /= n_batches
    val_loss /= n_batches
    val_inst_loss = val_inst_loss / max(1, inst_count)

    if n_classes == 2:
        try:
            auc = roc_auc_score(labels, prob[:, 1])
        except Exception:
            auc = float('nan')
    else:
        try:
            auc = roc_auc_score(labels, prob, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')

    print('Val | loss: {:.4f}, err: {:.4f}, auc: {:.4f}, inst: {:.4f}'.format(
        val_loss, val_error, auc, val_inst_loss))

    if writer is not None:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)

    if early_stopping is not None:
        early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summary(model, loader, n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_error = 0.0
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    preds = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, coords) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            label_d = label.to(device, non_blocking=True)
            coords_dev = coords.to(device, non_blocking=True) if coords is not None else None
            logits, Y_prob, Y_hat, _, _ = model(
                data, label=label_d, instance_eval=False, coords=coords_dev)
            acc_logger.log(Y_hat, label_d)
            error = calculate_error(Y_hat, label_d)
            test_error += error
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            preds[batch_idx] = int(Y_hat.item())

    test_error /= max(1, len(loader))

    if n_classes == 2:
        try:
            auc = roc_auc_score(labels, prob[:, 1])
        except Exception:
            auc = float('nan')
    else:
        try:
            auc = roc_auc_score(labels, prob, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')

    try:
        acc = accuracy_score(labels, preds)
    except Exception:
        acc = float('nan')
    try:
        f1 = f1_score(labels, preds, average='macro')
    except Exception:
        f1 = float('nan')

    return test_error, auc, acc, f1, acc_logger
