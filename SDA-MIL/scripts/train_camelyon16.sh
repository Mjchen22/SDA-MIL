#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/path/to/CAMELYON16/features"}
SPLIT_DIR_NAME=${SPLIT_DIR_NAME:-"task_1_tumor_vs_normal_100"}
RESULTS_DIR=${RESULTS_DIR:-"./results/camelyon16"}
SEED=${SEED:-1}
K_FOLDS=${K_FOLDS:-5}
MAX_EPOCHS=${MAX_EPOCHS:-80}

for FOLD in 0 1 2 3 4; do
  python main.py \
    --task task_1_tumor_vs_normal \
    --model_type clam_sb \
    --data_root_dir "${DATA_ROOT}" \
    --split_dir "${SPLIT_DIR_NAME}" \
    --results_dir "${RESULTS_DIR}" \
    --k "${K_FOLDS}" \
    --k_start "${FOLD}" \
    --k_end "$((FOLD + 1))" \
    --seed "${SEED}" \
    --exp_code "camelyon16_sda_mil" \
    --embed_dim 1024 \
    --model_size small \
    --bag_loss ce \
    --inst_loss svm \
    --B 32 \
    --bag_weight 0.7 \
    --weighted_sample \
    --drop_out 0.25 \
    --reg 1e-5 \
    --label_smoothing 0.05 \
    --lr 2e-4 \
    --max_epochs "${MAX_EPOCHS}" \
    --warmup_epochs 5 \
    --early_stopping \
    --early_stop_patience 20 \
    --early_stop_min_epoch 25 \
    --grad_clip 1.0 \
    --n_max_train 65536 \
    --spatial_target_per_cell 64 \
    --disable_ctx_branch \
    --pos_enc_scale 0.1 \
    --pos_enc_alpha_init 0.0 \
    --use_sca \
    --sca_split_dim 512 \
    --sca_local_k 4 \
    --sca_global_k 8 \
    --sca_ortho_weight 0.01 \
    --graph_layers 1 \
    --graph_heads 4 \
    --graph_candidate_k 16 \
    --graph_k_geo 2 \
    --graph_dropout 0.1 \
    --graph_chunk_size 1024 \
    --graph_rerank_chunk 4096 \
    --use_dma \
    --dma_beta 1.0
done
