#!/bin/bash
# ─── Ablation Runner ─────────────────────────────────────────────────────────
# Runs all 4 core ablations sequentially (or in parallel with &).
#
# Usage:
#   bash scripts/run_ablations.sh /path/to/libero_object/ libero
#   bash scripts/run_ablations.sh /path/to/lift.hdf5 robomimic

DATASET_PATH=${1:-/path/to/data}
DATASET_TYPE=${2:-libero}
CONFIG=configs/default.yaml
BASE_OUT=outputs/runs

python scripts/train.py \
    --config $CONFIG \
    --override model.type=flat_flow \
               data.dataset_type=$DATASET_TYPE \
               data.dataset_path=$DATASET_PATH \
               training.output_dir=$BASE_OUT/flat_flow \
               logging.run_name=flat_flow

python scripts/train.py \
    --config $CONFIG \
    --override model.type=det_latent \
               data.dataset_type=$DATASET_TYPE \
               data.dataset_path=$DATASET_PATH \
               training.output_dir=$BASE_OUT/det_latent \
               logging.run_name=det_latent

python scripts/train.py \
    --config $CONFIG \
    --override model.type=stoch_vae \
               data.dataset_type=$DATASET_TYPE \
               data.dataset_path=$DATASET_PATH \
               training.output_dir=$BASE_OUT/stoch_vae \
               logging.run_name=stoch_vae

python scripts/train.py \
    --config $CONFIG \
    --override model.type=stoch_flow_prior \
               data.dataset_type=$DATASET_TYPE \
               data.dataset_path=$DATASET_PATH \
               training.output_dir=$BASE_OUT/stoch_flow_prior \
               logging.run_name=stoch_flow_prior

# ── Ablation 4: semantic loss off ──────────────────────────────────────────
python scripts/train.py \
    --config $CONFIG \
    --override model.type=det_latent \
               data.dataset_type=$DATASET_TYPE \
               data.dataset_path=$DATASET_PATH \
               loss.semantic_future_weight=0.0 \
               training.output_dir=$BASE_OUT/det_latent_no_semantic \
               logging.run_name=det_latent_no_semantic

echo "All ablations done."
