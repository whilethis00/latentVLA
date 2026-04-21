#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/default.yaml --override model.type=flat_flow data.dataset_type=libero data.dataset_path=/home/introai4/home_lustre/introai4/libero/libero_object_openvla_processed/ data.proprio_dim=8 training.output_dir=outputs/runs/flat_flow_100ep_20260404 training.num_epochs=100
