#!/bin/bash
cd /home/introai4/.agile/users/hsjung/projects/VLA
conda run -n vla python scripts/train_vlm.py --config configs/vlm_paligemma_distill.yaml --override training.num_epochs=1 training.save_every=999 training.eval_every=1 training.batch_size=4 training.grad_accum_steps=1 training.output_dir=outputs/runs/debug_distill
