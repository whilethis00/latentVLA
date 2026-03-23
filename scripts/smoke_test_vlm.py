"""
VLM Smoke Test — PaliGemma 없이도 shape/interface 검증

사용법:
  python scripts/smoke_test_vlm.py

PaliGemma 미설치 시 dummy 모드로 동작.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import traceback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트 차원
B = 2
H = 8
ACTION_DIM = 7
PROPRIO_DIM = 9
CONTEXT_DIM = 256
Z_DIM = 128


def make_dummy_batch():
    return {
        "raw_image": torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8),
        "image": torch.randn(B, 3, 224, 224),
        "proprio": torch.randn(B, PROPRIO_DIM),
        "language": ["pick up the red cup"] * B,
        "actions": torch.randn(B, H, ACTION_DIM),
        "future_image": torch.randn(B, 3, 224, 224),
    }


def test_system2_vlm_all_forms():
    from models.system2_vlm import System2VLM
    batch = make_dummy_batch()

    for z_form in ["last", "pool", "plan"]:
        vlm = System2VLM(
            context_dim=CONTEXT_DIM,
            z_form=z_form,
            proprio_dim=PROPRIO_DIM,
        ).to(DEVICE)

        pv, ids, mask = vlm.prepare_inputs(
            batch["raw_image"], batch["language"], DEVICE
        )
        f_tilde = vlm(pv, ids, mask, batch["proprio"].to(DEVICE))

        assert f_tilde.shape == (B, CONTEXT_DIM), \
            f"z_form={z_form}: 기대 {(B, CONTEXT_DIM)}, 실제 {f_tilde.shape}"
        assert f_tilde.dtype == torch.float32, f"dtype must be float32, got {f_tilde.dtype}"
        print(f"  System2VLM z_form={z_form:5s}  f̃: {tuple(f_tilde.shape)}  ✓")


def test_latent_vla_forward():
    from models.system2_vlm import System2VLM
    from models.latent_vla import LatentVLA

    system2 = System2VLM(
        context_dim=CONTEXT_DIM,
        z_form="plan",
        proprio_dim=PROPRIO_DIM,
    )
    model = LatentVLA(
        system2=system2,
        action_dim=ACTION_DIM,
        action_horizon=H,
        z_dim=Z_DIM,
        context_dim=CONTEXT_DIM,
    ).to(DEVICE)

    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in make_dummy_batch().items()}

    # compute_loss
    loss_dict = model.compute_loss(batch, DEVICE, semantic_weight=0.1)
    assert "total_loss" in loss_dict
    assert "action_flow_loss" in loss_dict
    assert "prior_flow_loss" in loss_dict
    loss_dict["total_loss"].backward()
    print(f"  LatentVLA compute_loss  total={loss_dict['total_loss'].item():.4f}  ✓")

    # predict (prior)
    with torch.no_grad():
        pred = model.predict(batch, DEVICE)
    assert pred.shape == (B, H, ACTION_DIM), pred.shape
    print(f"  LatentVLA predict       shape={tuple(pred.shape)}  ✓")

    # predict n_samples=3
    with torch.no_grad():
        pred_multi = model.predict(batch, DEVICE, n_samples=3)
    assert pred_multi.shape == (B, 3, H, ACTION_DIM), pred_multi.shape
    print(f"  LatentVLA predict×3     shape={tuple(pred_multi.shape)}  ✓")

    # state_dict_for_save
    sd = model.state_dict_for_save()
    assert "system1" in sd
    print(f"  LatentVLA state_dict_for_save  keys={list(sd.keys())}  ✓")


def test_vlm_trainer_step():
    from models.system2_vlm import System2VLM
    from models.latent_vla import LatentVLA
    from training.trainer_vlm import VLMTrainer
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        def __len__(self): return 4
        def __getitem__(self, i):
            return {
                "raw_image": torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8),
                "image": torch.randn(3, 224, 224),
                "proprio": torch.randn(PROPRIO_DIM),
                "language": "pick up the cup",
                "actions": torch.randn(H, ACTION_DIM),
                "future_image": torch.randn(3, 224, 224),
            }

    system2 = System2VLM(context_dim=CONTEXT_DIM, z_form="plan", proprio_dim=PROPRIO_DIM)
    model = LatentVLA(system2=system2, action_dim=ACTION_DIM,
                      action_horizon=H, z_dim=Z_DIM, context_dim=CONTEXT_DIM)

    ds = DummyDataset()
    loader = DataLoader(ds, batch_size=2)

    cfg = {
        "system2": {"z_form": "plan"},
        "training": {
            "num_epochs": 2, "batch_size": 2, "grad_accum_steps": 1,
            "learning_rate": 3e-4, "lora_lr": 3e-5, "weight_decay": 1e-4,
            "grad_clip": 1.0, "stage2_epoch": 2,
            "eval_every": 2, "save_every": 2,
            "best_of_ks": [1, 5],
            "output_dir": "/tmp/latent_vla_smoke",
        },
        "loss": {"prior_weight": 1.0, "semantic_future_weight": 0.1},
        "logging": {"use_wandb": False},
    }

    trainer = VLMTrainer(model, loader, loader, cfg, DEVICE)
    trainer.train()
    print(f"  VLMTrainer 2 epochs  ✓")


def main():
    print(f"VLM Smoke Tests on {DEVICE}\n")

    tests = [
        ("System2VLM (3 z_forms)", test_system2_vlm_all_forms),
        ("LatentVLA forward/predict", test_latent_vla_forward),
        ("VLMTrainer 2 epochs", test_vlm_trainer_step),
    ]

    passed = 0
    for name, fn in tests:
        print(f"[ {name} ]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
        print()

    print(f"{passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("All VLM smoke tests PASSED. ✓")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
