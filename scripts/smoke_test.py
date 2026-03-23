"""
Smoke test — no real dataset needed.
Creates random tensors and verifies forward/loss/predict passes for all models.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import traceback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions
B = 4
H = 8
ACTION_DIM = 7
PROPRIO_DIM = 9
CONTEXT_DIM = 512
Z_DIM = 128
FUTURE_DIM = 768  # SigLIP dummy dim


def make_batch():
    context = torch.randn(B, CONTEXT_DIM).to(DEVICE)
    actions = torch.randn(B, H, ACTION_DIM).to(DEVICE)
    future_feat = torch.randn(B, FUTURE_DIM).to(DEVICE)
    return context, actions, future_feat


def test_flat_flow():
    from models.flat_flow import FlatFlow
    m = FlatFlow(CONTEXT_DIM, ACTION_DIM, H).to(DEVICE)
    ctx, act, _ = make_batch()
    loss_dict = m.compute_loss(ctx, act)
    assert "total_loss" in loss_dict
    pred = m.predict(ctx)
    assert pred.shape == (B, H, ACTION_DIM), pred.shape
    print(f"  FlatFlow OK | loss={loss_dict['total_loss'].item():.4f}")


def test_det_latent():
    from models.det_latent import DetLatent
    m = DetLatent(CONTEXT_DIM, ACTION_DIM, H, Z_DIM, FUTURE_DIM).to(DEVICE)
    ctx, act, fut = make_batch()
    loss_dict = m.compute_loss(ctx, act, fut)
    assert "action_flow_loss" in loss_dict
    assert "prior_loss" in loss_dict
    assert "semantic_future_loss" in loss_dict
    pred_prior = m.predict(ctx)
    pred_post = m.predict(ctx, use_posterior=True, actions_for_posterior=act, future_feat_for_posterior=fut)
    assert pred_prior.shape == (B, H, ACTION_DIM)
    assert pred_post.shape == (B, H, ACTION_DIM)
    print(f"  DetLatent OK | action={loss_dict['action_flow_loss'].item():.4f} "
          f"prior={loss_dict['prior_loss'].item():.4f} "
          f"sem={loss_dict['semantic_future_loss'].item():.4f}")


def test_stoch_vae():
    from models.stoch_latent_vae import StochLatentVAE
    m = StochLatentVAE(CONTEXT_DIM, ACTION_DIM, H, Z_DIM, FUTURE_DIM).to(DEVICE)
    ctx, act, fut = make_batch()
    loss_dict = m.compute_loss(ctx, act, fut)
    assert "kl_loss" in loss_dict
    pred = m.predict(ctx)
    assert pred.shape == (B, H, ACTION_DIM)
    pred_multi = m.predict(ctx, n_samples=3)
    assert pred_multi.shape == (B, 3, H, ACTION_DIM)
    print(f"  StochVAE OK | action={loss_dict['action_flow_loss'].item():.4f} "
          f"kl={loss_dict['kl_loss'].item():.4f}")


def test_stoch_flow_prior():
    from models.stoch_latent_flow_prior import StochLatentFlowPrior
    m = StochLatentFlowPrior(CONTEXT_DIM, ACTION_DIM, H, Z_DIM, FUTURE_DIM).to(DEVICE)
    ctx, act, fut = make_batch()
    loss_dict = m.compute_loss(ctx, act, fut)
    assert "prior_flow_loss" in loss_dict
    pred = m.predict(ctx)
    assert pred.shape == (B, H, ACTION_DIM)
    pred_multi = m.predict(ctx, n_samples=3)
    assert pred_multi.shape == (B, 3, H, ACTION_DIM)
    print(f"  StochFlowPrior OK | action={loss_dict['action_flow_loss'].item():.4f} "
          f"prior_flow={loss_dict['prior_flow_loss'].item():.4f}")


def test_offline_metrics():
    from models.flat_flow import FlatFlow
    from models.det_latent import DetLatent
    from evaluation.metrics import OfflineEvaluator

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_lang_encoder = type("E", (), {"embed_dim": FUTURE_DIM})()

        def forward(self, image, proprio, input_ids=None, attn=None):
            return torch.randn(image.shape[0], CONTEXT_DIM, device=image.device)

        def encode_future_image(self, future_image):
            return torch.randn(future_image.shape[0], FUTURE_DIM, device=future_image.device)

        def tokenize(self, texts, device):
            return None, None

    enc = DummyEncoder().to(DEVICE)

    # FlatFlow evaluator
    policy = FlatFlow(CONTEXT_DIM, ACTION_DIM, H).to(DEVICE)
    evaluator = OfflineEvaluator(enc, policy, DEVICE)

    # Create fake dataloader
    from torch.utils.data import DataLoader, TensorDataset
    fake_images = torch.randn(8, 3, 224, 224)
    fake_proprio = torch.randn(8, PROPRIO_DIM)
    fake_actions = torch.randn(8, H, ACTION_DIM)
    fake_future = torch.randn(8, 3, 224, 224)

    class DictDataset(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, i):
            return {
                "image": fake_images[i],
                "proprio": fake_proprio[i],
                "actions": fake_actions[i],
                "future_image": fake_future[i],
                "language": "robot task",
            }

    loader = DataLoader(DictDataset(), batch_size=4)
    metrics = evaluator.evaluate(loader)
    assert "action_mse_prior" in metrics
    print(f"  OfflineEvaluator OK | action_mse_prior={metrics['action_mse_prior']:.4f}")


def main():
    print(f"Running smoke tests on {DEVICE}\n")
    tests = [
        ("FlatFlow", test_flat_flow),
        ("DetLatent", test_det_latent),
        ("StochLatentVAE", test_stoch_vae),
        ("StochLatentFlowPrior", test_stoch_flow_prior),
        ("OfflineMetrics", test_offline_metrics),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("All smoke tests PASSED.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
