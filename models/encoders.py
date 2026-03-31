"""
Context Encoder
───────────────
Encodes (image I_t, language ℓ, proprio q_t) → context vector C_t.

Architecture:
  1. Frozen SigLIP (or CLIP) encodes image + language → fused vision-language feat
  2. Proprio MLP → proprio embedding
  3. Small MLP fusion → C_t ∈ ℝ^{context_dim}
"""

import torch
import torch.nn as nn
from typing import Optional


# ── SigLIP wrapper ────────────────────────────────────────────────────────────

class SigLIPEncoder(nn.Module):
    """
    Wraps HuggingFace SigLIP (google/siglip-*).
    Returns per-image [CLS] feature fused with text feature.

    If HF is not available, falls back to a dummy encoder for debugging.
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self._build()

    def _build(self):
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(self.model_name)
            self.processor = SiglipProcessor.from_pretrained(self.model_name)
            # Output dim of the vision/text projections
            self.embed_dim = self.model.config.vision_config.hidden_size
            self._use_hf = True
        except Exception as e:
            print(f"[SigLIPEncoder] Could not load {self.model_name}: {e}")
            print("[SigLIPEncoder] Falling back to dummy encoder (debug mode).")
            self.embed_dim = 768
            self._dummy = nn.Linear(1, self.embed_dim, bias=False)
            self._use_hf = False

        if self.freeze and self._use_hf:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,    # (B, 3, H, W)
        input_ids: Optional[torch.Tensor] = None,      # (B, L)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, embed_dim) vision-language fused feature."""
        if not self._use_hf:
            B = pixel_values.shape[0]
            return torch.zeros(B, self.embed_dim, device=pixel_values.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # SigLIP provides image_embeds and text_embeds (both L2-normalised)
        img_feat = outputs.image_embeds   # (B, D)
        if input_ids is not None:
            txt_feat = outputs.text_embeds  # (B, D)
            return img_feat + txt_feat      # simple additive fusion
        return img_feat

    @torch.no_grad()
    def encode_image_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image without language (for future image embedding)."""
        if not self._use_hf:
            B = pixel_values.shape[0]
            return torch.zeros(B, self.embed_dim, device=pixel_values.device)
        # vision_model.pooler_output → L2 normalize (transformers 5.x 호환)
        import torch.nn.functional as F
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        feat = vision_out.pooler_output  # (B, hidden_size)
        return F.normalize(feat, dim=-1)

    def tokenize(self, texts: list, device: torch.device):
        """Tokenize text list → (input_ids, attention_mask) on device."""
        if not self._use_hf:
            return None, None
        encoding = self.processor(
            text=texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )


# ── Proprio MLP ───────────────────────────────────────────────────────────────

class ProprioCoder(nn.Module):
    def __init__(self, proprio_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


# ── Context Encoder (main) ────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Encodes (image, language, proprio) → C_t ∈ ℝ^{context_dim}.

    Args:
        proprio_dim   : proprio feature dimension
        context_dim   : output context size C
        siglip_model  : HF model name for SigLIP
        freeze_encoder: freeze SigLIP weights
        proprio_hidden: hidden dim for proprio MLP
    """

    def __init__(
        self,
        proprio_dim: int,
        context_dim: int = 512,
        siglip_model: str = "google/siglip-base-patch16-224",
        freeze_encoder: bool = True,
        proprio_hidden: int = 128,
    ):
        super().__init__()
        self.context_dim = context_dim

        self.image_lang_encoder = SigLIPEncoder(siglip_model, freeze=freeze_encoder)
        img_lang_dim = self.image_lang_encoder.embed_dim  # e.g. 768

        self.proprio_encoder = ProprioCoder(proprio_dim, proprio_hidden, proprio_hidden)

        fusion_in = img_lang_dim + proprio_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(
        self,
        image: torch.Tensor,           # (B, 3, H, W)
        proprio: torch.Tensor,         # (B, proprio_dim)
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns C_t: (B, context_dim)."""
        img_lang = self.image_lang_encoder(image, input_ids, attention_mask)  # (B, D)
        prop_feat = self.proprio_encoder(proprio)                               # (B, H)
        combined = torch.cat([img_lang, prop_feat], dim=-1)
        return self.fusion(combined)

    def encode_future_image(self, future_image: torch.Tensor) -> torch.Tensor:
        """For semantic future loss: encode future image without language."""
        return self.image_lang_encoder.encode_image_only(future_image)

    def tokenize(self, texts: list, device):
        return self.image_lang_encoder.tokenize(texts, device)


# ── State-only Context Encoder (for low_dim datasets) ─────────────────────────

class StateContextEncoder(nn.Module):
    """
    Encodes concatenated state observation → C_t ∈ ℝ^{context_dim}.
    Used when no image is available (robomimic low_dim, etc.).

    Also encodes 'future state' for semantic future loss.

    Args:
        state_dim       : total concatenated state dimension
        context_dim     : output context size
        future_state_dim: dimension of future semantic target (e.g. object state)
        hidden          : MLP hidden dim
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int = 512,
        future_state_dim: int = 10,
        hidden: int = 256,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.future_state_dim = future_state_dim

        # Main context encoder
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, context_dim),
            nn.LayerNorm(context_dim),
        )

        # Future state encoder (produces "future_feat" for semantic loss)
        self.future_encoder = nn.Sequential(
            nn.Linear(future_state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, context_dim),
            nn.LayerNorm(context_dim),
        )

        # Expose embed_dim for compatibility with builder
        class _FakeSigLIP:
            embed_dim = context_dim
        self.image_lang_encoder = _FakeSigLIP()

    def forward(
        self,
        image: torch.Tensor,           # (B, state_dim) — reused as state
        proprio: torch.Tensor,         # ignored (same as image/state in low_dim)
        input_ids=None,
        attention_mask=None,
    ) -> torch.Tensor:
        return self.net(image)

    def encode_future_image(self, future_state: torch.Tensor) -> torch.Tensor:
        """Encode future state as semantic target (replaces future image embedding)."""
        return self.future_encoder(future_state)

    def tokenize(self, texts: list, device):
        return None, None
