"""
System 2 VLM — PaliGemma-3B Wrapper
────────────────────────────────────
PaliGemma(image, language) → semantic feature f̃ (context_dim)

z 추출 방식 3가지 (ablation 실험):
  'last' : 마지막 토큰 hidden state → Linear → f̃
  'pool' : 마지막 K 토큰 평균 pool → Linear → f̃
  'plan' : [PLAN] 특수 토큰 위치 hidden state → Linear → f̃

transformers/peft 미설치 시 자동으로 dummy 모드로 동작 (smoke test용).
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ── Dummy fallback (transformers 없을 때) ─────────────────────────────────────

class _DummyVLM(nn.Module):
    """PaliGemma 없이도 shape 테스트 가능한 더미 모델."""
    HIDDEN = 2048

    def __init__(self, context_dim: int):
        super().__init__()
        self.proj = nn.Linear(self.HIDDEN, context_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        B = pixel_values.shape[0]
        fake_hidden = torch.randn(B, self.HIDDEN, device=pixel_values.device)
        return self.proj(fake_hidden)

    def gradient_checkpointing_enable(self, **kwargs): pass
    def parameters(self, recurse=True): return super().parameters(recurse=recurse)
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True): return super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


# ── System2VLM ────────────────────────────────────────────────────────────────

class System2VLM(nn.Module):
    """
    PaliGemma-3B를 System 2로 사용하는 VLM wrapper.

    Args:
        model_name       : HuggingFace model id
        context_dim      : 출력 f̃ 차원 (StochFlowPrior의 context_dim과 일치)
        z_form           : 'last' | 'pool' | 'plan'
        pool_k           : z_form='pool'일 때 평균 낼 토큰 수
        freeze_backbone  : True = Stage 1 (VLM frozen), False = Stage 2 (LoRA)
        lora_rank        : LoRA rank
        lora_alpha       : LoRA alpha
        lora_target_modules: LoRA 적용 레이어 이름 (None → 자동 선택)
        proprio_dim      : proprio 차원 (VLM context에 추가 fusion)
        proprio_hidden   : proprio MLP hidden 차원
    """

    PALIGEMMA_HIDDEN = 2048   # Gemma-2B hidden size
    PLAN_SUFFIX = "\nPLAN:"   # z_form='plan' 용 suffix

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        context_dim: int = 256,
        z_form: str = "plan",
        pool_k: int = 8,
        freeze_backbone: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[List[str]] = None,
        proprio_dim: int = 9,
        proprio_hidden: int = 128,
    ):
        super().__init__()
        self.z_form = z_form
        self.pool_k = pool_k
        self.context_dim = context_dim
        self.proprio_dim = proprio_dim
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_targets = lora_target_modules or ["q_proj", "v_proj"]
        self._model_name = model_name
        self._lora_enabled = False

        # ── VLM backbone 로드 ──────────────────────────────────────────────
        self._use_real = self._try_load_paligemma(model_name, freeze_backbone)

        if not self._use_real:
            print("[System2VLM] PaliGemma 미설치 → dummy 모드로 동작 (smoke test용)")
            self.vlm = _DummyVLM(self.PALIGEMMA_HIDDEN)
            self.processor = None

        # ── Projection: VLM hidden → context_dim ──────────────────────────
        fusion_in = self.PALIGEMMA_HIDDEN + proprio_hidden
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, proprio_hidden),
            nn.SiLU(),
            nn.LayerNorm(proprio_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.SiLU(),
            nn.Linear(512, context_dim),
            nn.LayerNorm(context_dim),
        )

    def _try_load_paligemma(self, model_name: str, freeze: bool) -> bool:
        try:
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
            print(f"[System2VLM] PaliGemma 로딩 중: {model_name} ...")
            self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
            self.processor = PaliGemmaProcessor.from_pretrained(model_name)
            self.vlm.tie_weights()  # DDP rank간 param count 불일치 방지
            if freeze:
                for p in self.vlm.parameters():
                    p.requires_grad = False
                print("[System2VLM] Stage 1: VLM frozen")
            else:
                print("[System2VLM] VLM 전체 학습 모드")
            n_params = sum(p.numel() for p in self.vlm.parameters()) / 1e9
            print(f"[System2VLM] 로딩 완료 ({n_params:.1f}B params)")
            return True
        except Exception as e:
            print(f"[System2VLM] PaliGemma 로드 실패: {e}")
            return False

    # ── LoRA 활성화 (Stage 2 진입 시 호출) ───────────────────────────────────

    def enable_lora(self):
        """Stage 2: PaliGemma backbone에 LoRA 적용."""
        if self._lora_enabled:
            return
        if not self._use_real:
            print("[System2VLM] dummy 모드 — LoRA skip")
            return
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_alpha,
                target_modules=self._lora_targets,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.vlm = get_peft_model(self.vlm, lora_cfg)
            self._lora_enabled = True
            n_lora = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
            print(f"[System2VLM] Stage 2: LoRA 활성화 ({n_lora/1e6:.1f}M trainable params)")
            self.vlm.print_trainable_parameters()
        except ImportError:
            print("[System2VLM] peft 미설치 — LoRA skip, 전체 파라미터 학습")
            for p in self.vlm.parameters():
                p.requires_grad = True

    def lora_state_dict(self):
        """LoRA 가중치만 반환 (체크포인트 저장 시)."""
        if not self._lora_enabled or not self._use_real:
            return {}
        try:
            from peft import get_peft_model_state_dict
            return get_peft_model_state_dict(self.vlm)
        except Exception:
            return {}

    # ── 입력 전처리 ───────────────────────────────────────────────────────────

    def prepare_inputs(
        self,
        raw_images: torch.Tensor,   # (B, H, W, 3) uint8 or (B, 3, H, W) float
        texts: List[str],
        device: torch.device,
    ):
        """
        PaliGemmaProcessor로 이미지+텍스트 전처리.
        raw_images: uint8 HWC 또는 float CHW 텐서

        Returns: (pixel_values, input_ids, attention_mask) — 모두 device로 이동
        """
        if not self._use_real or self.processor is None:
            # Dummy: 그냥 float 텐서로 변환
            B = raw_images.shape[0]
            if raw_images.dim() == 4 and raw_images.shape[-1] == 3:
                pv = raw_images.permute(0, 3, 1, 2).float() / 255.0
            else:
                pv = raw_images.float()
            ids = torch.zeros(B, 32, dtype=torch.long)
            mask = torch.ones(B, 32, dtype=torch.long)
            return pv.to(device), ids.to(device), mask.to(device)

        from PIL import Image
        import numpy as np

        # (B, H, W, 3) uint8 → PIL 리스트
        if raw_images.dim() == 4 and raw_images.shape[-1] == 3:
            imgs_np = raw_images.cpu().numpy().astype(np.uint8)
        elif raw_images.dim() == 4 and raw_images.shape[1] == 3:
            # CHW float → HWC uint8
            imgs_np = (raw_images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected raw_image shape: {raw_images.shape}")

        pil_images = [Image.fromarray(img) for img in imgs_np]

        # z_form='plan' → 텍스트에 PLAN suffix 추가
        if self.z_form == "plan":
            prompts = [f"<image>{t}{self.PLAN_SUFFIX}" for t in texts]
        else:
            prompts = [f"<image>{t}" for t in texts]

        enc = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,   # 이미지 토큰 256 + 텍스트 여유
        )
        return (
            enc["pixel_values"].to(device),
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,   # (B, 3, 224, 224)
        input_ids: torch.Tensor,      # (B, seq_len)
        attention_mask: torch.Tensor, # (B, seq_len)
        proprio: Optional[torch.Tensor] = None,  # (B, proprio_dim)
    ) -> torch.Tensor:
        """
        Returns: f̃ (B, context_dim)
        """
        if self._use_real:
            # transformers 5.x: training mode에서 token_type_ids 필수
            # 0 = image token, 1 = text token
            n_img = getattr(self.processor, "image_seq_length", 256)
            B_seq, seq_len = input_ids.shape
            token_type_ids = torch.ones(B_seq, seq_len, dtype=torch.long,
                                        device=input_ids.device)
            token_type_ids[:, :n_img] = 0

            outputs = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # (B, seq_len, 2048)
        else:
            # Dummy: 랜덤 hidden state
            B = pixel_values.shape[0]
            seq_len = input_ids.shape[1]
            hidden = torch.randn(B, seq_len, self.PALIGEMMA_HIDDEN,
                                 device=pixel_values.device)

        # z_form에 따라 VLM feature 추출 (float32로 캐스팅 — proprio_encoder/proj는 float32)
        vlm_feat = self._extract_feature(hidden, attention_mask).float()  # (B, 2048)

        # Proprio fusion
        if proprio is not None:
            prop_feat = self.proprio_encoder(proprio.float())
            fused = torch.cat([vlm_feat, prop_feat], dim=-1)
        else:
            B = vlm_feat.shape[0]
            zeros = torch.zeros(B, self.proprio_encoder[0].in_features,
                                device=vlm_feat.device)
            prop_feat = self.proprio_encoder(zeros)
            fused = torch.cat([vlm_feat, prop_feat], dim=-1)

        return self.proj(fused)  # (B, context_dim), float32

    def _extract_feature(
        self,
        hidden: torch.Tensor,       # (B, seq_len, hidden_size)
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """z_form에 따라 hidden state에서 feature 추출 → (B, hidden_size)"""
        if self.z_form == "last":
            # 마지막 실제 토큰 위치의 hidden state
            # attention_mask에서 마지막 1인 위치 찾기
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            B, _, D = hidden.shape
            idx = lengths.clamp(0, hidden.shape[1] - 1)
            feat = hidden[torch.arange(B, device=hidden.device), idx]  # (B, D)

        elif self.z_form == "pool":
            # 마지막 pool_k 토큰 평균
            k = min(self.pool_k, hidden.shape[1])
            feat = hidden[:, -k:, :].mean(dim=1)  # (B, D)

        elif self.z_form == "plan":
            # PLAN suffix 마지막 토큰 위치 = 마지막 실제 토큰
            # (prepare_inputs에서 PLAN suffix를 마지막에 붙였으므로)
            lengths = attention_mask.sum(dim=1) - 1
            B = hidden.shape[0]
            idx = lengths.clamp(0, hidden.shape[1] - 1)
            feat = hidden[torch.arange(B, device=hidden.device), idx]  # (B, D)

        else:
            raise ValueError(f"Unknown z_form: {self.z_form}")

        return feat

    # ── 파라미터 그룹 (optimizer 분리용) ──────────────────────────────────────

    def vlm_lora_parameters(self):
        """LoRA 파라미터만 반환."""
        return [p for n, p in self.named_parameters()
                if "vlm" in n and p.requires_grad]

    def non_vlm_parameters(self):
        """proj, proprio_encoder 파라미터 반환."""
        return [p for n, p in self.named_parameters()
                if "vlm" not in n and p.requires_grad]
