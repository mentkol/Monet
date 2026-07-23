"""
SigLIP2 + DINOv2 Ensemble Detector for AI-generated content.
Uses the Bombek1/ai-image-detector-siglip-dinov2 model via PyTorch.
"""

import numpy as np
import cv2
import os
import sys
import time
import math
import torch
import torch.nn as nn
import timm
from transformers import AutoProcessor, SiglipVisionModel
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image


class LoRALinear(nn.Module):
    """Custom LoRA implementation for DINOv2 QKV layers."""
    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class ClassificationHead(nn.Module):
    """MLP classification head with LayerNorm and dropout."""
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class EnsembleAIDetector(nn.Module):
    """Ensemble model combining SigLIP2 and DINOv2 for AI image detection."""
    def __init__(self, siglip_model_name, dinov2_model_name, image_size=392, torch_dtype=torch.float32):
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(
            siglip_model_name, torch_dtype=torch_dtype
        )
        self.siglip_dim = self.siglip.config.hidden_size
        self.dinov2 = timm.create_model(
            dinov2_model_name, pretrained=True, num_classes=0, img_size=image_size
        )
        self.dinov2_dim = self.dinov2.num_features
        self.classifier = ClassificationHead(self.siglip_dim + self.dinov2_dim)

    def forward(self, siglip_pixels, dinov2_pixels):
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        combined = torch.cat([siglip_features.float(), dinov2_features.float()], dim=-1)
        logits = self.classifier(combined)
        return logits, siglip_features, dinov2_features


class SigLIPDinoV2Detector:
    def __init__(self, cache_dir=None):
        """
        Initialize the SigLIP+DINOv2 detector.

        Downloads model weights from HuggingFace on first run and caches
        them locally for subsequent fast loading.

        Args:
            cache_dir: Optional cache directory for the model weights
        """
        self.model = None
        self.siglip_processor = None
        self.dinov2_transform = None
        self.device = None
        self.loaded = False
        self._default_batch_size = int(os.environ.get("MONET_BATCH_SIZE", "4"))
        self._sample_count = int(os.environ.get("MONET_DETECTOR_FRAMES", "3"))

        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models", "siglip_dinov2"
            )
        self.cache_dir = os.path.abspath(cache_dir)

        self._load_model()

    def _load_model(self):
        """Load or download the model."""
        try:
            print(f"Loading SigLIP+DINOv2 detector...")
            start = time.time()

            forced = os.environ.get("MONET_DEVICE", "").strip().lower()

            # Intel Arc / XPU via Intel Extension for PyTorch (optional import).
            # If IPEX isn't installed (e.g. on macOS), XPU is simply unavailable.
            self._xpu_available = False
            if forced != "cpu":
                try:
                    import intel_extension_for_pytorch  # noqa: F401 — registers torch.xpu
                    if torch.xpu.is_available():
                        self._xpu_available = True
                except Exception:
                    pass
            if forced == "xpu" and not self._xpu_available:
                print("  ⚠️  MONET_DEVICE=xpu but Intel XPU/IPEX unavailable; ignoring.")
                forced = ""

            if forced in ("cpu", "mps", "cuda", "xpu"):
                self.device = torch.device(forced)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif self._xpu_available:
                self.device = torch.device("xpu")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            if self.device.type == "xpu":
                # Arc XMX engines are optimized for bfloat16.
                _xpu = os.environ.get("MONET_XPU_DTYPE", "bf16").strip().lower()
                self.inference_dtype = torch.float16 if _xpu in ("fp16", "float16") else torch.bfloat16
            elif self.device.type == "mps":
                # float32 is the only stable choice on MPS: float16 matmuls hit a
                # Metal accumulator-dtype assertion that aborts the process.
                # Override with MONET_MPS_DTYPE=fp16 only on setups known to work.
                _mps = os.environ.get("MONET_MPS_DTYPE", "fp32").strip().lower()
                self.inference_dtype = torch.float16 if _mps == "fp16" else torch.float32
            elif self.device.type == "cuda":
                self.inference_dtype = torch.bfloat16
            else:
                self.inference_dtype = torch.float32

            # --- Download weights ---
            os.makedirs(self.cache_dir, exist_ok=True)
            weights_path = os.path.join(self.cache_dir, "pytorch_model.pt")
            if not os.path.exists(weights_path):
                print("  Downloading model weights (one-time)...")
                weights_path = hf_hub_download(
                    repo_id="Bombek1/ai-image-detector-siglip-dinov2",
                    filename="pytorch_model.pt",
                    local_dir=self.cache_dir
                )
            else:
                print("  Loading cached model weights...")

            # --- Load checkpoint ---
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})

            siglip_name = config.get("siglip_model", "google/siglip2-so400m-patch14-384")
            dinov2_name = config.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m")
            image_size = config.get("image_size", 392)
            lora_rank = config.get("lora_rank", 32)
            lora_alpha = config.get("lora_alpha", 64)
            lora_dropout = config.get("lora_dropout", 0.1)

            # Build in float32 first (safe for load_state_dict + LoRA), then cast.
            model = EnsembleAIDetector(siglip_name, dinov2_name, image_size, torch_dtype=torch.float32)

            # Apply LoRA to SigLIP
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none"
            )
            model.siglip = get_peft_model(model.siglip, lora_config)

            # Apply LoRA to DINOv2
            for name, module in model.dinov2.named_modules():
                if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                    module.qkv = LoRALinear(module.qkv, lora_rank, lora_alpha, lora_dropout)

            # Load trained weights (in fp32 on CPU), then cast to inference dtype + device.
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device=self.device, dtype=self.inference_dtype)
            model.eval()

            # Optional IPEX optimization for Intel XPU (Arc) — accelerates inference.
            # Disable with MONET_IPEX_OPTIMIZE=0 if it causes issues with your build.
            if self.device.type == "xpu" and os.environ.get("MONET_IPEX_OPTIMIZE", "1") == "1":
                try:
                    import intel_extension_for_pytorch as ipex
                    model = ipex.optimize(model)
                    print("  IPEX optimize: applied")
                except Exception as e:
                    print(f"  IPEX optimize: skipped ({e})")

            self.model = model

            # --- Processors ---
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_name)
            self.dinov2_transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            # --- Warmup: compile kernels / sanity-check accelerator; fall back to CPU ---
            # Use a batch matching the real inference shape so MPS/Metal graph
            # compilation is paid here, not on the first real request.
            try:
                _n = max(1, self._sample_count)
                _warm = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)) for _ in range(_n)]
                _sig = self.siglip_processor(images=_warm, return_tensors="pt")["pixel_values"]
                _sig = _sig.to(self.device, dtype=self.inference_dtype)
                _dino = torch.stack([self.dinov2_transform(_im) for _im in _warm])
                _dino = _dino.to(self.device, dtype=self.inference_dtype)
                with torch.inference_mode():
                    self.model(_sig, _dino)
                print(f"  Device: {self.device.type} ({str(self.inference_dtype).split('.')[-1]}) — warmup OK (batch={_n})")
            except Exception as e:
                if self.device.type != "cpu":
                    print(f"  ⚠️  {self.device.type} failed warmup ({e}); falling back to CPU.")
                    self.device = torch.device("cpu")
                    self.inference_dtype = torch.float32
                    self.model.to(device=self.device, dtype=self.inference_dtype)
                else:
                    print(f"  ⚠️  warmup error on CPU: {e}")

            elapsed = time.time() - start
            self.loaded = True
            print(f"  SigLIP+DINOv2 model loaded in {elapsed:.1f}s")

        except Exception as e:
            print(f"  ⚠️  SigLIP+DINOv2 model failed to load: {e}")
            print("  Detector will return neutral scores (0.0)")
            self.loaded = False

    def _preprocess(self, frame):
        """
        Convert BGR OpenCV frame to PIL Image for processors.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            PIL Image in RGB format
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def analyze_frame(self, frame):
        """
        Analyze a single frame for AI-generated content.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            tuple: (ai_probability, description)
        """
        if not self.loaded:
            return 0.0, "AI detector unavailable"

        try:
            pil_image = self._preprocess(frame)

            # SigLIP preprocessing
            siglip_inputs = self.siglip_processor(images=pil_image, return_tensors="pt")
            siglip_pixels = siglip_inputs["pixel_values"]
            
            # Cast inputs to match model parameter dtypes dynamically
            siglip_dtype = next(self.model.siglip.parameters()).dtype
            siglip_pixels = siglip_pixels.to(self.device, dtype=siglip_dtype)

            # DINOv2 preprocessing
            dinov2_pixels = self.dinov2_transform(pil_image).unsqueeze(0)
            dinov2_dtype = next(self.model.dinov2.parameters()).dtype
            dinov2_pixels = dinov2_pixels.to(self.device, dtype=dinov2_dtype)

            # Inference
            with torch.inference_mode():
                logits, _, _ = self.model(siglip_pixels, dinov2_pixels)

            ai_prob = float(torch.sigmoid(logits).item())

            desc = self._desc_for_prob(ai_prob)

            return ai_prob, desc

        except Exception as e:
            print(f"  AI detector inference error: {e}")
            return 0.0, f"Detector error: {str(e)[:50]}"

    @staticmethod
    def _desc_for_prob(ai_prob):
        if ai_prob >= 0.65:
            return f"{ai_prob:.0%} AI"
        elif ai_prob >= 0.46:
            return f"{ai_prob:.0%} possibly AI"
        else:
            return f"{ai_prob:.0%} likely authentic"

    def _score_frames_batched(self, frames, batch_size=None):
        """Score a list of BGR frames in batches (one forward pass per batch).

        Returns a list of float probabilities aligned with the input order.
        """
        if not self.loaded or not frames:
            return []

        bs = batch_size or self._default_batch_size
        siglip_dtype = next(self.model.siglip.parameters()).dtype
        dinov2_dtype = next(self.model.dinov2.parameters()).dtype

        probs = []
        with torch.inference_mode():
            for i in range(0, len(frames), bs):
                chunk = frames[i:i + bs]
                pil_images = [self._preprocess(f) for f in chunk]

                siglip_inputs = self.siglip_processor(images=pil_images, return_tensors="pt")
                siglip_pixels = siglip_inputs["pixel_values"].to(self.device, dtype=siglip_dtype)

                dinov2_pixels = torch.stack(
                    [self.dinov2_transform(p) for p in pil_images]
                ).to(self.device, dtype=dinov2_dtype)

                logits, _, _ = self.model(siglip_pixels, dinov2_pixels)
                probs.extend(torch.sigmoid(logits).float().cpu().tolist())

        return probs

    def analyze_frames(self, frames, batch_size=None):
        """
        Analyze multiple frames and return the maximum AI probability (batched).

        Args:
            frames: list of BGR numpy arrays
            batch_size: override the default batch size

        Returns:
            tuple: (max_ai_probability, description)
        """
        if not self.loaded:
            return 0.0, "AI detector unavailable"

        if not frames:
            return 0.0, "No frames"

        sample_count = min(self._sample_count, len(frames))
        key_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        key_indices = list(dict.fromkeys(int(i) for i in key_indices))
        sampled = [frames[i] for i in key_indices]

        scores = self._score_frames_batched(sampled, batch_size=batch_size)
        if not scores:
            return 0.0, "No detector scores"

        max_idx = int(np.argmax(scores))
        return scores[max_idx], self._desc_for_prob(scores[max_idx])

    def analyze_frame_stats(self, frames, batch_size=None):
        """
        Analyze multiple frames and return mean/max/std statistics (batched).

        Returns:
            tuple: (mean_probability, max_probability, std_probability, description)
        """
        if not self.loaded:
            return 0.0, 0.0, 0.0, "AI detector unavailable"

        if not frames:
            return 0.0, 0.0, 0.0, "No frames"

        sample_count = min(self._sample_count, len(frames))
        key_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        key_indices = list(dict.fromkeys(int(i) for i in key_indices))
        sampled = [frames[i] for i in key_indices]

        scores = self._score_frames_batched(sampled, batch_size=batch_size)
        if not scores:
            return 0.0, 0.0, 0.0, "No detector scores"

        max_idx = int(np.argmax(scores))
        return float(np.mean(scores)), float(np.max(scores)), float(np.std(scores)), self._desc_for_prob(scores[max_idx])
