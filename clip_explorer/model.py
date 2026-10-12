"""Cached, offline-first inference with the original OpenAI CLIP implementation."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import threading
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from .config import CHECKPOINT, SEED
from .math_ops import normalize_embeddings, cosine_similarity, compute_probabilities
from .math_ops import compute_logits as scaled_logits


class ModelUnavailable(RuntimeError):
    """Actionable local model setup error."""


@dataclass
class ClipRuntime:
    model: torch.nn.Module
    preprocess: object
    device: str
    lock: threading.RLock


@st.cache_resource(show_spinner="Loading frozen OpenAI CLIP · ViT-B/32…")
def load_clip_model() -> ClipRuntime:
    """Load once per process, using local weights; never download in the app."""
    if not CHECKPOINT.is_file():
        raise ModelUnavailable("Run python scripts/prepare.py --download-model before class.")
    import clip
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    model, preprocess = clip.load(str(CHECKPOINT), device=device, jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return ClipRuntime(model, preprocess, device, threading.RLock())


def image_bytes(image: Image.Image) -> bytes:
    """Canonical, orientation-corrected RGB PNG for content-based caching."""
    buffer = BytesIO()
    ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=1024)
def _encode_images(payloads: tuple[bytes, ...]) -> np.ndarray:
    runtime = load_clip_model()
    encoded = []
    with runtime.lock, torch.no_grad():
        for start in range(0, len(payloads), 16):
            batch = torch.stack([runtime.preprocess(Image.open(BytesIO(p)).convert("RGB")) for p in payloads[start:start + 16]]).to(runtime.device)
            encoded.append(normalize_embeddings(runtime.model.encode_image(batch)).cpu())
    return torch.cat(encoded).numpy()


def encode_images(images: list[Image.Image]) -> torch.Tensor:
    """Return cached normalized CPU float32 embeddings in input order."""
    if not images:
        raise ValueError("Supply at least one image.")
    return torch.from_numpy(_encode_images(tuple(image_bytes(image) for image in images)).copy())


def encode_image(image: Image.Image) -> torch.Tensor:
    """Encode one image, with shape [1, features]."""
    return encode_images([image])


@st.cache_data(show_spinner=False, max_entries=512)
def _encode_text(texts: tuple[str, ...]) -> np.ndarray:
    import clip
    runtime = load_clip_model()
    encoded = []
    with runtime.lock, torch.no_grad():
        for start in range(0, len(texts), 128):
            # Reject overlong text explicitly rather than silently truncate.
            tokens = clip.tokenize(texts[start:start + 128], truncate=False).to(runtime.device)
            encoded.append(normalize_embeddings(runtime.model.encode_text(tokens)).cpu())
    return torch.cat(encoded).numpy()


def encode_text(texts: list[str]) -> torch.Tensor:
    """Encode user descriptions, preserving the original CLIP tokenizer."""
    if not texts or any(not t.strip() for t in texts):
        raise ValueError("Descriptions cannot be empty.")
    return torch.from_numpy(_encode_text(tuple(texts)).copy())


def compute_logits(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """Use the checkpoint's actual learned logit_scale (never a fixed 100)."""
    runtime = load_clip_model()
    return scaled_logits(image_embeddings, text_embeddings, runtime.model.logit_scale.detach().cpu())
