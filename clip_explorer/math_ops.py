"""Pure tensor operations. No checkpoint, Streamlit, or network required."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize nonzero finite vectors along the embedding dimension."""
    if embeddings.ndim < 2 or not torch.isfinite(embeddings).all():
        raise ValueError("Expected finite embeddings with at least two dimensions.")
    embeddings = embeddings.float()
    if (embeddings.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Zero embeddings have no defined cosine direction.")
    return F.normalize(embeddings, dim=-1)


def cosine_similarity(images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
    """Return an [images, descriptions] matrix of normalized dot products."""
    return normalize_embeddings(images) @ normalize_embeddings(texts).T


def compute_logits(images: torch.Tensor, texts: torch.Tensor, logit_scale: float | torch.Tensor) -> torch.Tensor:
    """Scale cosine scores by exp(logit_scale), the model's learned parameter."""
    scale = torch.as_tensor(logit_scale, device=images.device).float().exp()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Logit scale must yield a finite positive multiplier.")
    return scale * cosine_similarity(images, texts)


def compute_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Candidate-dependent softmax along descriptions; not calibrated confidence."""
    return logits.float().softmax(dim=-1)


def temperature_logits(similarities: torch.Tensor, temperature: float) -> torch.Tensor:
    """Teaching override: S / tau. Not the pretrained model's learned scale."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    return similarities / temperature


def directional_probabilities(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalized i2t and column-normalized t2i, both in image-row layout."""
    return logits.softmax(dim=1), logits.softmax(dim=0)


def contrastive_loss(logits: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric CE, supporting a permutation of captions and positive targets."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or logits.shape[0] == 0:
        raise ValueError("Contrastive batches must be nonempty and square.")
    n = logits.shape[0]
    targets = torch.arange(n, device=logits.device) if targets is None else targets.to(logits.device)
    if targets.shape != (n,) or not torch.equal(targets.sort().values, torch.arange(n, device=logits.device)):
        raise ValueError("Targets must be a permutation of caption indices.")
    inverse = targets.argsort()
    i2t = F.cross_entropy(logits.float(), targets)
    t2i = F.cross_entropy(logits.T.float(), inverse)
    return i2t, t2i, (i2t + t2i) / 2


def ensemble_embeddings(prompt_embeddings: torch.Tensor) -> torch.Tensor:
    """[classes, templates, features] → normalize(mean(normalize(prompts)))."""
    if prompt_embeddings.ndim != 3 or prompt_embeddings.shape[1] == 0:
        raise ValueError("Expected [classes, templates, features].")
    return normalize_embeddings(normalize_embeddings(prompt_embeddings).mean(dim=1))


def retrieval_ranking(gallery: torch.Tensor, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Descending cosine scores; ties retain gallery order."""
    if query.ndim != 2 or query.shape[0] != 1 or not 1 <= k <= len(gallery):
        raise ValueError("Supply one query and K between 1 and gallery size.")
    scores = cosine_similarity(gallery, query).flatten()
    indices = torch.argsort(scores, descending=True, stable=True)[:k]
    return indices, scores[indices]


def similarity_margins(correct: torch.Tensor, incorrect: torch.Tensor) -> torch.Tensor:
    """Positive means the correct caption outranked its paired distractor."""
    if correct.shape != incorrect.shape:
        raise ValueError("Correct and incorrect scores must have matching shapes.")
    return correct - incorrect


def format_prompts(classes: list[str], template: str) -> list[str]:
    """Validate one positional placeholder before formatting user prompts."""
    if template.count("{}") != 1 or "{" in template.replace("{}", "") or "}" in template.replace("{}", ""):
        raise ValueError('Use exactly one {} placeholder, e.g. "a photo of a {}".')
    if not classes or any(not name.strip() for name in classes):
        raise ValueError("Enter at least one nonempty class.")
    return [template.format(name) for name in classes]
