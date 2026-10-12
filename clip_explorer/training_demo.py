"""Actual gradient updates on two tiny synthetic linear encoders, never CLIP."""
from __future__ import annotations
import math
import numpy as np
import torch
from .config import SEED
from .math_ops import normalize_embeddings, contrastive_loss, directional_probabilities


def toy_training(steps: int = 40, learning_rate: float = .08, temperature: float = .2) -> list[dict]:
    """Record forward pass, both CE losses, and gradients before every SGD update.

    Inputs are four one-hot feature vectors per modality. Independent 4→8
    linear maps and a scalar logit scale are optimized on this single batch.
    Artwork is an illustration of the paired inputs, not the model's input.
    This is a mechanics demonstration, not CLIP training or an evaluation.
    """
    if not 0<=steps<=100 or not math.isfinite(learning_rate) or not 0<learning_rate<=1 or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid toy optimizer configuration.')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    image_weights = torch.nn.Parameter(torch.randn(4,8,generator=generator)*.35)
    text_weights = torch.nn.Parameter(torch.randn(4,8,generator=generator)*.35)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1/temperature)))
    optimizer = torch.optim.SGD([image_weights,text_weights,logit_scale],lr=learning_rate)
    features = torch.eye(4)
    snapshots = []
    # Explicitly enable gradients for this isolated toy, even if a caller is
    # inside a no_grad context. The pretrained model is never referenced.
    with torch.enable_grad():
        for iteration in range(steps+1):
            optimizer.zero_grad(set_to_none=True)
            raw_images,raw_texts = features@image_weights, features@text_weights
            images,texts = normalize_embeddings(raw_images),normalize_embeddings(raw_texts)
            similarity = images@texts.T
            logits = logit_scale.exp()*similarity
            li,lt,total = contrastive_loss(logits)
            row_p,column_p = directional_probabilities(logits)
            row_nll = -logits.log_softmax(1).diag()
            column_nll = -logits.log_softmax(0).diag()
            total.backward()
            def array(tensor): return tensor.detach().numpy().copy()
            snapshots.append({'iteration':iteration,'images':array(images),'texts':array(texts),
                'raw_images':array(raw_images),'raw_texts':array(raw_texts),
                'similarity':array(similarity),'logits':array(logits),
                'image_to_text':array(row_p),'text_to_image':array(column_p),
                'row_nll':array(row_nll),'column_nll':array(column_nll),
                'loss_i2t':li.item(),'loss_t2i':lt.item(),'loss':total.item(),
                'scale':logit_scale.exp().item(),'temperature':logit_scale.exp().reciprocal().item(),
                'gradient_norms':[image_weights.grad.norm().item(),text_weights.grad.norm().item(),logit_scale.grad.abs().item()],
                'image_weights':array(image_weights),'text_weights':array(text_weights),'logit_scale':logit_scale.item()})
            if iteration<steps: optimizer.step()
    return snapshots
