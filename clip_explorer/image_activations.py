"""Observe ViT image activations without altering the original CLIP forward pass."""
from __future__ import annotations
from io import BytesIO
import numpy as np
import torch
import streamlit as st
from PIL import Image
from . import model as clip_model


def capture_image_states(model: torch.nn.Module, image: torch.Tensor) -> dict:
    """Caller holds the runtime lock. Temporary hooks always detach and clean up."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('Inspect one image with shape [1,3,H,W] at a time.')
    visual = model.visual
    if not hasattr(visual, 'transformer'):
        raise ValueError('This inspector requires a CLIP vision Transformer.')
    stages = {}; handles = []; grid = []
    def save(name, value, sequence_first=False):
        if sequence_first: value = value.permute(1,0,2)
        stages[name] = value[0].detach().float().cpu().numpy().copy()
    def patches(module,args,output):
        grid.extend(output.shape[-2:])
        save('Patch projection', output.flatten(2).permute(0,2,1))
    def before_norm(module,args): save('Class + patches + position',args[0])
    def pre_norm(module,args,output): save('Pre-transformer layer norm',output)
    def block_hook(name):
        def observe(module,args,output): save(name,output,True)
        return observe
    def post_norm(module,args,output): save('Final class layer norm',output.unsqueeze(1))
    with torch.no_grad():
        try:
            handles.append(visual.conv1.register_forward_hook(patches))
            handles.append(visual.ln_pre.register_forward_pre_hook(before_norm))
            handles.append(visual.ln_pre.register_forward_hook(pre_norm))
            for i,block in enumerate(visual.transformer.resblocks):
                handles.append(block.register_forward_hook(block_hook(f'Vision block {i+1}')))
            handles.append(visual.ln_post.register_forward_hook(post_norm))
            projected=model.encode_image(image)
            stages['Projected class vector']=projected.detach().float().cpu().numpy().copy()
            stages['Normalized CLIP vector']=torch.nn.functional.normalize(projected.float(),dim=-1).cpu().numpy().copy()
        finally:
            for handle in handles: handle.remove()
    return dict(stages=stages,grid=grid,patch_size=list(visual.conv1.kernel_size),
                input_shape=list(image.shape),hidden_width=visual.conv1.out_channels,
                output_width=projected.shape[-1])


@st.cache_data(show_spinner=False,max_entries=8)
def inspect_image_payload(payload: bytes) -> dict:
    runtime=clip_model.load_clip_model()
    with Image.open(BytesIO(payload)) as image:
        tensor=runtime.preprocess(image.convert('RGB')).unsqueeze(0).to(runtime.device)
    with runtime.lock:
        trace=capture_image_states(runtime.model,tensor)
    trace['preprocessed']=tensor[0].detach().float().cpu().numpy().copy()
    return trace


def stage_labels(trace: dict, stage: str) -> list[str]:
    rows=trace['stages'][stage].shape[0]
    if rows==1: return ['Class → whole-image vector']
    h,w=trace['grid']
    patches=[f'Patch ({r},{c})' for r in range(h) for c in range(w)]
    return patches if rows==h*w else ['Class token']+patches


def patch_values(trace: dict, stage: str, feature: int | None=None) -> np.ndarray:
    """Map residual patch rows to the actual row-major spatial grid."""
    values=trace['stages'][stage]
    h,w=trace['grid']
    if values.shape[0]==h*w+1: values=values[1:]
    if values.shape[0]!=h*w: raise ValueError('This stage contains only a pooled class vector, not spatial patches.')
    scalar=np.sqrt(np.mean(values**2,axis=1)) if feature is None else values[:,feature]
    return scalar.reshape(h,w)


def preprocessed_image(trace: dict) -> Image.Image:
    mean=np.array([.48145466,.4578275,.40821073])[:,None,None]
    std=np.array([.26862954,.26130258,.27577711])[:,None,None]
    rgb=np.clip(trace['preprocessed']*std+mean,0,1)
    return Image.fromarray(np.rint(rgb.transpose(1,2,0)*255).astype(np.uint8))
