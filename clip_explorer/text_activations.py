"""Read-only observation of the original CLIP text forward pass."""
from __future__ import annotations
import numpy as np
import torch
import streamlit as st
from . import model as clip_model


def capture_text_states(model: torch.nn.Module, tokens: torch.Tensor) -> dict:
    """Capture one prompt via temporary hooks; return detached CPU float32 arrays.

    The full context (including zero-filled positions after EOS) is forwarded.
    Hooks observe outputs and never replace them. The caller must hold the
    cached runtime lock to isolate observation from concurrent inference.
    All handles are removed even if encode_text raises.
    """
    if tokens.ndim!=2 or tokens.shape[0]!=1:
        raise ValueError('Inspect one tokenized prompt at a time.')
    stages={};handles=[]
    def save(name: str, value: torch.Tensor, sequence_first: bool=False):
        if sequence_first:value=value.permute(1,0,2)
        stages[name]=value[0].detach().float().cpu().numpy().copy()
    def token_hook(module,args,output):save('Token lookup',output.type(model.dtype))
    def first_block_input(module,args):save('Token + position',args[0],True)
    def block_hook(name):
        def observe(module,args,output):save(name,output,True)
        return observe
    def final_hook(module,args,output):save('Final layer norm',output.type(model.dtype))
    with torch.no_grad():
        try:
            handles.append(model.token_embedding.register_forward_hook(token_hook))
            handles.append(model.transformer.resblocks[0].register_forward_pre_hook(first_block_input))
            for index,block in enumerate(model.transformer.resblocks):
                handles.append(block.register_forward_hook(block_hook(f'Transformer block {index+1}')))
            handles.append(model.ln_final.register_forward_hook(final_hook))
            projected=model.encode_text(tokens)
            # Positional vectors are a separate learned input, not contextual states.
            position=model.positional_embedding.detach().type(model.dtype).float().cpu().numpy().copy()
            stages={'Token lookup':stages['Token lookup'],'Position embeddings':position,
                    **{name:value for name,value in stages.items() if name!='Token lookup'}}
            stages['Projected EOS']=projected.detach().float().cpu().numpy().copy()
            normalized=torch.nn.functional.normalize(projected.float(),dim=-1)
            stages['Normalized CLIP vector']=normalized.detach().cpu().numpy().copy()
        finally:
            for handle in handles:handle.remove()
    return {'stages':stages,'token_ids':tokens[0].detach().cpu().numpy().copy(),
            'eos_index':int(tokens[0].argmax()),'context_length':tokens.shape[1],
            'hidden_width':model.token_embedding.weight.shape[1],
            'output_width':model.text_projection.shape[1]}


@st.cache_data(show_spinner=False,max_entries=16)
def inspect_text_prompt(prompt: str) -> dict:
    """Inspect genuine local CLIP activations, without gradients or parameter changes."""
    if not prompt.strip():raise ValueError('Enter a nonempty text prompt.')
    import clip
    tokens=clip.tokenize([prompt],truncate=False)
    runtime=clip_model.load_clip_model()
    with runtime.lock:
        trace=capture_text_states(runtime.model,tokens.to(runtime.device))
    trace['prompt']=prompt
    trace['token_pieces']=[clip.clip._tokenizer.decoder[int(token)] for token in trace['token_ids']]
    return trace


def token_labels(trace: dict, show_padding: bool=False) -> list[str]:
    """Use positional labels, treating post-EOS zero IDs as padding positions."""
    count=trace['context_length'] if show_padding else trace['eos_index']+1
    return [f'{position}: '+(trace['token_pieces'][position] if position<=trace['eos_index'] else '[zero-filled padding]') for position in range(count)]


def contextual_stages(trace: dict) -> list[str]:
    """Ordered residual-stream states, excluding positions alone and pooled outputs."""
    return [name for name in trace['stages'] if name not in ['Position embeddings','Projected EOS','Normalized CLIP vector']]
