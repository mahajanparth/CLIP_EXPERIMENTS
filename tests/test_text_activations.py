"""Forward-observation tests on a small random CLIP, without checkpoint downloads."""
import numpy as np
import pytest
import torch
import clip
from clip_explorer.text_activations import capture_text_states,token_labels


def small_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021)
        model=clip.model.CLIP(embed_dim=8,image_resolution=32,vision_layers=1,vision_width=64,
            vision_patch_size=16,context_length=6,vocab_size=32,transformer_width=16,
            transformer_heads=2,transformer_layers=2).eval()
    return model


def hook_counts(model):
    return [(len(module._forward_hooks),len(module._forward_pre_hooks)) for module in model.modules()]


def test_observation_matches_forward_and_actual_input_embeddings():
    model=small_model();tokens=torch.tensor([[30,2,3,31,0,0]])
    before=hook_counts(model)
    versions=[p._version for p in model.parameters()]
    with torch.no_grad():expected=model.encode_text(tokens)
    trace=capture_text_states(model,tokens)
    assert trace['eos_index']==3
    assert trace['stages']['Transformer block 1'].shape==(6,16)
    assert trace['stages']['Transformer block 2'].shape==(6,16)
    assert trace['stages']['Projected EOS'].shape==(1,8)
    np.testing.assert_allclose(trace['stages']['Projected EOS'],expected.numpy(),atol=1e-6)
    np.testing.assert_allclose(trace['stages']['Token lookup'],model.token_embedding(tokens).detach().numpy()[0],atol=1e-6)
    np.testing.assert_allclose(trace['stages']['Token + position'],trace['stages']['Token lookup']+trace['stages']['Position embeddings'],atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(trace['stages']['Normalized CLIP vector'],axis=1),1,atol=1e-6)
    assert hook_counts(model)==before
    assert [p._version for p in model.parameters()]==versions
    assert all(p.grad is None for p in model.parameters())


def test_hooks_are_removed_even_when_forward_fails(monkeypatch):
    model=small_model();before=hook_counts(model)
    def fail(tokens):raise RuntimeError('Intentional forward error')
    monkeypatch.setattr(model,'encode_text',fail)
    with pytest.raises(RuntimeError,match='Intentional'):
        capture_text_states(model,torch.tensor([[30,2,3,31,0,0]]))
    assert hook_counts(model)==before


def test_causal_prefix_stays_unchanged_but_eos_responds_to_word_change():
    model=small_model()
    first=capture_text_states(model,torch.tensor([[30,2,3,31,0,0]]))
    second=capture_text_states(model,torch.tensor([[30,2,4,31,0,0]]))
    for stage in ['Token + position','Transformer block 1','Transformer block 2','Final layer norm']:
        np.testing.assert_allclose(first['stages'][stage][:2],second['stages'][stage][:2],atol=1e-6)
    assert not np.allclose(first['stages']['Projected EOS'],second['stages']['Projected EOS'])


def test_padding_is_not_labeled_as_a_real_word():
    trace={'context_length':6,'eos_index':3,'token_pieces':['SOT','a','cat','EOS','!','!']}
    assert len(token_labels(trace))==4
    assert token_labels(trace,True)[4]=='4: [zero-filled padding]'
