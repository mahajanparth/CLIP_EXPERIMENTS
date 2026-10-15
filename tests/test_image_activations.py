"""Observe a small genuine CLIP architecture without downloading weights."""
import numpy as np
import pytest
import torch
import clip
from clip_explorer.image_activations import capture_image_states,patch_values,stage_labels


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021)
        return clip.model.CLIP(embed_dim=8,image_resolution=32,vision_layers=2,vision_width=64,
            vision_patch_size=16,context_length=6,vocab_size=32,transformer_width=16,
            transformer_heads=2,transformer_layers=2).eval()


def hooks(m):return [(len(n._forward_hooks),len(n._forward_pre_hooks)) for n in m.modules()]


def test_image_observation_matches_forward_and_patch_order():
    m=model();image=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    before=hooks(m);versions=[p._version for p in m.parameters()]
    with torch.no_grad():
        expected=m.encode_image(image)
        patches=m.visual.conv1(image).flatten(2).permute(0,2,1)[0].numpy()
    trace=capture_image_states(m,image)
    assert trace['grid']==[2,2]
    assert trace['stages']['Vision block 2'].shape==(5,64)
    np.testing.assert_allclose(trace['stages']['Projected class vector'],expected.numpy(),atol=1e-6)
    np.testing.assert_allclose(trace['stages']['Patch projection'],patches,atol=1e-6)
    np.testing.assert_allclose(patch_values(trace,'Patch projection',3),patches[:,3].reshape(2,2))
    np.testing.assert_allclose(patch_values(trace,'Vision block 2',3),trace['stages']['Vision block 2'][1:,3].reshape(2,2))
    np.testing.assert_allclose(np.linalg.norm(trace['stages']['Normalized CLIP vector']),1,atol=1e-6)
    assert stage_labels(trace,'Vision block 2')==['Class token','Patch (0,0)','Patch (0,1)','Patch (1,0)','Patch (1,1)']
    assert hooks(m)==before
    assert [p._version for p in m.parameters()]==versions
    assert all(p.grad is None for p in m.parameters())
    with pytest.raises(ValueError,match='pooled'):patch_values(trace,'Normalized CLIP vector')


def test_image_hooks_removed_on_forward_exception(monkeypatch):
    m=model();before=hooks(m)
    def fail(image):raise RuntimeError('intentional')
    monkeypatch.setattr(m,'encode_image',fail)
    with pytest.raises(RuntimeError,match='intentional'):capture_image_states(m,torch.zeros(1,3,32,32))
    assert hooks(m)==before
