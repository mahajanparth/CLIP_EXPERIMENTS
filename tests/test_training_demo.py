"""Verify the displayed training cycle is measured toy optimization."""
import math
import numpy as np
import pytest
import torch
from clip_explorer.training_demo import toy_training
from clip_explorer.math_ops import contrastive_loss


def test_recorded_losses_probabilities_and_gradients_are_real():
    history=toy_training(steps=3)
    for snapshot in history:
        logits=torch.tensor(snapshot['logits'])
        li,lt,total=contrastive_loss(logits)
        assert snapshot['loss']==pytest.approx(total.item())
        assert snapshot['loss_i2t']==pytest.approx(li.item())
        assert snapshot['loss_t2i']==pytest.approx(lt.item())
        assert snapshot['loss_i2t']==pytest.approx(snapshot['row_nll'].mean())
        assert snapshot['loss_t2i']==pytest.approx(snapshot['column_nll'].mean())
        np.testing.assert_allclose(snapshot['image_to_text'].sum(1),1,atol=1e-6)
        np.testing.assert_allclose(snapshot['text_to_image'].sum(0),1,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(snapshot['images'],axis=1),1,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(snapshot['texts'],axis=1),1,atol=1e-6)
        assert all(np.isfinite(snapshot['gradient_norms']))
    assert not np.array_equal(history[0]['image_weights'],history[1]['image_weights'])
    assert not np.array_equal(history[0]['text_weights'],history[1]['text_weights'])
    assert history[0]['logit_scale']!=history[1]['logit_scale']
    assert history[1]['loss']<history[0]['loss']


def test_toy_is_deterministic_and_preserves_rng():
    before=torch.random.get_rng_state().clone()
    first=toy_training(2)
    second=toy_training(2)
    assert torch.equal(before,torch.random.get_rng_state())
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a['similarity'],b['similarity'])
        assert a['loss']==b['loss']


def test_toy_enables_its_own_gradients_only():
    with torch.no_grad():
        history=toy_training(1)
        assert not torch.is_grad_enabled()
    assert history[1]['loss']<history[0]['loss']


@pytest.mark.parametrize('rate',[.01,.03,.08,.2])
def test_full_animation_configuration_is_finite(rate):
    for temperature in [.1,.2,.5,1.]:
        history=toy_training(40,rate,temperature)
        assert np.isfinite([s['loss'] for s in history]).all()
        assert history[-1]['loss']<history[0]['loss']


def render_walkthrough():
    from clip_explorer.process_lab import render
    render()


def test_architecture_and_training_stages_need_no_pretrained_model(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import clip_explorer.process_lab as lab
    def reject_model():raise AssertionError('Toy must never access pretrained weights')
    monkeypatch.setattr(lab.clip_model,'load_clip_model',reject_model)
    app=AppTest.from_function(render_walkthrough).run(timeout=30)
    assert not app.exception
    app.radio[0].set_value('Training cycle').run(timeout=30)
    assert not app.exception
    from clip_explorer.process_visuals import TRAIN_STAGES
    for stage in TRAIN_STAGES[1:]:
        app.select_slider[2].set_value(stage).run(timeout=30)
        assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.select_slider[2].value==TRAIN_STAGES[-2]
