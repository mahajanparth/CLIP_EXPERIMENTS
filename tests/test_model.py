"""Opt-in integration tests: RUN_CLIP_SMOKE=1 python -m pytest -m model."""
import os
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.model,pytest.mark.skipif(os.environ.get('RUN_CLIP_SMOKE')!='1',reason='Opt-in; requires locally prepared weights.')]


def test_real_pretrained_inference():
    from clip_explorer.model import load_clip_model, encode_image, encode_text, compute_logits
    from clip_explorer.math_ops import cosine_similarity
    runtime = load_clip_model()
    assert not runtime.model.training
    assert all(not p.requires_grad for p in runtime.model.parameters())
    image = encode_image(Image.new('RGB',(224,224),'red'))
    text = encode_text(['a red square','a blue square'])
    assert image.shape==(1,512)
    assert text.shape==(2,512)
    assert torch.allclose(image.norm(dim=-1),torch.ones(1),atol=1e-5)
    scores = cosine_similarity(image,text)
    logits = compute_logits(image,text)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits,scores*runtime.model.logit_scale.detach().cpu().float().exp(),atol=1e-5)


def render_zero():
    from clip_explorer.labs import zero_shot
    zero_shot()


def render_prompts():
    from clip_explorer.labs import prompts
    prompts()


def render_retrieval():
    from clip_explorer.labs import retrieval
    retrieval()


def render_failure():
    from clip_explorer.labs import failures
    failures()


def render_embeddings():
    from clip_explorer.embedding_lab import render
    render()


@pytest.mark.parametrize('render',[render_zero,render_prompts,render_retrieval,render_failure,render_embeddings])
def test_real_labs_render(render):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(render).run(timeout=60)
    assert not app.exception


def test_embedding_lab_interactions():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(render_embeddings).run(timeout=60)
    assert not app.exception
    app.select_slider[0].set_value(2).run(timeout=60)
    assert not app.exception
    app.text_input[0].set_value('a bicycle').run(timeout=60)
    assert not app.exception
    app.radio[0].set_value('Shared prompt template').run(timeout=60)
    assert not app.exception
    app.selectbox[0].set_value('t-SNE').run(timeout=60)
    assert not app.exception
    app.multiselect[0].set_value([]).run(timeout=60)
    assert not app.exception
    assert any('Select at least two images' in element.value for element in app.info)


def test_testing_walkthrough_all_stages_are_frozen():
    from streamlit.testing.v1 import AppTest
    def render_walkthrough():
        from clip_explorer.process_lab import render
        render()
    from clip_explorer.process_visuals import TEST_STAGES
    from clip_explorer.model import load_clip_model
    runtime=load_clip_model()
    parameter_versions=[parameter._version for parameter in runtime.model.parameters()]
    app=AppTest.from_function(render_walkthrough).run(timeout=60)
    app.radio[0].set_value('Testing cycle').run(timeout=60)
    assert not app.exception
    for stage in TEST_STAGES[1:]:
        app.select_slider[0].set_value(stage).run(timeout=60)
        assert not app.exception
    assert all(not parameter.requires_grad for parameter in runtime.model.parameters())
    assert [parameter._version for parameter in runtime.model.parameters()]==parameter_versions


def test_real_text_activation_trace_and_inspector():
    import numpy as np
    from clip_explorer.text_activations import inspect_text_prompt
    from clip_explorer.model import encode_text,load_clip_model
    runtime=load_clip_model()
    before=[len(module._forward_hooks)+len(module._forward_pre_hooks) for module in runtime.model.modules()]
    trace=inspect_text_prompt('a photo of a golden retriever')
    np.testing.assert_allclose(trace['stages']['Normalized CLIP vector'],encode_text(['a photo of a golden retriever']).numpy(),atol=1e-6)
    assert trace['stages']['Transformer block 12'].shape==(77,512)
    assert [len(module._forward_hooks)+len(module._forward_pre_hooks) for module in runtime.model.modules()]==before
    def render_inspector():
        from clip_explorer.activation_lab import render
        render()
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_function(render_inspector).run(timeout=60)
    assert not app.exception
    app.text_input[1].set_value('a photo of a cat').run(timeout=60)
    assert not app.exception
    app.selectbox[0].set_value('Normalized CLIP vector').run(timeout=60)
    assert not app.exception
    app.checkbox[0].check().run(timeout=60)
    assert not app.exception


def test_real_image_activation_trace_and_inspector():
    import numpy as np
    from clip_explorer.data import photo_for_label,open_asset
    from clip_explorer.image_activations import inspect_image_payload
    from clip_explorer.model import image_bytes,encode_image
    image=open_asset(photo_for_label('cat'))
    trace=inspect_image_payload(image_bytes(image))
    np.testing.assert_allclose(trace['stages']['Normalized CLIP vector'],encode_image(image).numpy(),atol=1e-6)
    assert trace['stages']['Vision block 12'].shape==(50,768)
    assert trace['stages']['Patch projection'].shape==(49,768)
    from streamlit.testing.v1 import AppTest
    def render_inspector():
        from clip_explorer.image_activation_lab import render
        render()
    app=AppTest.from_function(render_inspector).run(timeout=90)
    assert not app.exception
    stage=next(e for e in app.selectbox if e.label=='Image activation stage')
    stage.set_value('Vision block 12').run(timeout=60)
    assert not app.exception
    stage=next(e for e in app.selectbox if e.label=='Image activation stage')
    stage.set_value('Normalized CLIP vector').run(timeout=60)
    assert not app.exception
    app.text_input[0].set_value('a photo of a dog').run(timeout=60)
    assert not app.exception
