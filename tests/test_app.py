from streamlit.testing.v1 import AppTest
from clip_explorer.config import ROOT


def test_synthetic_lab_runs_without_loading_model(monkeypatch):
    import clip_explorer.model as model
    def reject_model(): raise AssertionError('Synthetic demo must not load weights')
    monkeypatch.setattr(model,'load_clip_model',reject_model)
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==3
    assert app.metric[2].value=='0.0003'
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.metric[2].value=='0.0003'
    app.slider[1].set_value(0.0).run()
    assert not app.exception
    assert app.metric[2].value=='1.3863'
