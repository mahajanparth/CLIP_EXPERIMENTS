"""Standalone, offline visual guide with real toy updates and real CLIP testing."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from html import escape
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from clip_explorer.config import ROOT,MODEL_NAME,SEED
from clip_explorer.data import manifest,open_asset,asset_embeddings,photo_for_label
from clip_explorer.model import encode_text,compute_logits,compute_probabilities
from clip_explorer.math_ops import cosine_similarity
from clip_explorer.training_demo import toy_training
from clip_explorer.process_visuals import pipeline_svg,loss_animation,image_uri,LABELS,PALETTE


def export_process_artifacts() -> None:
    target=ROOT/'artifacts'
    history=toy_training()
    losses=pd.DataFrame([{key:s[key] for key in ['iteration','loss_i2t','loss_t2i','loss','scale','temperature']} for s in history])
    losses.to_csv(target/'toy_training_losses.csv',index=False)
    training_svg=pipeline_svg(True)
    testing_svg=pipeline_svg(False,image_count=1,text_count=len(LABELS))
    (target/'clip_training_architecture.svg').write_text(training_svg)
    (target/'clip_testing_architecture.svg').write_text(testing_svg)
    gallery=manifest()['gallery']
    record=next(r for r in gallery if r['id']=='photo_cat') if any(r['id']=='photo_cat' for r in gallery) else next(r for r in gallery if r['label']=='cat')
    image=asset_embeddings([record])
    prompts=[f'a photo of a {label}' for label in LABELS]
    texts=encode_text(prompts)
    similarity=cosine_similarity(image,texts)[0]
    logits=compute_logits(image,texts)[0]
    probability=compute_probabilities(logits[None])[0]
    measured=pd.DataFrame({'Class':LABELS,'Prompt':prompts,'Cosine similarity':similarity.numpy(),'Scaled logit':logits.numpy(),'Probability':probability.numpy()})
    measured.to_csv(target/'process_testing_results.csv',index=False)
    figure=make_subplots(rows=1,cols=3,subplot_titles=['1 · Cosine similarity','2 · Learned-scale logits','3 · Candidate softmax'])
    for col,(name,values) in enumerate([('Cosine similarity',similarity),('Scaled logit',logits),('Probability',probability)],1):
        figure.add_trace(go.Bar(x=LABELS,y=values.numpy(),marker_color=PALETTE,text=[f'{v:.3f}' for v in values.tolist()],textposition='auto',name=name,showlegend=False),row=1,col=col)
    figure.update_yaxes(range=[0,1],row=1,col=3)
    figure.update_layout(height=440,title=f'Real frozen {MODEL_NAME} testing · {record["id"]}',font=dict(size=15),template='plotly_white')
    training_html=loss_animation(history).to_html(full_html=False,include_plotlyjs=True)
    testing_html=figure.to_html(full_html=False,include_plotlyjs=False)
    cards=[]
    for label in LABELS:
        asset=photo_for_label(label)
        cards.append(f'<article class="pair"><img src="{image_uri(open_asset(asset))}" alt="{label} photograph"><strong>{escape(asset["caption"])}</strong><small>Paired identity; toy input is a one-hot feature vector.</small><small>{escape(asset.get("author",""))} · <a href="{escape(asset["source"],quote=True)}">{escape(asset["license"])}</a></small></article>')
    prediction=LABELS[int(similarity.argmax())]
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP · Visual training and testing cycles</title>
<style>body{margin:0;background:#f5f7fa;color:#172b42;font:17px/1.6 Arial,sans-serif}main{max-width:1200px;margin:auto;padding:40px 28px}h1{font-size:42px;letter-spacing:-1.5px;margin-bottom:8px}h2{font-size:28px}h3{font-size:21px}.eyebrow{font-size:13px;letter-spacing:2px;color:#137c75;font-weight:bold}.note{border-left:5px solid #137c75;background:#e6f1ef;padding:18px 22px;border-radius:8px;margin:24px 0}.toy{border-color:#d08036;background:#fff3df}.panel{background:#fff;padding:28px;border:1px solid #dbe3e9;border-radius:18px;margin:24px 0}.pair-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.pair img{width:100%;border-radius:12px}.pair strong,.pair small{display:block}.pair small{color:#617888;font-size:13px}.diagram{overflow:auto}.diagram svg{max-width:100%;height:auto}.tabs{display:flex;gap:10px;margin:24px 0}button{font:inherit;border:1px solid #137c75;border-radius:9px;padding:12px 20px;cursor:pointer;background:white;color:#137c75}button.active{background:#137c75;color:white}.pane[hidden]{display:none}.test-image{max-width:320px;border-radius:14px}.metrics{display:flex;gap:24px;flex-wrap:wrap}.metric{padding:20px;background:#f5f7fa;border-radius:12px}.metric b{display:block;font-size:28px}footer{color:#617888;font-size:14px;margin:32px 0}@media(max-width:700px){.pair-grid{grid-template-columns:repeat(2,1fr)}h1{font-size:32px}.tabs{flex-wrap:wrap}}</style></head><body><main>
<div class="eyebrow">CLIP EXPLORER · VISUAL GUIDE</div><h1>From paired inputs to predictions</h1><p>Two encoders. One shared space. A training loop with gradients, and a testing pass with frozen parameters.</p>
<nav class="tabs"><button class="active" onclick="openPane('overview',this)">Architecture overview</button><button onclick="openPane('training',this)">Training cycle</button><button onclick="openPane('testing',this)">Testing cycle</button></nav>
'''
    document+=f'''<section class="pane" id="overview"><div class="panel"><h2>Training: learn alignment from matching pairs</h2><div class="diagram">{training_svg}</div><p>Every image is compared with every caption. Image-to-text and text-to-image cross-entropies supervise the known pair indices; their average sends gradients to both encoders and the learned scale. The optimizer updates parameters, then training repeats with another batch.</p></div><div class="panel"><h2>Testing: use words as classifier weights</h2><div class="diagram">{testing_svg}</div><p>Encode a new image and class descriptions with frozen encoders. Normalize, score, apply the learned scale and candidate softmax, then select a candidate. Labels are optional evaluation information; they are not needed to construct the zero-shot classifier.</p></div></section>'''
    document+=f'''<section class="pane" id="training" hidden><div class="note toy"><strong>ACTUAL TOY OPTIMIZATION · NOT PRETRAINED CLIP TRAINING</strong><br>Two independent 4→8 linear maps and a learned scale are optimized by SGD on a single batch of synthetic one-hot inputs. Real photographs illustrate pair identities. Seed {SEED}; learning rate 0.08; initial temperature 0.2; 40 updates. The same batch is reused; no claim of generalization is made.</div><div class="pair-grid">{"".join(cards)}</div><div class="panel"><h2>Forward → symmetric loss → backward → update</h2><div class="diagram">{pipeline_svg(True,toy=True)}</div><p>L = exp(α) × normalize(V) × normalize(T)ᵀ<br>Loss = ½ [CE(L, pair indices) + CE(Lᵀ, pair indices)]</p><div class="metrics"><div class="metric">Initial symmetric loss<b>{history[0]["loss"]:.4f}</b></div><div class="metric">After 40 toy updates<b>{history[-1]["loss"]:.4f}</b></div></div><h3>Press Play to watch the measured matrix and losses evolve</h3>{training_html}<p>The gold diagonal marks matching pairs. Each frame comes from a real toy gradient update. Both directional losses are plotted alongside their symmetric average.</p></div></section>'''
    document+=f'''<section class="pane" id="testing" hidden><div class="note"><strong>REAL PRETRAINED INFERENCE · FROZEN {MODEL_NAME}</strong><br>Measured on a local prepared image with the actual checkpoint scale. No backward pass, optimizer update, or task-specific classifier training.</div><div class="panel"><div class="diagram">{testing_svg}</div><img class="test-image" src="{image_uri(open_asset(record))}" alt="Test image: {escape(record["label"])}"><div class="metrics"><div class="metric">Image<b>{escape(record["id"])}</b></div><div class="metric">Top candidate<b>{escape(prediction)}</b></div></div>{testing_html}<p>Candidate descriptions: {escape("; ".join(prompts))}.</p><p>Softmax probabilities depend on these candidates and are not calibrated confidence. Removing the true class can still leave an incorrect class with high probability. The same positive scale preserves the ranking of cosine, logits, and probabilities.</p></div></section>'''
    document+='''<footer>All figures, diagrams, artwork, and Plotly code are embedded for offline use. Interactive stage-by-stage inspection, actual image preprocessing/tokenization, per-pair losses, gradient norms, and labeled evaluation are available in the local Streamlit “CLIP Process Walkthrough” page. Research source: Radford et al., Learning Transferable Visual Models From Natural Language Supervision (ICML 2021), Figure 3 and §3.1.</footer></main>
<script>function openPane(id,button){document.querySelectorAll('.pane').forEach(p=>p.hidden=p.id!==id);document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active',b===button));setTimeout(()=>document.querySelectorAll('#'+id+' .js-plotly-plot').forEach(p=>Plotly.Plots.resize(p)),80);}</script></body></html>'''
    (target/'clip_process.html').write_text(document)
    print('Exported offline CLIP process guide, SVG architectures, actual toy losses, and measured testing scores.')

if __name__=='__main__':export_process_artifacts()
