"""One-time checkpoint download and reproducible, offline presentation artifacts."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from clip_explorer.config import ROOT, MODEL_NAME, MODEL_DIR, MODEL_REVISION, CHECKPOINT, SEED, TEMPLATES
from scripts.generate_assets import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true', help='Allow one-time download of official pretrained weights.')
    parser.add_argument('--assets-only', action='store_true', help='Generate deterministic artwork without a checkpoint.')
    args = parser.parse_args()
    data = generate()
    if args.assets_only:
        print('Assets ready. No inference or downloads performed.')
        return
    if args.download_model:
        import clip
        MODEL_DIR.mkdir(parents=True,exist_ok=True)
        # Upstream loader verifies the download checksum.
        model, _ = clip.load(MODEL_NAME, device='cpu', download_root=str(MODEL_DIR))
        del model
    if not CHECKPOINT.exists():
        parser.error('Checkpoint missing. Run with --download-model once, while online.')
    from clip_explorer.model import encode_images, encode_text, load_clip_model, compute_logits
    from clip_explorer.data import open_asset
    from clip_explorer.math_ops import cosine_similarity, ensemble_embeddings
    records = data['gallery'] + data['probes']
    vectors = []
    for start in range(0,len(records),16):
        chunk = records[start:start+16]
        vectors.extend(encode_images([open_asset(r) for r in chunk]).numpy())
        print(f'Encoded {min(start+16,len(records))}/{len(records)} images',flush=True)
    with CHECKPOINT.open('rb') as handle:
        checksum = hashlib.file_digest(handle,'sha256').hexdigest()
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    metadata = {'model': MODEL_NAME, 'model_revision': MODEL_REVISION, 'checkpoint_sha256':checksum, 'seed':SEED, 'device':load_clip_model().device, 'python':platform.python_version(), 'prepared_at_utc':datetime.now(timezone.utc).isoformat(), 'learned_logit_scale':load_clip_model().model.logit_scale.detach().float().cpu().item(), 'assets':{r['id']:hashlib.sha256((ROOT/'assets'/r['file']).read_bytes()).hexdigest() for r in records}, 'versions':{p:importlib.metadata.version(p) for p in ['torch','torchvision','clip','streamlit','numpy','Pillow','plotly','pandas']}}
    np.savez_compressed(artifacts/'image_embeddings.npz', **{r['id']:v for r,v in zip(records,vectors)})
    (artifacts/'embedding_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    offset = len(data['gallery'])
    probe_vectors = torch.tensor(np.stack(vectors[offset:]))
    probes = data['probes']
    correct_text = encode_text([r['correct'] for r in probes])
    wrong_text = encode_text([r['incorrect'] for r in probes])
    correct = (probe_vectors * correct_text).sum(dim=-1)
    wrong = (probe_vectors * wrong_text).sum(dim=-1)
    rows = [{**{k:r[k] for k in ['id','experiment','correct','incorrect']},'correct_cosine':float(c),'incorrect_cosine':float(w),'margin':float(c-w),'correct_ranked_first':bool(c>w)} for r,c,w in zip(probes,correct,wrong)]
    results = pd.DataFrame(rows)
    results.to_csv(artifacts/'failure_results.csv',index=False)
    summary = results.groupby('experiment').agg(examples=('id','count'),accuracy=('correct_ranked_first','mean'),mean_margin=('margin','mean'))
    summary.to_csv(artifacts/'failure_summary.csv')
    # Fixed zero-shot and prompt strategy reports on the same real labeled photographs.
    labels = ['dog','cat','car','bicycle']
    sample = [r for r in data['gallery'] if r['label'] in labels and r['medium']=='photograph']
    image_vectors = torch.tensor(np.stack([vectors[records.index(r)] for r in sample]))
    strategies = {
        'Raw class names':encode_text(labels),
        'Single contextual prompt':encode_text([f'a photo of a {c}' for c in labels]),
        'Template ensemble':ensemble_embeddings(encode_text([t.format(c) for c in labels for t in TEMPLATES]).reshape(len(labels),len(TEMPLATES),-1)),
    }
    prompt_rows = []
    for name, text_vectors in strategies.items():
        cosine = cosine_similarity(image_vectors,text_vectors)
        probs = compute_logits(image_vectors,text_vectors).softmax(-1)
        for r,scores,probabilities in zip(sample,cosine,probs):
            pred = labels[int(scores.argmax())]
            prompt_rows.append({'id':r['id'],'strategy':name,'label':r['label'],'prediction':pred,'correct':pred==r['label'], **{f'cosine_{label}':float(score) for label,score in zip(labels,scores)}, **{f'probability_{label}':float(p) for label,p in zip(labels,probabilities)}})
    pd.DataFrame(prompt_rows).to_csv(artifacts/'prompt_results.csv',index=False)
    print(summary.to_string())
    print('Prepared embeddings and measured CSV reports. Classroom inference now needs no internet.')

if __name__ == '__main__':
    main()
