"""Extend the validated local image cache with the 2,500-image collection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import numpy as np
from clip_explorer.config import ROOT,CHECKPOINT,MODEL_REVISION
from clip_explorer.collections import collection_photos
from clip_explorer.data import open_asset
from clip_explorer.model import encode_images


def main():
    target=ROOT/'artifacts';meta_path=target/'embedding_metadata.json'
    metadata=json.loads(meta_path.read_text())
    with CHECKPOINT.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if metadata['checkpoint_sha256']!=digest or metadata['model_revision']!=MODEL_REVISION:
        raise ValueError('Prepare the base image cache for the current checkpoint first.')
    with np.load(target/'image_embeddings.npz',allow_pickle=False) as cache:
        values={name:cache[name].copy() for name in cache.files}
    records=collection_photos('Caltech · 50 × 50')
    missing=[r for r in records if r['id'] not in values or metadata['assets'].get(r['id'])!=r['sha256']]
    for start in range(0,len(missing),16):
        chunk=missing[start:start+16]
        encoded=encode_images([open_asset(r) for r in chunk]).numpy()
        for record,vector in zip(chunk,encoded):
            values[record['id']]=vector;metadata['assets'][record['id']]=record['sha256']
        print(f'Encoded {min(start+16,len(missing))}/{len(missing)} collection images',flush=True)
    np.savez_compressed(target/'image_embeddings.npz',**values)
    metadata['additional_collection']='Caltech-101 · 50 categories × 50 images'
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
