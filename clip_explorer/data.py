"""Manifest-backed assets and content-validated prepared embeddings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import torch
from .config import ROOT, CHECKPOINT, MODEL_REVISION


def manifest() -> dict:
    return json.loads((ROOT / 'assets/manifest.json').read_text())


def open_asset(record: dict) -> Image.Image:
    with Image.open(ROOT / 'assets' / record['file']) as image:
        return image.convert('RGB')


@st.cache_data(show_spinner=False)
def _prepared_embeddings(cache_mtime: int, checkpoint_mtime: int) -> tuple[dict[str, np.ndarray], dict]:
    cache = ROOT / 'artifacts/image_embeddings.npz'
    metadata_path = ROOT / 'artifacts/embedding_metadata.json'
    if not cache.exists() or not metadata_path.exists() or not CHECKPOINT.exists():
        return {}, {}
    metadata = json.loads(metadata_path.read_text())
    # Full checkpoint digest binds persisted outputs to these exact model weights.
    with CHECKPOINT.open('rb') as handle:
        digest = hashlib.file_digest(handle,'sha256').hexdigest()
    if metadata.get('checkpoint_sha256') != digest or metadata.get('model_revision') != MODEL_REVISION:
        return {}, {}
    with np.load(cache, allow_pickle=False) as values:
        vectors = {key: values[key].copy() for key in values.files}
    return vectors, metadata


def asset_embeddings(records: list[dict]) -> torch.Tensor:
    """Use prepared vectors only if both checkpoint and current asset bytes match."""
    from .model import encode_images
    cache = ROOT / 'artifacts/image_embeddings.npz'
    vectors, metadata = _prepared_embeddings(cache.stat().st_mtime_ns if cache.exists() else 0, CHECKPOINT.stat().st_mtime_ns if CHECKPOINT.exists() else 0)
    found = []
    missing = []
    for idx, record in enumerate(records):
        digest = hashlib.sha256((ROOT / 'assets' / record['file']).read_bytes()).hexdigest()
        vector = vectors.get(record['id'])
        if vector is not None and metadata.get('assets', {}).get(record['id']) == digest:
            found.append(torch.from_numpy(vector.copy()))
        else:
            found.append(None)
            missing.append(idx)
    if missing:
        encoded = encode_images([open_asset(records[idx]) for idx in missing])
        for idx, vector in zip(missing,encoded): found[idx] = vector
    return torch.stack(found)


def photographic_gallery() -> list[dict]:
    """Real local photographs for the main demonstrations."""
    records=[r for r in manifest()['gallery'] if r.get('medium')=='photograph']
    if not records:
        raise ValueError('Prepare real images with python scripts/download_real_gallery.py.')
    return records


def photo_for_label(label: str) -> dict:
    """Select a documented real photograph with the requested semantic label."""
    preferred={'bicycle':'commons_bicycle_46433264','robot':'commons_robot_18968955'}
    candidates=[r for r in photographic_gallery() if r['label']==label]
    return next((r for r in candidates if r['id']==preferred.get(label)),candidates[0])


def balanced_photos(gallery: list[dict], categories: list[str], per_category: int) -> list[dict]:
    """Spread across labels first, then include additional views of each label."""
    selected=[]
    for category in categories:
        unique,duplicates,seen=[],[],set()
        for record in gallery:
            if record['category']!=category or record.get('medium')!='photograph':continue
            if record['label'] in seen:duplicates.append(record)
            else:unique.append(record);seen.add(record['label'])
        selected.extend((unique+duplicates)[:per_category])
    return selected
