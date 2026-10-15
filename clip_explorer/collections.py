"""Separate curated demonstrations from the larger fixed photographic collection."""
import json
from .config import ROOT
from .data import photographic_gallery


def collection_names():
    return ['Curated gallery']+(['Caltech · 50 × 50'] if (ROOT/'assets/caltech50_manifest.json').exists() else [])


def collection_photos(name):
    if name=='Curated gallery':return photographic_gallery()
    if name!='Caltech · 50 × 50':raise ValueError('Unknown image collection.')
    return json.loads((ROOT/'assets/caltech50_manifest.json').read_text())['gallery']
